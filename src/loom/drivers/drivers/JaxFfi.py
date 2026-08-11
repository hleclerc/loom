"""Compile C++ kernels into XLA-FFI handlers and expose them to Jax.

Pipeline (see `JaxDriver.call`): a C++ *body* is wrapped into a self-registering XLA FFI
handler, compiled to a shared library with AdaptiveCpp (`make_library`), `dlopen`ed, and
registered with `jax.ffi.register_ffi_target`. The returned target name feeds
`jax.ffi.ffi_call`, which inserts the call into the XLA program (works eager and under
`jax.jit`, on CPU and — later — CUDA).

Two caches, both keyed by a content hash of (source + compilation target):
* disk : the compiled `.so`/`.dylib` (handled by `make_library`; a changed source yields a
         new hash, hence a new file and a rebuild).
* RAM  : `_loaded` keeps the `ctypes` handle mapped and marks the target as already
         registered, so we never `dlopen`/`register_ffi_target` the same handler twice in a
         process (a duplicate registration would raise).

STATUS — this is the minimal bootstrap: a no-argument handler that just prints and returns
a dummy int32 token (the token exists only to keep the call from being dead-code-eliminated
by XLA). Real argument/output binding — driven by `CallArgsAnalysis` / `IoCategory` — and
the separate backward handler for `custom_vjp` come next.
"""
from __future__ import annotations

import ctypes
import sys

import jax
import jax.numpy as jnp
import numpy

from ..compilation.adaptive_cpp import make_library, resolve_targets, ACPP_VERSION
from ..compilation import build_dir
from ..util.encode_base_62 import encode_base_62
from .CallArg_Errors import ERRORS_VAR_NAME

# Fixed C symbol exported by every generated library. It is looked up per-`dlopen`ed handle,
# so the same name across distinct `.so` files never collides; uniqueness at the Jax level is
# carried by the *target name* (the content hash) instead.
_HANDLER_SYMBOL = "sdot_ffi_entry"

# Self-registering handler skeleton. `{body}` is the caller's C++ statements; the trailing
# int32 token write + `Ret<BufferR1<S32>>` binding give XLA a visible result so the call
# survives dead-code elimination.
_SOURCE_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#include <cstdio>
#include <iostream>

namespace ffi = xla::ffi;

static ffi::Error sdot_ffi_impl( ffi::Result<ffi::BufferR1<ffi::S32>> out ) {{
{body}
    out->typed_data()[ 0 ] = 0;
    return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL( sdot_ffi_entry, sdot_ffi_impl,
    ffi::Ffi::Bind().Ret<ffi::BufferR1<ffi::S32>>() );
"""

# RAM cache: target name -> ctypes handle. Presence == "already registered with Jax".
_loaded: dict[ str, ctypes.CDLL ] = {}


def _lib_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _ffi_include_flags() -> list:
    # jaxlib ships the header-only XLA FFI C++ API (xla/ffi/api/ffi.h) under this dir; the shared
    # generated headers (`sdot/generated/...`) live under the build tree, on their own `-I` root.
    from ..compilation.generated_headers import include_root
    return [ "-I", jax.ffi.include_dir(), "-I", str( include_root() ) ]


def render_source( body: str ) -> str:
    """Wrap C++ *body* statements into a complete self-registering FFI handler source."""
    return _SOURCE_TEMPLATE.format( body = body )


def compile_and_register( source: str, device, prefix: str = "" ) -> str:
    """Compile *source*, load and register it, and return its Jax FFI target name.

    Idempotent and cached: repeated calls with the same source + device reuse the compiled
    library and the existing registration.
    """

    if not prefix:
        prefix = "sdot_ffi_"

    # Cache key = what actually changes the binary: the source, and how it is compiled.
    # Dropping the device OBJECT from the key is safe because the source always carries the
    # device anyway (`_CALL_TEMPLATE` substitutes `device.cpp_queue_type`), so a CPU and a CUDA
    # handler can never hash to the same name -- which matters, since this name is also the Jax
    # target and registration is per platform. What we gain: `str( CudaGpu:1 )` used to compile
    # the very same library twice on a two-GPU node, while conversely it never mentioned the
    # compute capability -- so under the ahead-of-time targets a cache shared between machines
    # (a cluster home, a baked container image) could hand an sm_75 binary to an sm_90 card.
    # Naming the target fixes both, and under `generic` the architecture stops being part of
    # the question at all.
    targets, _, _ = resolve_targets( device )
    name = prefix + encode_base_62( f"{ source }|{ targets }|{ ACPP_VERSION }" )
    if name in _loaded:
        return name

    src_path = build_dir() / f"{ name }.cpp"
    src_path.parent.mkdir( parents = True, exist_ok = True )
    src_path.write_text( source )

    lib_path = make_library(
        name + _lib_suffix(), [ src_path ], device,
        extra_flags = _ffi_include_flags(),
    )

    # RTLD_GLOBAL, deliberately: under the `generic` target the kernel is JIT-compiled at run
    # time into a SEPARATE shared library, which the AdaptiveCpp runtime dlopens. Any host
    # symbol that library needs (libstdc++, libm, ...) has to be resolvable in the process's
    # global namespace, and python itself brings none of them. Loading our handler globally
    # publishes them (the flag propagates to its dependencies). With RTLD_LOCAL the JIT'd
    # library fails to load with an `undefined symbol` — at run time, long after compiling fine.
    lib = ctypes.CDLL( str( lib_path ), mode = ctypes.RTLD_GLOBAL )
    handler = getattr( lib, _HANDLER_SYMBOL )
    jax.ffi.register_ffi_target(
        name, jax.ffi.pycapsule( handler ), platform = device.ffi_platform,
    )

    _loaded[ name ] = lib
    return name


def call_body( body: str, device ):
    """Compile a C++ *body* (no arguments yet) and return the result of its FFI call.

    Convenience for the current bootstrap step: renders the source, compiles/registers it,
    and invokes it. The int32 token array is returned as-is.
    """
    target = compile_and_register( render_source( body ), device )
    return jax.ffi.ffi_call( target, jax.ShapeDtypeStruct( ( 1, ), jnp.int32 ) )()


# Full handler skeleton: input buffers are bound as FFI args, output buffers as FFI results,
# and each aggregate arg is materialized as a small `struct` of views over them, so the C++
# body can read and write `cell.<field>`.
_CALL_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#define SDOT_QUEUE {queue_type}
#include "sdot/Queue.h"
#include "sdot/support/algorithms/CartesianIndices.h"
#include "sdot/support/kernels/run_parallel.h"
#include "sdot/support/common_types.h"
#include "sdot/support/Ct.h"
#include "sdot/support/containers/TensorView.h"
#include "sdot/support/containers/ShapeVarView.h"
#include "sdot/support/containers/ErrorBuffer.h"
#include "sdot/support/containers/NoneTensor.h"
#include "sdot/support/containers/ZeroTensor.h"
#include "sdot/support/containers/FillTensor.h"
#include <cstdint>
#include <iostream>

namespace ffi = xla::ffi;
using namespace sdot;

// the axes this call names, from their shared generated headers (`DEFINE_AXIS`), so anything
// below can spell them.
{axis_includes}

// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
{extra_includes}

static ffi::Error sdot_ffi_impl( {params} ) {{
    // the one execution context of this call. A `sycl::queue` is expensive to create, and this
    // handler always runs on the same device (the device is in the TYPE of everything below), so
    // there is exactly one, made on first use -- and never destroyed: the SYCL runtime is torn
    // down before the statics of a dlopen'ed handler are, and a queue outliving it deadlocks.
    static Queue &queue = *new Queue();

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
{batch_indices}

{decls}
{seeds}
    {{
{body}
    }}
    return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL( sdot_ffi_entry, sdot_ffi_impl,
    ffi::Ffi::Bind(){binds} );
"""


def _batch_indices_decl( ca ):
    """`global_batch_indices`: the multi-indices the body iterates over.

    Unmapped, `CartesianIndices<Tuple<>>` -- one item, the empty multi-index. A `vmap` gives it a
    NAMED axis, whose extent is read from a buffer at run time (a batch size is an extent like any
    other: making it a literal would recompile the kernel for every batch size)."""
    if not ca.batch_axes:
        return "    CartesianIndices<Tuple<>> global_batch_indices;"
    shape = ", ".join( "SI" for _ in ca.batch_axes )
    names = ", ".join( "_" + n for n in ca.batch_axes )
    sizes = ", ".join( ca.batch_dim_expr( n ) for n in ca.batch_axes )
    return ( f"    CartesianIndices<Tuple<{ shape }>,Tuple<{ names }>> "
             f"global_batch_indices{{ tuple( { sizes } ) }};" )


def _render_call( code, ca, device ):
    """The complete FFI handler source for this code bound to these buffers, and the attributes
    it expects.

    Inputs and outputs are disjoint buffers: an input is bound at the size its data actually
    has, an output is allocated at the capacity declared in Python. XLA FFI wants args before
    results, so the parameter list is inputs then outputs, `ca.tensors` fixing the order within
    each group.

    A root argument declares itself (`cpp_root_decl`): an aggregate as a `struct` of views over
    its buffers -- a *template*, hence the dedup of definitions by `type_name`, since the same
    class may appear twice in a call with different compile-time parameters -- and a bare
    tensor as the view itself, no wrapper needed.
    """
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    # scalars that are neither data nor structure -- a capacity bound, a bare `int` argument. They
    # cross as XLA FFI ATTRIBUTES: baked into the call, not into the kernel, so a new value does
    # not mean a new compilation. (Extents need no attribute at all: XLA carries them next to the
    # data.) Gathered by folding the node tree, each node answering for itself.
    attrs = [ a for n in ca.nodes() if hasattr( n, "jax_attrs" ) for a in n.jax_attrs() ]

    # the headers the arguments ask for (an aggregate names its struct header), then whatever the
    # body itself listed. Collected blind: the call never knows which node brought a header, nor
    # that any of it is hand-written or generated behind the scenes.
    includes = []
    for inc in [ i for arg_ca in ca.args.values() for i in arg_ca.cpp_includes() ] + list( code.includes ):
        if inc not in includes:
            includes.append( inc )

    # XLA FFI binds in this order, and the handler's parameters must follow it: args, results,
    # then attributes.
    params = [ f"{ b.jax_ffi_type() } { b.ffi_name }" for b in inputs ]
    params += [ f"ffi::Result<{ b.jax_ffi_type() }> { b.ffi_name }" for b in outputs ]
    params += [ f"{ cpp_type } { name }" for name, cpp_type, _ in attrs ]

    binds = "".join( f"\n        .Arg<{ b.jax_ffi_type() }>()" for b in inputs )
    binds += "".join( f"\n        .Ret<{ b.jax_ffi_type() }>()" for b in outputs )
    binds += "".join( f'\n        .Attr<{ cpp_type }>( "{ name }" )' for name, cpp_type, _ in attrs )

    # the error buffer comes FIRST: the values that can fail are built holding a view on it.
    decls = [ ca.errors.cpp_root_decl( ERRORS_VAR_NAME ) ]
    decls += [ ca_.cpp_root_decl( n ) for n, ca_ in ca.args.items() ]

    seeds = [ ca.errors.cpp_seed_root( ERRORS_VAR_NAME ) ]
    seeds += [ ca_.cpp_seed_root( n ) for n, ca_ in ca.args.items() if hasattr( ca_, "cpp_seed_root" ) ]

    from ..tensor.AbstractAxis import AbstractAxis
    source = _CALL_TEMPLATE.format(
        queue_type    = device.cpp_queue_type,
        extra_includes = "".join( f'#include "{ inc }"\n' for inc in includes ),
        axis_includes = "".join( f'#include "{ AbstractAxis.cpp_shared_header( n ) }"\n'
                                 for n in ca.axis_names ),
        params        = ", ".join( params ),
        batch_indices = _batch_indices_decl( ca ),
        decls         = "\n".join( decls ),
        seeds         = "\n".join( s for s in seeds if s ),
        body          = code.code_for( "fwd", ca ),
        binds         = binds,
    )
    return source, inputs, outputs, attrs


def _make_op( code, ca, device, prefix ):
    """The call as a Jax operation -- with its own batching rule.

    A `vmap` cannot batch an FFI call by itself (it could only replay it item by item, or
    broadcast everything). Ours does the one thing that makes sense here: it RECOMPILES. The rule
    derives the code (one more batch axis) and the lowering (the buffers that gained a leading
    dimension), and calls the kernel that comes out -- one launch over N items, not N launches.

    The derived call is an op of the same kind, so a nested `vmap` just derives again.
    """
    @jax.custom_batching.custom_vmap
    def op( *arrays ):
        source, _, outputs, attrs = _render_call( code, ca, device )
        target = compile_and_register( source, device, prefix )
        results = jax.ffi.ffi_call( target, [ b.jax_out_spec() for b in outputs ] )(
            *arrays, **{ name: numpy.int64( value ) for name, _, value in attrs }
        )
        return list( results ) if isinstance( results, ( list, tuple ) ) else [ results ]

    @op.def_vmap
    def _( axis_size, in_batched, *arrays ):
        # `arrays` come with the mapped axis leading, and `in_batched` says which ones the vmap
        # actually mapped -- an unmapped input keeps its shape, and the kernel will let the batch
        # index pass through it rather than read a slice of it.
        inputs = [ t for t in ca.tensors if t.io_category.is_input ]
        batched_inputs = { t.ffi_name for t, mapped in zip( inputs, in_batched ) if mapped }

        axis_name, batched_code = code.with_batch_axis()
        batched_ca = ca.batched( axis_name, axis_size, batched_inputs )

        results = _make_op( batched_code, batched_ca, device, prefix )( *arrays )

        # ... and one output per item, save what belongs to the CALL rather than to an item: the
        # error buffer is one, and comes back unbatched.
        outputs = [ t for t in ca.tensors if t.io_category.is_output ]
        return results, [ t.takes_batch_axis() for t in outputs ]

    return op


def _run( code, ca, device, prefix ):
    """Run `code` on the buffers of `ca` and return `( output CallArgs, result arrays )`, WITHOUT
    writing anything back. The caller decides what the results are: outputs to rebind onto Python
    objects (a forward call), or cotangents to hand back to Jax (a backward call)."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    # the kernel dereferences its buffers where IT runs, so an input has to be there: an array
    # built on the host would otherwise be read through a device pointer.
    arrays = [ jax.device_put( b.jax_input_array(), device.driver_version ) for b in inputs ]

    results = _make_op( code, ca, device, prefix )( *arrays )
    return outputs, list( results ) if isinstance( results, ( list, tuple ) ) else [ results ]


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects
    the caller handed us.

    When `code` has a backward, the call is made DIFFERENTIABLE: Jax is given a VJP rule
    (`jax.custom_vjp`) whose backward is itself an ordinary kernel call (see `_call_backward`)."""
    if code.has_code_for( "bwd" ):
        outputs, results = _call_with_vjp( code, ca, device, prefix )
    else:
        outputs, results = _run( code, ca, device, prefix )

    # an output attribute was EMPTY (that is what made it declarable as one), so filling it in
    # is not a mutation of anything the caller could already have observed. Under a `vmap` these
    # are the OUTER values (batch axis stripped by Jax), which is why the batched lowering had to
    # be a copy: this one still describes the tensors as the caller knows them.
    for buffer, array in zip( outputs, results ):
        buffer.jax_write_back( array )


def _call_with_vjp( code, ca, device, prefix ):
    """The forward call, wrapped in a `jax.custom_vjp` so `jax.grad`/`jax.vjp` reach the backward
    kernel. Returns the same `( outputs, results )` as `_run`, so the write-back is common.

    ALL inputs -- float and integer alike -- cross as real elements of `op`'s argument tuple,
    threaded through Jax's own tracing machinery, rather than split into "differentiable primals
    passed as arguments" + "everything else closed over as Python constants" (the previous
    design). The closure form is only safe if `op_fwd`/`op_bwd` are invoked in the very trace that
    built the closed-over values -- true under a bare `jit`/`grad`/`vmap`, but NOT when the call
    sits inside a `lax.scan` body that is later differentiated: scan's differentiation rule
    re-invokes `op_fwd`/`op_bwd` in a separate, nested trace to linearize/transpose the body, and
    by then any closed-over tracer belongs to an already-exited trace -> `UnexpectedTracerError`.
    Threading everything through the real argument list sidesteps this: Jax re-binds every
    argument fresh for whatever trace context replays `op_fwd`/`op_bwd`.

    `symbolic_zeros = True` gives us the two facts the backward needs to stay cheap: which inputs
    Jax actually wants a gradient for (`perturbed`), and which output cotangents are structurally
    zero (a `SymbolicZero`). An integer input is forced non-perturbed regardless of what Jax
    reports (a mesh of indices, a count is never differentiated -- its tangent space is trivial)."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    in_arrays = tuple( jax.device_put( t.jax_input_array(), device.driver_version ) for t in inputs )

    fwd_op = _make_op( code, ca, device, prefix )

    @jax.custom_vjp
    def op( values ):
        return tuple( fwd_op( *values ) )

    def op_fwd( values ):
        # symbolic_zeros wraps each primal in `CustomVJPPrimal( value, perturbed )`.
        perturbed = tuple( getattr( v, "perturbed", True ) and t.dtype.floating_point
                           for v, t in zip( values, inputs ) )
        full_in = tuple( getattr( v, "value", v ) for v in values )
        outs = tuple( fwd_op( *full_in ) )
        return outs, ( full_in, outs, perturbed )

    def op_bwd( residuals, cotangents ):
        full_in, out_values, perturbed = residuals
        grads = _call_backward( code, ca, device, prefix, inputs, outputs,
                                full_in, out_values, perturbed, cotangents )
        return ( grads, )

    op.defvjp( op_fwd, op_bwd, symbolic_zeros = True )

    results = op( in_arrays )
    return outputs, list( results )


def _grad_tensor( inst, array ):
    """A bare tensor holding `array`, shaped like `inst` -- a residual (a forward input/output) or
    a cotangent, entering the backward kernel as an input bound at the size its data has."""
    from ..tensor.Tensor import Tensor
    res = Tensor.like( inst )
    res.set_raw( array )
    # a FILL re-enters the backward as a fill too (its residual is the same scalar): keep it symbolic
    # so it lowers to a `FillTensor` again, not a scalar-buffer TensorView with a [n] logical shape.
    if getattr( inst, "is_fill", False ):
        res._fill  = True
        res._shape = inst._shape
    return res


def _grad_shapevar( inst, raw ):
    """A `ShapeVar`-shaped object carrying a FIXED count `raw` -- the residual for a `ShapeVar`
    member entering the backward, mirroring `_grad_tensor` for a `Tensor` member.

    Needed because `ShapeVar._count` is, by its own contract, "a count produced by a kernel: a
    driver tensor, POSSIBLY TRACED" (`tensor/ShapeVar.py`). Reusing the live, shared `inst` object
    directly (as the backward used to, for every ShapeVar/Axis/CtShapeVar member alike) reads
    whatever `_count` holds AT THE MOMENT `op_bwd` executes -- safe only when that is the very
    trace that resolved it. Under `lax.scan`'s differentiation, `op_bwd` is replayed in a later,
    separate trace, so a resolved-but-still-tracer count from the original trace is dead by then.
    `raw` here is instead the count as it flowed through `driver.call`'s own residual channel
    (`full_in`/`out_values`), which Jax DOES keep valid across that replay."""
    from ..tensor.ShapeVar import ShapeVar
    res = ShapeVar.__new__( ShapeVar )
    res.usages = []
    res.dep_axes = inst.dep_axes
    res.prescribed_value = None
    res._count = raw
    return res


def _call_backward( code, ca, device, prefix, inputs, outputs,
                    full_in, out_values, perturbed, cotangents ):
    """The backward pass, expressed as an ORDINARY kernel call whose body is the code's backward.

    Each forward argument `X` yields two backward arguments, of the SAME type as `X` (a bare
    tensor, or an aggregate mirrored member by member):

    * a RESIDUAL `X`: the forward values, re-entering as backward INPUTS under the very name they
      had -- so the body reads `cell.vertex_positions`, `inp`, ... exactly as the forward did;
    * a gradient `grad_for_X`, whose tensors are, per member:
        - a float forward OUTPUT   -> its cotangent, a backward INPUT (a `SymbolicZero` lowers to a
          `ZeroTensor`: read as 0, no buffer, dropped at compile time);
        - a float forward INPUT     -> a backward OUTPUT when perturbed, else a `NoneTensor` (the
          body skips it at compile time, `grad_for_...is_valid()` being false);
        - anything else             -> a `NoneTensor`.

    An aggregate `grad_for_cell` thus carries a MIX of backward-input and backward-output members;
    the per-member io policy already handles that (see `CallArg_Aggregate`). Non-tensor members
    (`Axis`, `ShapeVar`, `CtShapeVar`) are SHARED from the primal, so a gradient buffer resolves
    its capacity from the forward tensor it mirrors.

    Returns the tuple of cotangents, one per input, in `inputs` order -- `None` (Jax's own
    symbolic-zero marker for a `custom_vjp` bwd output) wherever no gradient is wanted, which
    covers both non-float inputs and non-perturbed float ones uniformly.
    """
    from ..tensor.Tensor import Tensor
    from .CallArgsAnalysis import CallArgsAnalysis
    from ..util.annotations import annotations
    from ..util.Aggregate import Aggregate, get_attribute

    # leaf-indexed facts (by tensor identity), so the structural walk below can consult them.
    io_of, residual_of = {}, {}
    for k, t in enumerate( inputs ):
        if hasattr( t, "inst" ):
            io_of[ id( t.inst ) ], residual_of[ id( t.inst ) ] = "input", full_in[ k ]
    for j, t in enumerate( outputs ):
        if hasattr( t, "inst" ):
            io_of[ id( t.inst ) ], residual_of[ id( t.inst ) ] = "output", out_values[ j ]
    cotangent_of = { id( t.inst ): cotangents[ j ]
                     for j, t in enumerate( outputs ) if hasattr( t, "inst" ) }
    perturbed_of = { id( t.inst ): perturbed[ k ] for k, t in enumerate( inputs ) if hasattr( t, "inst" ) }

    output_paths = []
    grad_obj_of = {}   # id( primal input leaf ) -> its gradient tensor (a backward output)

    def _is_agg( obj ):
        return isinstance( obj, Aggregate )

    def _blank( inst ):
        obj = type( inst ).__new__( type( inst ) )
        obj.name = getattr( inst, "name", None )   # only a NESTED aggregate carries a field name
        # carry the batch axes over: the backward is an ordinary call, so `CallArgsAnalysis` must see
        # them on the residual/grad aggregates to build `global_batch_indices` and let the body's
        # `plan( batch_index )` squeeze the batch (without this the backward would run unbatched).
        obj.batch_axes = list( getattr( inst, "batch_axes", [] ) )
        return obj

    def _build( inst, path ):
        """`( residual, grad )` mirroring `inst` (a tensor or a whole aggregate subtree)."""
        if _is_agg( inst ):
            residual, grad = _blank( inst ), _blank( inst )
            for mname in annotations( type( inst ) ):
                member = get_attribute( mname, inst )
                r, g = _build( member, f"{ path }.{ mname }" )
                residual.__dict__[ mname ], grad.__dict__[ mname ] = r, g
            return residual, grad

        if not isinstance( inst, Tensor ):
            from ..tensor.ShapeVar import ShapeVar
            if isinstance( inst, ShapeVar ):
                # a data-dependent ShapeVar (its count came from a kernel, via `driver.call`'s own
                # input/output tracking) reuses the properly-threaded residual value; a purely
                # static one (never bound as an FFI buffer -- `residual_of` has nothing for it)
                # falls through to the plain shared-object case below, same as Axis/CtShapeVar.
                raw = residual_of.get( id( inst ) )
                if raw is not None:
                    shared = _grad_shapevar( inst, raw )
                    return shared, shared
            return inst, inst   # Axis / CtShapeVar (or a static ShapeVar): shared, so shapes
                                 # resolve -- this holds whether `inst` is a nested aggregate
                                 # member OR a bare top-level kwarg (e.g. `Cell.measure`'s
                                 # `nb_map_items`)

        # a tensor leaf: the residual is bound to whatever forward value it held.
        arr = residual_of.get( id( inst ) )
        residual = _grad_tensor( inst, arr ) if arr is not None else Tensor.like( inst )

        io = io_of.get( id( inst ) )
        if io == "output" and inst.dtype.floating_point:
            # the cotangent enters as a backward INPUT -- a real buffer (a `TensorView`) or the
            # framework's symbolic zero (a `ZeroTensor`): both just get stored, `_grad_tensor` /
            # `is_symbolic_zero` tell them apart, no special case here.
            grad = _grad_tensor( inst, cotangent_of.get( id( inst ) ) )
        elif io == "input" and inst.dtype.floating_point and perturbed_of.get( id( inst ), False ):
            grad = Tensor.like( inst )
            output_paths.append( path )
            grad_obj_of[ id( inst ) ] = grad
        else:
            grad = Tensor.like( inst )   # non-differentiable or non-perturbed -> a NoneTensor
        return residual, grad

    kwargs = {}
    for name, arg in ca.args.items():
        if not hasattr( arg, "inst" ):
            continue
        # SCRATCH: the backward gets a FRESH writable buffer under the same name (an output of the
        # backward call), NOT the forward's transient per-thread values as a residual. The body
        # re-derives into it whatever it needs (a re-sort). Capacity resolves on its own -- the
        # thread axis is a `CtShapeVar` (static), the item axis is shared with a residual it mirrors.
        if name in ca.scratch_paths:
            kwargs[ name ] = Tensor.like( arg.inst )
            output_paths.append( name )
            continue
        residual, grad = _build( arg.inst, "grad_for_" + name )
        kwargs[ name ] = residual
        kwargs[ "grad_for_" + name ] = grad

    # the backward runs as an ordinary forward whose body is our backward one -- of the same kind,
    # so a `FfiCodeParallel` scaffolds it over the residual+gradient arguments just as it did the
    # forward over the primal ones.
    bwd_code = code.for_backward()
    bwd_ca = CallArgsAnalysis( kwargs, device, output_attributes = output_paths )
    bwd_outputs, bwd_results = _run( bwd_code, bwd_ca, device, prefix + "bwd_" )

    result_of = { id( o.inst ): r for o, r in zip( bwd_outputs, bwd_results ) if hasattr( o, "inst" ) }

    grads = []
    for t in inputs:
        gobj = grad_obj_of.get( id( t.inst ) ) if hasattr( t, "inst" ) else None
        if gobj is not None and id( gobj ) in result_of:
            grads.append( result_of[ id( gobj ) ] )
        else:
            # non-float, or a non-perturbed float primal: Jax's own symbolic-zero handling for a
            # `custom_vjp` bwd converts a bare `None` leaf into the right zero cotangent -- no
            # `float0`/dtype ceremony needed, and it is well-defined for an integer primal too.
            grads.append( None )
    return tuple( grads )
