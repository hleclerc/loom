"""The jax-free half of a kernel call: what the CALL says, as C++.

`_render_call` turns a `FfiCode` and the buffers of a `CallArgsAnalysis` into the source of an
XLA-FFI-shaped handler. Nothing in it needs jax -- only the XLA FFI *API* (`ffi::BufferR2<ffi::F64>`,
`ffi::Result`, `Bind()`) is spelled in the generated text. `JaxFfi` compiles it against jaxlib's
header; `NumpyFfi` against a header of our own that gives the same spelling to plain numpy buffers
(`loom/support/numpy_ffi`). One generator, two ABIs.
"""
from __future__ import annotations
from pathlib import Path

from .CallArg_Errors import ERRORS_VAR_NAME

def call_signature( ca ):
    """What describes the CALL, and not the source: enough to answer "why is this kernel
    new?" (see `compilation/journal.py`).

    An argument enters it through its LOWERING FORM -- a buffer, a `NoneTensor` (unperturbed),
    a `ZeroTensor` (symbolically zero cotangent), a `FillTensor` -- plus its type and the
    extents the type carries. This is exactly the axis along which a backward multiplies."""
    # A node is described by WHAT IT HAS: `ca.tensors` mixes tensors, counts
    # (`CallArg_ShapeVar`: an `inst`, but a `ShapeVar` -- neither storage flags nor type extents)
    # and the call's error buffer (neither of the two). It is a report: it describes what it
    # finds and demands nothing.
    def describe( t ):
        if not t.io_category.is_bound:
            return "none"
        inst = getattr( t, "inst", None )
        if getattr( inst, "is_symbolic_zero", False ):
            return "zero"
        if getattr( inst, "is_fill", False ):
            return "fill"

        res = "out" if t.io_category.is_output else "in"
        dtype = getattr( t, "dtype", None )
        if dtype is not None:
            res += f" { dtype.cpp_name }"
        extents = getattr( t, "_dim_ct_extent", None )
        if extents:
            res += "[" + ",".join( "*" if e is None else str( e ) for e in extents ) + "]"
        return res

    res = { t.name: describe( t ) for t in ca.tensors }
    if ca.batch_axes:
        res[ "<batch>" ] = ",".join( ca.batch_axes )
    return res


# Full handler skeleton: input buffers are bound as FFI args, output buffers as FFI results,
# and each aggregate arg is materialized as a small `struct` of views over them, so the C++
# body can read and write `cell.<field>`.
_CALL_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#include <{queue_include}>
// the device is in the TYPE of everything below: the queue decides the memory space the kernel
// dereferences. `SDOT_QUEUE` is also what a hand-written header may read (`sdot/Queue.h`).
#define SDOT_QUEUE {queue_type}
namespace sdot {{ using Queue = SDOT_QUEUE; }}
#include <loom/support/algorithms/CartesianIndices.h>
#include <loom/support/kernels/run_parallel.h>
#include <loom/support/common_types.h>
#include <loom/support/Ct.h>
#include <loom/support/containers/TensorView.h>
#include <loom/support/containers/ShapeVarView.h>
#include <loom/support/containers/ErrorBuffer.h>
#include <loom/support/containers/NoneTensor.h>
#include <loom/support/containers/ZeroTensor.h>
#include <loom/support/containers/FillTensor.h>
#include <loom/support/containers/ScalarValue.h>
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

// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
{preamble}

static ffi::Error sdot_ffi_impl( {params} ) {{
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    {queue_decl}

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

// the ONE exported symbol (everything else is hidden, see `HostCxx`): the declaration carries the
// visibility, the macro below defines it. Its NAME is a define, not part of the source (the source
// is hashed into the kernel's name): `sdot_ffi_entry` in a library of its own, a unique name when a
// catalogue links many kernels into one library (see `compilation/catalogue.py`).
#ifndef SDOT_FFI_ENTRY
#define SDOT_FFI_ENTRY sdot_ffi_entry
#endif
extern "C" LOOM_EXPORT XLA_FFI_Error *SDOT_FFI_ENTRY( XLA_FFI_CallFrame * );
XLA_FFI_DEFINE_HANDLER_SYMBOL( SDOT_FFI_ENTRY, sdot_ffi_impl,
    ffi::Ffi::Bind(){binds} );
"""


def _batch_indices_decl( ca ):
    """`global_batch_indices`: the multi-indices the body iterates over.

    Unmapped, `CartesianIndices<Tuple<>>` -- one item, the empty multi-index. A `vmap` gives it a
    NAMED axis, whose extent is read from a buffer at run time (a batch size is an extent like any
    other: making it a literal would recompile the kernel for every batch size).

    That extent crosses BY VALUE, as an FFI attribute -- see `CallArgsAnalysis.batch_dim_expr`.
    Reading it off a buffer is what this used to do, and it was wrong: a leading dimension is a
    padded CAPACITY, so the body iterated over 32 slots for 20 cells, read the count buffer past
    its end, and let the garbage bound a loop over the geometry."""
    if not ca.batch_axes:
        return "    CartesianIndices<Tuple<>> global_batch_indices;"
    shape = ", ".join( "SI" for _ in ca.batch_axes )
    names = ", ".join( "_" + n for n in ca.batch_axes )
    sizes = ", ".join( ca.batch_dim_expr( n ) for n in ca.batch_axes )
    return ( f"    CartesianIndices<Tuple<{ shape }>,Tuple<{ names }>> "
             f"global_batch_indices{{ tuple( { sizes } ) }};" )


def _render_call( code, ca, device ):
    """`_render_source`, plus the generated headers the rendering asked for ( `{ rel_path: content }` ):
    the kernel compiles against THOSE, from an include overlay of its own, never against the shared
    tree another call may have rewritten meanwhile (see `compilation/generated_headers.py`).

    Remembered: a call that repeats an earlier one -- same code, same shapes, other numbers -- gets the
    same text, so it is not rendered again (see `render_key.py`, which says when that is certain)."""
    from . import render_key
    if not render_key.enabled():
        return _render_uncached( code, ca, device )

    key = render_key.render_key( code, ca, device )
    if key is None:
        return _render_uncached( code, ca, device )

    hit = render_key.lookup( key )
    if hit is None:
        render_key.stats[ "miss" ] += 1
        res = _render_uncached( code, ca, device )
        source, _, _, _, sources, headers = res
        render_key.store( key, ( source, sources, headers ) )
        return res

    render_key.stats[ "hit" ] += 1
    source, sources, headers = hit
    # what rendering would have done on the way: the call's headers are declared, so that the shared
    # tree and any collector around us see them
    from ..compilation.generated_headers import shared_header
    for rel_path, content in headers.items():
        shared_header( rel_path, content )
    inputs, outputs = _call_buffers( ca )
    res = ( source, inputs, outputs, _call_attrs( ca ), sources, dict( headers ) )
    if render_key.verifying():
        fresh = _render_uncached( code, ca, device )
        _check_same( fresh, res, code )
    return res


def _check_same( fresh, cached, code ):
    names = ( "source", "inputs", "outputs", "attrs", "sources", "headers" )
    for name, a, b in zip( names, fresh, cached ):
        if name in ( "inputs", "outputs" ):
            same = [ id( x ) for x in a ] == [ id( x ) for x in b ]
        elif name == "attrs":
            same = [ tuple( x ) for x in a ] == [ tuple( x ) for x in b ]
        else:
            same = a == b
        if not same:
            raise RuntimeError( f"loom: the render cache gave a different `{ name }` for the call `{ code.name }` "
                                f"than rendering it again -- the key of `render_key.py` misses something the "
                                f"rendering reads" + ( "\n" + _first_difference( a, b ) if name == "source" else "" ) )


def _first_difference( a, b ):
    import difflib
    return "\n".join( list( difflib.unified_diff( a.splitlines(), b.splitlines(), "fresh", "cached", lineterm = "", n = 1 ) )[ :20 ] )


def _render_uncached( code, ca, device ):
    from ..compilation.generated_headers import collecting_headers
    with collecting_headers() as headers:
        res = _render_source( code, ca, device )
    return ( *res, headers )


def _call_buffers( ca ):
    """The FFI buffers of the call, inputs then outputs -- which is the order XLA binds them in."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]
    return inputs, outputs


def _call_attrs( ca ):
    """The scalars that are neither data nor structure -- a capacity bound, a bare `int` argument. They
    cross as XLA FFI ATTRIBUTES: baked into the call, not into the kernel, so a new value does not
    mean a new compilation. ( Extents need no attribute at all: XLA carries them next to the data.)
    Gathered by folding the node tree, each node answering for itself -- and NOT part of what the
    render cache remembers: they carry the VALUES of this call."""
    # imported here and not at the top: `JaxFfi` <-> `CallArgsAnalysis` depend on each other
    # (see the import of `CallArgsAnalysis` further down, already local for the same reason).
    from .CallArgsAnalysis import batch_attr_name

    attrs = [ a for n in ca.nodes() if hasattr( n, "jax_attrs" ) for a in n.jax_attrs() ]
    # the extent of each batch axis, BY VALUE: a plain count, of rank 0, read-only
    # and known to the host -- exactly what `CallArg_ShapeVar.as_scalar` sends across as an
    # attribute. A batch axis is not a special case on this point.
    attrs += [ ( batch_attr_name( n ), "int64_t", ca.batch_axis_size( n ) ) for n in ca.batch_axes ]
    return attrs


def _render_source( code, ca, device ):
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
    inputs, outputs = _call_buffers( ca )
    attrs = _call_attrs( ca )

    # the headers the arguments ask for (an aggregate names its struct header), then whatever the
    # body itself listed. Collected blind: the call never knows which node brought a header, nor
    # that any of it is hand-written or generated behind the scenes.
    includes = []
    if code.wants_allocator:
        includes.append( "loom/support/kernels/Scratch.h" )
    for inc in [ i for arg_ca in ca.args.values() for i in arg_ca.cpp_includes() ] + list( code.includes ):
        if inc not in includes:
            includes.append( inc )

    # the units to LINK, collected the same blind way: `( path, defines )`, resolved against the
    # C++ source roots. Compiled once per (source, defines, compiler) by the build graph.
    # kept AS GIVEN (`sdot/x.cpp`, relative to the C++ roots): the path is part of the kernel's
    # key, and a catalogue key must be the same on every machine. Resolved at compile time.
    sources = []
    for src in [ s for arg_ca in ca.args.values() for s in getattr( arg_ca, "cpp_sources", lambda: () )() ] + list( code.sources ):
        path, defines = ( src if isinstance( src, tuple ) else ( src, {} ) )
        item = ( str( path ), tuple( sorted( dict( defines ).items() ) ) )
        if item not in sources:
            sources.append( item )

    # XLA FFI binds in this order, and the handler's parameters must follow it: the platform
    # stream if the device runs on one (CUDA: the queue is BUILT on XLA's stream, so the call is
    # ordered with the rest of the program by the stream itself, see `CudaQueue.h`), then args,
    # results, and attributes.
    stream_param = device.cpp_stream_param()
    params = [ stream_param ] if stream_param else []

    # XLA's allocator, ON REQUEST ( `FfiCode( allocator = True )` ). It is bound as a `Ctx`, so
    # its place in the signature is that of its `Bind()` clause -- right after the stream, before
    # the arguments. And it is added ONLY if the body asked for it: a kernel's name is the hash
    # of its source, so a clause added unconditionally would recompile the whole repository.
    scratch = code.wants_allocator
    if scratch and device.cpp_scratch_param():
        params.append( device.cpp_scratch_param() )

    params += [ f"{ b.jax_ffi_type() } { b.ffi_name }" for b in inputs ]
    params += [ f"ffi::Result<{ b.jax_ffi_type() }> { b.ffi_name }" for b in outputs ]
    params += [ f"{ cpp_type } { name }" for name, cpp_type, _ in attrs ]

    binds = ( "\n        " + device.cpp_stream_bind() ) if stream_param else ""
    if scratch and device.cpp_scratch_bind():
        binds += "\n        " + device.cpp_scratch_bind()
    binds += "".join( f"\n        .Arg<{ b.jax_ffi_type() }>()" for b in inputs )
    binds += "".join( f"\n        .Ret<{ b.jax_ffi_type() }>()" for b in outputs )
    binds += "".join( f'\n        .Attr<{ cpp_type }>( "{ name }" )' for name, cpp_type, _ in attrs )

    # the error buffer comes FIRST: the values that can fail are built holding a view on it.
    decls = [ ca.errors.cpp_root_decl( ERRORS_VAR_NAME ) ]
    decls += [ ca_.cpp_root_decl( n ) for n, ca_ in ca.args.items() ]

    seeds = [ ca.errors.cpp_seed_root( ERRORS_VAR_NAME ) ]
    seeds += [ ca_.cpp_seed_root( n ) for n, ca_ in ca.args.items() if hasattr( ca_, "cpp_seed_root" ) ]

    # what gives the body the allocator, and what reports a refusal. XLA's pool can say
    # no, and `Scratch` turns that into a VALUE ( an empty view, a flag ) rather than an exception:
    # the body dereferences nothing, and it is here that the refusal becomes an error for XLA again.
    queue_decl = device.cpp_queue_decl()
    body = code.code_for( ca )

    # THE ASSEMBLED ARGUMENTS -- two structs, because they do not live in the same place.
    #
    # `<name>_args` is HOST side: it carries `queue`, `machine`, `errors`, the call's arguments and
    # the io policy that Python deduced for each one ( `<name>_io` ). It is what the user's
    # `kernel` function receives.
    #
    # `<name>_kargs` is what crosses over to the KERNEL: same data, retyped into the kernel's
    # memory zone, without the `queue` ( which cannot cross ) and without the policies ( their
    # job is done ). It is what the functor receives. The crossing is a `kernel_form`, like
    # for any aggregate -- so `run_parallel` has nothing special to know.
    #
    # Emitted for the general form only: a `per_item` kernel gives back an unchanged source.
    args_struct = ""
    if code.is_handler:
        names = list( ca.args )
        kargs, hargs = code.args_name() + "_k", code.args_name()

        fields_k = [ ( "machine", "machine" ), ( "errors", ERRORS_VAR_NAME ) ]
        fields_k += [ ( n, n ) for n in names ]
        tp_k = ", ".join( f"class T_{ n }" for n, _ in fields_k )
        decl_k = "".join( f"    T_{ n } { n };\n" for n, _ in fields_k )

        fields_h = [ ( "queue", "queue", True ), ( "machine", "queue.machine()", False ),
                     ( "errors", ERRORS_VAR_NAME, False ) ]
        # THE ALLOCATOR, on request ( `FfiCode( allocator = True )` ): allocating is a
        # HOST operation, so it lives in `args` and not in the kernel form.
        #
        # It used to be called `args.scratch`, which collided head-on with the `args.scratch` group
        # -- a call's work buffers. Two different things bore the same name: this one ALLOCATES
        # during the call, those are already allocated. The name now says which is which.
        if scratch:
            fields_h.append( ( "allocator", "scratch", True ) )
        for n in names:
            fields_h.append( ( n, n, False ) )
            fields_h.append( ( f"{ n }_io", ca.args[ n ].cpp_io_expr(), False ) )
        tp_h = ", ".join( f"class T_{ n }" for n, _, _ in fields_h )
        decl_h = "".join( f"    T_{ n } { '&' if r else '' }{ n };\n" for n, _, r in fields_h )

        # what crosses over: each argument with ITS policy, `errors` as MutList ( the kernel writes to it )
        passed = [ "machine", f"sdot::kernel_form( q, MutList(), { ERRORS_VAR_NAME } )" ]
        passed += [ f"sdot::kernel_form( q, { n }_io, { n } )" for n in names ]

        # `TF`: the scalar of the call's reals ( what `RealTensor` produces when nothing is
        # specified, hence `loom.resolved_dtype()` ). A body writes it without having to derive it from a view.
        # `cpp_name` gives the ALIAS `TF` as long as the size is left to the driver; here we want
        # the resolved type, otherwise we would write `using TF = TF;`.
        from . import framework_defaults
        tf = ( "\n/// the real scalar of this call ( the default real type )\n"
               f"using TF = { framework_defaults.ftype().cpp_name };\n" )

        args_struct = ( tf +
            f"\n// what crosses over to the kernel ( see `FfiCode` )\n"
            f"template<{ tp_k }>\nstruct { kargs } {{\n{ decl_k }}};\n"
            f"\n// the arguments of this call, assembled, host side\n"
            f"template<{ tp_h }>\nstruct { hargs } {{\n{ decl_h }\n"
            f"    auto kernel_form( auto &&q, auto ) const {{\n"
            f"        return { kargs }{{ " + ", ".join( passed ) + " };\n    }\n};\n" )

        body = ( f"    { hargs } args{{ " + ", ".join( e for _, e, _ in fields_h ) + " };\n" ) + body

    if scratch:
        queue_decl += "\n    " + device.cpp_scratch_decl()
        # the WHY is lost along the way: `ScratchAllocator::Allocate` emits the reason into a
        # `DiagnosticEngine` that `Handler::Call` only consults on a DECODING failure, and returns
        # `nullopt` without it. The message is therefore `Scratch`'s: the size refused, what the call
        # had taken, and what the body said of itself ( `scratch.why` ).
        body += ( '\n        if ( scratch.failed )\n'
                  '            return ffi::Error( ffi::ErrorCode::kResourceExhausted, scratch.refusal_message() );' )

    from ..tensor.AbstractAxis import AbstractAxis
    source = _CALL_TEMPLATE.format(
        queue_decl    = queue_decl,
        queue_include = device.cpp_queue_include,
        queue_type    = device.cpp_queue_type,
        preamble      = args_struct + code.preamble_for( ca ),
        extra_includes = "".join( f'#include "{ inc }"\n' for inc in includes ),
        axis_includes = "".join( f'#include "{ AbstractAxis.cpp_shared_header( n ) }"\n'
                                 for n in ca.axis_names ),
        params        = ", ".join( params ),
        batch_indices = _batch_indices_decl( ca ),
        decls         = "\n".join( decls ),
        seeds         = "\n".join( s for s in seeds if s ),
        body          = body,
        binds         = binds,
    )
    return source, inputs, outputs, attrs, tuple( sources )


def _resolve_source( path ):
    """A kernel source, as given (`sdot/density/gaussians.cpp`, like an include) or absolute."""
    from ..compilation import include_roots
    p = Path( path )
    if p.is_absolute():
        return p
    for root in include_roots():
        if ( Path( root ) / p ).is_file():
            return Path( root ) / p
    raise FileNotFoundError( f"kernel source `{ path }` not found under the C++ source roots" )


