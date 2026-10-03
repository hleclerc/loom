"""Compile C++ kernels into XLA-FFI handlers and expose them to Jax.

Pipeline (see `JaxDriver.call`): a C++ *body* is wrapped into a self-registering XLA FFI
handler, compiled to a shared library with the device's compiler (`make_library`), `dlopen`ed, and
registered with `jax.ffi.register_ffi_target`. The returned target name feeds
`jax.ffi.ffi_call`, which inserts the call into the XLA program (works eager and under
`jax.jit`, on CPU and — later — CUDA).

Two caches, both keyed by a content hash of (source + the compiler's build signature):
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
import os

import ctypes
import time
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy

from ..compilation import build_dir, journal, make_library
from ..compilation.build import kernels_root
from ..util.encode_base_62 import encode_base_62
from .CallArg_Errors import ERRORS_VAR_NAME
from .BackwardCall import call_backward
from .FfiSource import call_signature, _render_call, _resolve_source, _batch_indices_decl, _CALL_TEMPLATE

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
    # jaxlib ships the header-only XLA FFI C++ API (xla/ffi/api/ffi.h) under this dir. `-isystem`,
    # not `-I`: its headers do not compile warning-free, and those warnings are not ours to read.
    return [ "-isystem", ffi_include_dir() ]


def ffi_include_dir() -> str:
    return jax.ffi.include_dir()


def render_source( body: str ) -> str:
    """Wrap C++ *body* statements into a complete self-registering FFI handler source."""
    return _SOURCE_TEMPLATE.format( body = body )


def compile_and_register( source: str, device, prefix: str = "", sources = (),
                          code_name = None, signature = None ) -> str:
    """Compile *source* (plus the `sources` units it links, see `make_library`), load and
    register it, and return its Jax FFI target name.

    Idempotent and cached: repeated calls with the same source + device reuse the compiled
    library and the existing registration.
    """

    if not prefix:
        prefix = "sdot_ffi_"

    # Cache key = what actually changes the binary: the source, and how it is compiled.
    # Dropping the device OBJECT from the key is safe because the source always carries the
    # device anyway (`_CALL_TEMPLATE` substitutes `device.cpp_queue_type`), so a CPU and a CUDA
    # handler can never hash to the same name -- which matters, since this name is also the Jax
    # target and registration is per platform. (`str( CudaGpu:1 )` used to compile the very same
    # library twice on a two-GPU node.) How it is compiled = the compiler's `build_signature`:
    # the flags, and the machine when `-march=native` is among them -- a compilation setting
    # changes the binary as much as the source does, and a GPU architecture belongs there too.
    name = prefix + encode_base_62( f"{ source }|{ sources }|{ device.compiler.build_signature }" )
    if name in _loaded:
        journal.record_reuse()
        return name

    # a precompiled kernel first (a wheel's catalogue, see `compilation/catalogue.py`): the same
    # source, compiled for a CPU level this machine can run -- nothing to compile, nothing to
    # check. Recording (a catalogue being built) sees every source that goes by.
    from ..compilation import catalogue
    catalogue.record( source, sources, device )
    found = catalogue.lookup( source, sources, device )
    started = time.monotonic()
    if found is not None:
        lib, handler = found
        journal.record( name, code_name, signature, "catalogue" )
    else:
        if catalogue.policy() == "catalogue":
            raise RuntimeError( f"sdot: kernel `{ name }` is not in the catalogue and SDOT_KERNELS=catalogue forbids compiling it" )

        # ONE DIRECTORY PER KERNEL: the generated source, its object and its library live there,
        # and are removed in one go when no longer wanted -- see the docstring of `build.py`. The
        # name is already a hash of the content, so it identifies the directory unambiguously.
        work_dir = kernels_root() / name
        work_dir.mkdir( parents = True, exist_ok = True )

        # write-if-changed: the build graph decides on dates, and a rewrite of identical bytes
        # would look like a change to it (one recompilation per process, for nothing).
        src_path = work_dir / f"kernel{ device.compiler.source_suffix() }"
        if not ( src_path.exists() and src_path.read_text() == source ):
            src_path.write_text( source )

        lib_path = make_library(
            name + _lib_suffix(), [ src_path ], device,
            extra_flags = _ffi_include_flags(),
            sources = [ ( _resolve_source( p ), dict( d ) ) for p, d in sources ],
            work_dir = work_dir,
        )
        lib = ctypes.CDLL( str( lib_path ) )
        handler = getattr( lib, _HANDLER_SYMBOL )
        journal.record( name, code_name, signature, "compiled", time.monotonic() - started )

    jax.ffi.register_ffi_target(
        name, jax.ffi.pycapsule( handler ), platform = device.ffi_platform,
    )

    _loaded[ name ] = lib
    # the kernel-only timing ( `LOOM_KERNEL_TIMING`, CUDA ) asks each library for its totals
    if device.is_cuda_gpu:
        from ..devices import kernel_timing
        kernel_timing.register( name, code_name, lib )
    return name


def call_body( body: str, device ):
    """Compile a C++ *body* (no arguments yet) and return the result of its FFI call.

    Convenience for the current bootstrap step: renders the source, compiles/registers it,
    and invokes it. The int32 token array is returned as-is.
    """
    target = compile_and_register( render_source( body ), device )
    return jax.ffi.ffi_call( target, jax.ShapeDtypeStruct( ( 1, ), jnp.int32 ) )()


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
        source, _, outputs, attrs, sources = _render_call( code, ca, device )
        target = compile_and_register( source, device, prefix, sources,
                                       code_name = code.name, signature = call_signature( ca ) )
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
    if code.has_backward:
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
        perturbed = tuple( getattr( v, "perturbed", True ) and t.is_differentiable
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


def _call_backward( code, ca, device, prefix, inputs, outputs,
                    full_in, out_values, perturbed, cotangents ):
    return call_backward( code, ca, device, prefix, inputs, outputs,
                          full_in, out_values, perturbed, cotangents, _run )
