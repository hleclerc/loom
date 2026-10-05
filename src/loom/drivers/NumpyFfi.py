"""Compile C++ kernels against a plain-buffer FFI and call them on numpy arrays -- no jax.

The same generated source as the Jax path (`FfiSource._render_call`), compiled against
`loom/support/numpy_ffi/xla/ffi/api/ffi.h` instead of jaxlib's header: that one spells the XLA FFI
API over host buffers, and its handler takes a call frame -- a few C structs -- that is filled here
with `ctypes`. The kernel itself is untouched, and runs on the CPU queue.

What this driver does NOT do is differentiate or trace: it is the forward driver (see `NumpyDriver`).
A backward kernel, if the code has one, is simply not run.

Two things keep it apart from the Jax path, on purpose:
* the compiled library is named after the shim (`np_` prefix, in the hash too), because a kernel
  built against XLA's header has another ABI -- the same source text must never find the other's binary;
* the precompiled catalogue is not consulted, for the same reason (it is built for XLA).
"""
from __future__ import annotations

import ctypes
import time
import sys

import numpy

from ..compilation import include_roots, journal, make_library
from ..compilation.build import kernels_root
from ..util.encode_base_62 import encode_base_62
from .FfiSource import call_signature, _render_call, _resolve_source
from ..compilation.generated_headers import headers_key, write_overlay

_HANDLER_SYMBOL = "sdot_ffi_entry"
_MAX_RANK = 8       # `XLA_FFI_Buffer::dims`

# a loaded handler per kernel name, for the life of the process: never `dlclose`d (a thread pool may
# still own threads in it when the process tears down)
_loaded: dict = {}


class _Buffer( ctypes.Structure ):
    _fields_ = [ ( "data", ctypes.c_void_p ), ( "rank", ctypes.c_int64 ), ( "dims", ctypes.c_int64 * _MAX_RANK ) ]


class _CallFrame( ctypes.Structure ):
    _fields_ = [ ( "args", ctypes.POINTER( _Buffer ) ), ( "rets", ctypes.POINTER( _Buffer ) ), ( "attrs", ctypes.POINTER( ctypes.c_int64 ) ) ]


def shim_include_dir() -> str:
    """The directory holding our `xla/ffi/api/ffi.h`, `-isystem` so it is the one `#include` finds."""
    from ..compilation import loom_include_root
    return str( loom_include_root() / "loom" / "support" / "numpy_ffi" )


def _lib_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _load( source: str, device, prefix: str, sources, code_name, signature, headers = None ):
    """The handler of this source, compiled (or reused from disk) and loaded."""
    # the key is what changes the binary: the source, how it is compiled -- and WHICH FFI it is
    # compiled against, so a numpy kernel never collides with a Jax one of the same text.
    # and the generated headers it was rendered with (`JaxFfi.compile_and_register`)
    name = "np_" + prefix + encode_base_62( f"{ source }|{ sources }|{ device.compiler.build_signature }|numpy_ffi"
                                            + headers_key( headers ) )
    if name in _loaded:
        journal.record_reuse()
        return _loaded[ name ]

    started = time.monotonic()
    work_dir = kernels_root() / name
    work_dir.mkdir( parents = True, exist_ok = True )

    src_path = work_dir / f"kernel{ device.compiler.source_suffix() }"
    if not ( src_path.exists() and src_path.read_text() == source ):
        src_path.write_text( source )

    lib_path = make_library(
        name + _lib_suffix(), [ src_path ], device,
        extra_flags = [ "-isystem", shim_include_dir() ],
        sources = [ ( _resolve_source( p ), dict( d ) ) for p, d in sources ],
        work_dir = work_dir,
        include_overlay = write_overlay( work_dir / "include", headers ) if headers else None,
    )
    lib = ctypes.CDLL( str( lib_path ) )
    entry = getattr( lib, _HANDLER_SYMBOL )
    entry.restype = ctypes.c_void_p
    entry.argtypes = [ ctypes.POINTER( _CallFrame ) ]
    journal.record( name, code_name, signature, "compiled", time.monotonic() - started )

    _loaded[ name ] = entry
    return entry


def _as_buffer( array: numpy.ndarray ) -> _Buffer:
    if array.ndim > _MAX_RANK:
        raise ValueError( f"numpy driver: a buffer of rank { array.ndim } (the limit is { _MAX_RANK })" )
    buffer = _Buffer( array.ctypes.data, array.ndim )
    for i, n in enumerate( array.shape ):
        buffer.dims[ i ] = n
    return buffer


def _run( code, ca, device, prefix ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result arrays )`, nothing written back."""
    source, inputs, outputs, attrs, sources, headers = _render_call( code, ca, device )
    entry = _load( source, device, prefix, sources, code.name, call_signature( ca ), headers )

    # contiguous, and kept alive by this list until the kernel is done: the frame only holds addresses
    in_arrays = [ numpy.ascontiguousarray( b.jax_input_array() ) for b in inputs ]
    out_arrays = [ numpy.zeros( shape, dtype ) for shape, dtype in ( b.out_shape_dtype() for b in outputs ) ]

    args = ( _Buffer * max( 1, len( in_arrays ) ) )( *[ _as_buffer( a ) for a in in_arrays ] )
    rets = ( _Buffer * max( 1, len( out_arrays ) ) )( *[ _as_buffer( a ) for a in out_arrays ] )
    values = ( ctypes.c_int64 * max( 1, len( attrs ) ) )( *[ int( v ) for _, _, v in attrs ] )
    frame = _CallFrame( args, rets, values )

    error = entry( ctypes.byref( frame ) )
    if error:
        raise RuntimeError( ctypes.string_at( error ).decode( errors = "replace" ) )
    return outputs, out_arrays


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`, minus differentiation)."""
    outputs, results = _run( code, ca, device, prefix )
    for buffer, array in zip( outputs, results ):
        buffer.jax_write_back( array )
