"""Compile C++ kernels against a plain-buffer FFI and call them on numpy arrays -- no jax.

The engine is `PointerFfi` (the same generated source as the Jax path, compiled against
`loom/support/numpy_ffi/xla/ffi/api/ffi.h` instead of jaxlib's header, its call frame filled with
`ctypes`); this module is the numpy side of it: a numpy array IS a host buffer, so it crosses by
address. The kernel itself is untouched, and runs on the CPU queue.

What this driver does NOT do is differentiate or trace: it is the forward driver (see `NumpyDriver`).
A backward kernel, if the code has one, is simply not run.
"""
from __future__ import annotations

import numpy

from . import PointerFfi
from .PointerFfi import Operand, PointerAdapter
# the shared pieces, under the names this module used to define them
from .PointerFfi import _load, _Buffer, _CallFrame, shim_include_dir     # noqa: F401


class NumpyAdapter( PointerAdapter ):
    framework = "numpy"

    def is_strided( self, x ) -> bool:
        return isinstance( x, numpy.ndarray ) and x.ndim > 0 and not x.flags.c_contiguous

    def operand( self, x, device, strided = False ) -> Operand:
        # kept alive by the operand until the kernel is done: the frame only holds addresses
        if strided and isinstance( x, numpy.ndarray ):
            return Operand( x.ctypes.data, x.shape, x, strides_bytes = x.strides )
        x = numpy.ascontiguousarray( x )
        return Operand( x.ctypes.data, x.shape, x, itemsize = x.itemsize )

    def zeros( self, shape, dtype, device ):
        return numpy.zeros( shape, dtype )

    def operand_of_output( self, x ) -> Operand:
        return Operand( x.ctypes.data, x.shape, x, itemsize = x.itemsize )


_adapter = NumpyAdapter()


def _run( code, ca, device, prefix ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result arrays )`, nothing written back."""
    return PointerFfi.run( code, ca, device, prefix, _adapter )


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`, minus differentiation)."""
    PointerFfi.call( code, ca, device, prefix, _adapter )
