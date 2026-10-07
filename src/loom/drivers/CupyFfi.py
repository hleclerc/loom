"""Run C++ kernels on CuPy arrays -- the CUDA twin of `NumpyFfi`, over the engine of `PointerFfi`.

A CuPy array is a buffer on a card, and the kernel is compiled for the card (`CudaGpu`): it crosses by
ADDRESS (`data.ptr`, which already counts the offset of a view), with the strides CuPy holds, so a transposed
view, a slice or a broadcast is read where it is. The kernel is launched on the stream CuPy is working on
(`get_current_stream`) and takes its scratch from CuPy's memory pool: what CuPy launched before is finished when
the kernel starts, and what the kernel launched is finished when CuPy reads the result.

An array on another card than the kernel's is refused, with a message, and not moved: a transfer is a cost
the caller should see.
"""
from __future__ import annotations

import cupy

from . import PointerFfi
from .PointerFfi import Operand, PointerAdapter


class CupyAdapter( PointerAdapter ):
    framework = "cupy"

    @staticmethod
    def _on( device ):
        """The CuPy device a loom `CudaGpu` stands for (`None` for anything else: CuPy has only cards)."""
        if not getattr( device, "is_cuda_gpu", False ):
            raise NotImplementedError( f"the cupy driver launches kernels on a CUDA card (got { device })" )
        return cupy.cuda.Device( device.device_id )

    def is_strided( self, x ) -> bool:
        return isinstance( x, cupy.ndarray ) and x.ndim > 0 and not x.flags.c_contiguous

    def operand( self, x, device, strided = False ) -> Operand:
        card = self._on( device )
        if not isinstance( x, cupy.ndarray ):
            with card:
                x = cupy.asarray( x )               # a host value (a list, a numpy array) is given to the card
        # an array on another card is taken where it is: the view's memory space says so, and the queue transfers
        if x.flags.c_contiguous:
            return Operand( x.data.ptr, x.shape, x, itemsize = x.itemsize )
        if strided:
            # as it is: the kernel reads these strides (CuPy's are in bytes, like ours)
            return Operand( x.data.ptr, x.shape, x, strides_bytes = x.strides )
        # the kernel was generated for a dense buffer: the one case that costs a copy
        x = cupy.ascontiguousarray( x )
        return Operand( x.data.ptr, x.shape, x, itemsize = x.itemsize )

    def zeros( self, shape, dtype, device ):
        with self._on( device ):
            return cupy.zeros( tuple( shape ), dtype )

    def empty( self, shape, dtype, device ):
        with self._on( device ):
            return cupy.empty( tuple( shape ), dtype )

    def operand_of_output( self, x ) -> Operand:
        return Operand( x.data.ptr, x.shape, x, itemsize = x.itemsize )

    _side_streams: dict = {}

    def _streams( self, device ):
        """`( cupy's current stream, ours or None )`: ours only when cupy's is the legacy default one."""
        with self._on( device ) as card:
            current = cupy.cuda.get_current_stream()
            if current.ptr != 0:
                return current, None
            if card.id not in self._side_streams:
                self._side_streams[ card.id ] = cupy.cuda.Stream( non_blocking = True )
            return current, self._side_streams[ card.id ]

    def stream( self, device ):
        current, side = self._streams( device )
        return ( side or current ).ptr

    def order_before( self, device ):
        current, side = self._streams( device )
        if side is not None:
            side.wait_event( current.record() )

    def order_after( self, device ):
        current, side = self._streams( device )
        if side is not None:
            current.wait_event( side.record() )


_adapter = CupyAdapter()


def _run( code, ca, device, prefix ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result arrays )`, nothing written back."""
    return PointerFfi.run( code, ca, device, prefix, _adapter )


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`, minus differentiation)."""
    PointerFfi.call( code, ca, device, prefix, _adapter )
