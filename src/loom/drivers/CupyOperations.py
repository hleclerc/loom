"""The cupy framework's operations (see `CupyFramework`)."""
from .NumpyOperations import NumpyOperations
import numpy
import cupy


class CupyOperations( NumpyOperations ):
    """What the cupy framework DOES with an array that already exists: its operations, its
    transforms, its readings, its way of building one from a concrete dtype and an explicit device, and of
    launching a kernel. It has no default of its own: the defaults are loom's (`Settings`), and are handed in."""

    xp = cupy

    def to_numpy( self, x ):
        return cupy.asnumpy( x ) if isinstance( x, cupy.ndarray ) else numpy.asarray( x )

    def matmul( self, a, b ):
        a, b = cupy.asarray( a ), cupy.asarray( b )
        if a.dtype.kind in "iub":       # cuBLAS has no integer product: contracted by hand, as under torch
            a2 = a[ None ] if a.ndim == 1 else a
            b2 = b[ :, None ] if b.ndim == 1 else b
            res = ( a2[ ..., :, :, None ] * b2[ ..., None, :, : ] ).sum( -2 )
            if a.ndim == 1:
                res = res[ ..., 0, : ]
            if b.ndim == 1:
                res = res[ ..., 0 ]
            return res
        return a @ b

    def where( self, cond, a, b ):
        # a mask worked out on the host (from counts, say) meets an array on the card: CuPy does not mix them
        host = lambda v: cupy.asarray( v ) if isinstance( v, numpy.ndarray ) else v
        return cupy.where( host( cond ), host( a ), host( b ) )

    @property
    def available_gpus( self ):
        return cupy.cuda.runtime.getDeviceCount()

    def _default_device( self ):
        from ..devices.CudaGpu import CudaGpu
        gpu = CudaGpu( cupy.cuda.runtime.getDevice() )
        if not gpu.device_is_present:
            raise RuntimeError( "the cupy driver needs a CUDA card loom can compile for ( nvcc and a driver: see "
                                "`loom[cuda]` ), and found none" )
        return gpu

    def _check_device( self, device ):
        if not device.is_cuda_gpu:
            raise ValueError( f"the cupy driver runs on a CUDA card (asked for { device }): use numpy for the CPU" )
        cupy.cuda.Device( device.device_id ).use()

    @staticmethod
    def _ffi():
        from . import CupyFfi
        return CupyFfi
