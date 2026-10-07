import numpy

from ..Buffer import Buffer


class NumpyBuffer( Buffer ):
    """A numpy array: host memory every framework reads, and that asks no framework for anything."""

    framework = "numpy"

    def is_strided( self ) -> bool:
        return self.raw.ndim > 0 and not self.raw.flags.c_contiguous

    def astype( self, dtype ):
        return numpy.asarray( self.raw, dtype = dtype.numpy_dtype )

    def device( self ):
        return ( "cpu", 0 )

    def memory( self ):
        from ..Memory import Memory
        return Memory( self.raw.ctypes.data, self.raw.shape, self.raw.strides, self.device(), self.raw )

    def contiguous( self ):
        return numpy.ascontiguousarray( self.raw )
