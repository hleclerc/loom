import cupy

from ..Buffer import Buffer


class CupyBuffer( Buffer ):
    """A cupy array, on a card."""

    framework = "cupy"

    def is_strided( self ) -> bool:
        return self.raw.ndim > 0 and not self.raw.flags.c_contiguous

    def to_numpy( self ):
        return cupy.asnumpy( self.raw )

    def astype( self, dtype ):
        return cupy.asarray( self.raw, dtype = dtype.numpy_dtype )

    def device( self ):
        return ( "cuda", self.raw.device.id )

    def memory( self ):
        from ..Memory import Memory
        x = self.raw
        return Memory( x.data.ptr, x.shape, x.strides, self.device(), x )

    def contiguous( self ):
        return cupy.ascontiguousarray( self.raw )
