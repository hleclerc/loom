from ..Buffer import Buffer


class JaxBuffer( Buffer ):
    """A concrete jax array: a value Python can read, and that XLA hands to a kernel dense."""

    framework = "jax"

    def astype( self, dtype ):
        import jax.numpy as jnp
        return jnp.asarray( self.raw, dtype = dtype.numpy_dtype )

    def device( self ):
        devices = list( self.raw.devices() )
        if len( devices ) != 1:
            return None
        return ( "cpu", 0 ) if devices[ 0 ].platform == "cpu" else ( "cuda", devices[ 0 ].id )

    def memory( self ):
        """A jax array is dense, immutable, on ONE device. Its address is read, never written through."""
        from ..Memory import Memory
        x = self.raw
        device = self.device()
        if device is None:
            raise ValueError( "a jax array spread over several devices cannot be read in place by a kernel" )
        step, strides = x.dtype.itemsize, []
        for n in reversed( x.shape ):
            strides.append( step )
            step *= max( n, 1 )
        return Memory( x.unsafe_buffer_pointer(), x.shape, reversed( strides ), device, x )

    def contiguous( self ):
        return self.raw
