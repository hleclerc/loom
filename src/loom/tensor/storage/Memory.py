class Memory:
    """A buffer's memory as a kernel sees it: no framework in it, only where the bytes are.

    `device` is `( "cpu", 0 )` or `( "cuda", index )`. `strides_bytes` are the buffer's own, so a
    transposed view, a slice or a broadcast is described as it is. `owner` keeps the memory alive."""

    __slots__ = ( "address", "shape", "strides_bytes", "device", "owner" )

    def __init__( self, address, shape, strides_bytes, device, owner ):
        self.address       = int( address )
        self.shape         = tuple( int( n ) for n in shape )
        self.strides_bytes = tuple( int( s ) for s in strides_bytes )
        self.device        = device
        self.owner         = owner

    def is_dense( self, itemsize ):
        """Row-major with no gap: what a kernel generated for a plain buffer reads."""
        step = int( itemsize )
        for n, s in zip( reversed( self.shape ), reversed( self.strides_bytes ) ):
            if n > 1 and s != step:
                return False
            step *= max( n, 1 )
        return True
