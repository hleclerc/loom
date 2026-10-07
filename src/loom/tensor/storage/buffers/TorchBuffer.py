from ..Buffer import Buffer


class TorchBuffer( Buffer ):
    """A torch tensor, on the CPU or on a card. It TRACES when autograd records it."""

    framework = "torch"

    @property
    def traces( self ) -> bool:
        return bool( self.raw.requires_grad )

    def is_strided( self ) -> bool:
        return self.raw.dim() > 0 and not self.raw.is_contiguous()

    def to_numpy( self ):
        return self.raw.detach().cpu().numpy()

    def astype( self, dtype ):
        from ....drivers.TorchFfi import _torch_dtype
        return self.raw.to( _torch_dtype( dtype.numpy_dtype ) )

    def device( self ):
        d = self.raw.device
        return ( "cpu", 0 ) if d.type == "cpu" else ( "cuda", d.index )

    def memory( self ):
        from ..Memory import Memory
        t = self.raw
        return Memory( t.data_ptr(), t.shape, [ s * t.element_size() for s in t.stride() ], self.device(), t )

    def contiguous( self ):
        return self.raw.contiguous()
