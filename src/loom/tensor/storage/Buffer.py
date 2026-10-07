import numpy

from ..PhysicalLayout import PhysicalLayout
from .Storage import Storage


class Buffer( Storage ):
    """A real backend buffer -- the ordinary case.

    The buffer is sized at CAPACITY, padding included, because that is what a kernel writes into;
    `view` crops it back to the logical extents. `layout` is the physical arrangement relative to
    the logical axes: `None` means the plain contiguous one (buffer in logical order, padding only
    from a ragged reference shape), which is what almost every tensor has. A construction site (a
    kernel output with a flattened/padded batch) may state an explicit one instead.

    The layout lives HERE, with the buffer, not on the axes: the same axis sits differently in
    different buffers."""

    holds_value = True

    @property
    def dtype( self ):
        """The `Dtype` the buffer ACTUALLY holds, read off the buffer: no driver, no host sync."""
        from ..Dtype import Dtype
        return Dtype.of( self.raw )

    def is_strided( self ) -> bool:
        """Held otherwise than dense row-major: a view the kernel reads with its own strides."""
        return False

    def to_numpy( self ):
        """The value as a host numpy array, wherever it lives (a card included)."""
        return numpy.asarray( self.raw )

    def memory( self ):
        """WHERE the buffer is, in terms any kernel can use whatever framework holds it: a `Memory` of
        address, shape, strides in bytes and device. Read by us, off the framework's own accessors -- so a
        kernel can take what it is given where it sits, and decide for itself whether it wants a copy
        (alignment, a dense layout). No conversion, no framework call that might make one."""
        raise NotImplementedError( f"{ type( self ).__name__ } cannot tell where its memory is" )

    def contiguous( self ):
        """The same value as a dense row-major array of ITS OWN framework: us, when it already is."""
        raise NotImplementedError( f"{ type( self ).__name__ } cannot be made contiguous" )

    def to_framework( self, target ):
        """The same value held by the framework named `target`: us when it is already ours, else a
        buffer of that framework sharing our memory where it can (see `buffers/conversion.py`)."""
        if target == self.framework:
            return self
        from .buffers.conversion import convert
        return Storage.of( convert( self, target ), self.reference_shape, self.explicit_layout )

    def astype( self, dtype ):
        """The buffer re-typed as `dtype` (a `Dtype`), in ITS OWN framework -- a no-op when it
        already is. Spelled through numpy dtypes, which every framework reads."""
        raise NotImplementedError( f"{ type( self ).__name__ } cannot be re-typed" )

    def __init__( self, raw, reference_shape = None, layout = None ) -> None:
        super().__init__( raw, reference_shape )
        self.explicit_layout = layout

    @property
    def buffer( self ):
        return self.raw

    def layout( self, rank ):
        if self.explicit_layout is not None:
            return self.explicit_layout
        return PhysicalLayout.contiguous( list( self.raw.shape ) )

    def allocated_sizes( self, rank ):
        return [ numpy.array( c, dtype = int ) for c in self.layout( rank ).caps ]

    def view( self, tensor ):
        # With an explicit non-contiguous layout the plain slice does not apply: the logical view is
        # GATHERED from the physical buffer through the layout's strides (a differentiable gather).
        # This is the one physical->logical boundary; everything else reads `view`, so ops and
        # results stay logical whatever the storage.
        if self.explicit_layout is not None and not self.explicit_layout.is_identity:
            return self._gather_logical( tensor )
        return self.raw[ tuple( slice( 0, s ) for s in tensor.shape ) ]

    def _gather_logical( self, tensor ):
        """`flat[ offsets ]` where `offsets[ i0, ..., ik ] = sum_d i_d * stride_d` (element strides
        from the layout), so a flattened / padded / reordered buffer is read back in logical order.
        `offsets` is a static index grid; the gather rides the backend (differentiable)."""
        extents = tensor.shape
        strides = self.explicit_layout.strides
        offsets = numpy.zeros( tuple( extents ), dtype = int )
        for i, ( e, s ) in enumerate( zip( extents, strides ) ):
            shape = [ 1 ] * len( extents )
            shape[ i ] = e
            offsets = offsets + numpy.arange( e, dtype = int ).reshape( shape ) * int( s )
        return self.raw.reshape( -1 )[ offsets ]

    def __repr__( self ) -> str:
        laid = "" if self.explicit_layout is None else ", laid out"
        return f"Buffer( { getattr( self.raw, 'shape', () ) }{ laid } )"
