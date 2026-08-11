import math


def _round_up( x, multiple ):
    """`x` rounded UP to the next multiple of `multiple` (>= 1). `multiple == 1` returns `x`."""
    return ( ( x + multiple - 1 ) // multiple ) * multiple if multiple > 1 else x


def items_per_alignment( alignment_bytes, itemsize ):
    """The number of ITEMS whose byte size is a whole number of `alignment_bytes` blocks -- the item
    granularity a BYTE alignment imposes. It DEPENDS on the item size: a 128-byte alignment is 32
    items in fp32 but 16 in fp64. `alignment_bytes == 1` (or dividing `itemsize`) gives 1 (no padding)."""
    return math.lcm( int( alignment_bytes ), int( itemsize ) ) // int( itemsize )


class PhysicalLayout:
    """How a tensor's LOGICAL dimensions sit in a PHYSICAL buffer.

    The LOGICAL shape (what the DSL presents -- batch axes first) is separate from how the bytes are
    actually laid out. Here the policy is: the BATCH axes are FLATTENED into a single leading physical
    dimension whose item capacity is padded so its BYTE size aligns to the hardware; the non-batch
    axes follow contiguously. So `product(batch sizes) + padding` is a multiple of the item
    granularity a hardware BYTE alignment imposes -- which is why it differs between fp32 and fp64.

    Two properties keep this safe to switch on:
      * with NO batch axis it is EXACTLY the contiguous layout (`is_identity`);
      * at alignment 1 (or none) the flattened+contiguous layout is byte-identical to the plain one --
        the strides below reduce to the contiguous ones -- so turning the machinery on changes nothing.

    The C++ side needs no change: a `TensorView< TF, Shape, Space, AxisNames, Strides >` already
    separates the logical extents (`Shape`, what the kernel iterates) from the physical BYTE strides
    (`Strides`); `tensor_view( ptr, shape, names, strides )` is the 4-arg hook. And nothing has to be
    "un-padded": the padding is a property the `Tensor` carries (capacity vs count), so it rides
    through ops and the backward (an ordinary forward over `Tensor`s); only the raw/FFI BOUNDARY maps
    logical <-> physical, uniformly for inputs, outputs and gradients.

    Physical axis ORDER: each axis carries a hardware `phys_num` (an integer; lower = more leading).
    The non-batch axes are placed in ASCENDING `phys_num`, EQUAL numbers keeping their logical order
    (ties tolerated -- axes of equal number may sit in any relative order, so we keep the logical
    one). The leading batch group always flattens+pads first. This reorder is pure PERFORMANCE: the
    per-axis strides make the logical view transparent, so it lives entirely in this class -- the API
    and every logical op are unaffected. `phys_num = None` keeps the logical order.
    """

    def __init__( self, caps, buffer_shape, strides, is_identity ):
        self.caps = caps                   # capacity (allocated extent) per LOGICAL dimension
        self.buffer_shape = buffer_shape   # dense physical buffer to allocate (a list of ints)
        self.strides = strides             # one stride (in ELEMENTS) per LOGICAL dimension
        self.is_identity = is_identity     # True when this is just the plain contiguous layout

    @classmethod
    def contiguous( cls, caps ):
        """The plain layout: one physical dim per logical dim, row-major, no padding -- what a Tensor
        whose buffer is a dense array in logical order has (the current default)."""
        caps = [ int( c ) for c in caps ]
        return cls( caps, list( caps ), _contiguous( caps ), is_identity = True )

    @classmethod
    def of( cls, caps, is_batch, alignment_bytes = 1, itemsize = 1, phys_num = None ):
        """Build the layout from `caps` (capacity per logical dim), `is_batch` (a bool per logical
        dim marking the batch axes to flatten+pad), the hardware `alignment_bytes` and the element
        `itemsize` (their ratio gives the item padding granularity).

        `phys_num` (optional, one int per LOGICAL dim, lower = more leading) is the hardware
        physical-order POLICY: the NON-batch axes are laid out in ascending `phys_num`, EQUAL numbers
        keeping their logical order (a STABLE sort -- ties tolerated, as intended). This is a pure
        PERFORMANCE reorder: the strides below make the logical view transparent, so `None` (keep
        logical order) and any permutation give the same logical tensor. The batch group always
        leads physically (it flattens+pads into the single leading dim), whatever `phys_num` says
        about it. No batch dim AND no reorder -> the plain contiguous layout."""
        caps = [ int( c ) for c in caps ]
        rank = len( caps )

        batch = [ i for i, b in enumerate( is_batch ) if b ]
        other = [ i for i, b in enumerate( is_batch ) if not b ]

        # PHYSICAL order of the non-batch axes: ascending `phys_num`, ties keep logical order.
        if phys_num is None:
            other_phys = list( other )
        else:
            other_phys = sorted( other, key = lambda i: ( phys_num[ i ], i ) )
        reordered = other_phys != other

        if not batch and not reordered:
            return cls.contiguous( caps )

        batch_caps = [ caps[ i ] for i in batch ]
        other_caps_phys = [ caps[ i ] for i in other_phys ]   # non-batch caps in PHYSICAL order
        multiple = items_per_alignment( alignment_bytes, itemsize )
        padded = _round_up( math.prod( batch_caps ), multiple ) if batch else 0

        # physical buffer, row-major: [ flattened+padded batch ] (only if there IS a batch) then the
        # non-batch axes in physical order. `phys_strides` are its row-major element strides.
        buffer_shape = ( [ padded ] if batch else [] ) + other_caps_phys
        phys_strides = _contiguous( buffer_shape )
        inner_strides = phys_strides[ 1: ] if batch else phys_strides

        strides = [ 0 ] * rank
        # the batch axes decompose the single leading physical dim (row-major within it); each block
        # spans `inner` elements (the leading physical stride).
        lead = phys_strides[ 0 ] if batch else 0
        for k, i in enumerate( batch ):
            strides[ i ] = math.prod( batch_caps[ k + 1: ] ) * lead
        # each non-batch axis gets the row-major stride of its PHYSICAL position.
        for pos, i in enumerate( other_phys ):
            strides[ i ] = inner_strides[ pos ]

        # if nothing actually moves -- no padding, no reorder, and the batch axes were already the
        # leading contiguous prefix so the flattened strides equal the plain ones -- keep the LOGICAL
        # shape (do NOT flatten): byte-identical AND same-rank, so the lowering is unchanged.
        if strides == _contiguous( caps ) and ( not batch or padded == math.prod( batch_caps ) ):
            return cls.contiguous( caps )
        return cls( caps, buffer_shape, strides, is_identity = False )

    def strides_bytes( self, itemsize ):
        return [ s * int( itemsize ) for s in self.strides ]

    @property
    def element_count( self ):
        return math.prod( self.buffer_shape )


def _contiguous( shape ):
    """Row-major element strides for a dense `shape` (last dim = 1)."""
    strides, acc = [ 0 ] * len( shape ), 1
    for i in range( len( shape ) - 1, -1, -1 ):
        strides[ i ] = acc
        acc *= int( shape[ i ] )
    return strides
