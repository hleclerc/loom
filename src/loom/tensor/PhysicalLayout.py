import math


def _round_up( x, multiple ):
    """`x` rounded UP to the next multiple of `multiple` (>= 1). `multiple == 1` returns `x`."""
    return ( ( x + multiple - 1 ) // multiple ) * multiple if multiple > 1 else x


# Ce qu'on s'autorise a depenser en padding, PAR TENSEUR, pour separer les blocs du lot (voir
# `PhysicalLayout.of`). Assez large pour le scratch par work-item -- quelques dizaines d'items,
# donc quelques kilo-octets -- et assez etroit pour ne jamais toucher un tableau dimensionne sur
# les donnees, ou le meme padding se compterait en dizaines de mega-octets.
MAX_PADDING_BYTES = 1 << 20


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
    def of( cls, caps, is_batch, alignment_bytes = 1, itemsize = 1, phys_num = None,
            item_alignment_bytes = 0 ):
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

        # ---- la FOULEE par item du lot, arrondie a l'alignement -- SUR DEMANDE SEULEMENT ------
        #
        # `item_alignment_bytes = 0` par defaut, donc rien ne bouge tant qu'un tenseur ne le demande
        # pas : la disposition reste EXACTEMENT celle d'avant, aux octets pres. C'est voulu -- ce
        # padding se paie en memoire et ne rapporte que sur les tenseurs REELLEMENT partages entre
        # work-items, qui sont une poignee. Voir `Tensor.item_alignment_bytes` pour le demander.
        #
        # Deux items voisins du lot sont traites par deux work-items DIFFERENTS, en meme temps (la
        # boucle du kernel est striee). Si leurs blocs se touchent, ils partagent une ligne de
        # cache -- et chaque ecriture de l'un invalide la ligne chez l'autre. C'est le FAUX PARTAGE,
        # et il ne se voit pas: le travail est le meme, seuls les cycles doublent.
        #
        # Mesure qui a motive ceci (Xeon W-2145, 8 threads, 1e6 germes en 2D, `sdot` PowerDiagram):
        # a instructions EGALES a 0.1 % pres, l'IPC tombe de 1.60 a 0.455 et
        # `mem_load_l3_hit_retired.xsnp_hitm` -- « ce chargement a trouve la ligne modifiee dans le
        # cache d'un AUTRE coeur » -- passe de 1.5e3 a 2.7e7. `perf c2c` a nomme les lignes: deux
        # tableaux d'`int32` a un element par work-item (des compteurs) portent a eux seuls 65 % du
        # trafic, aux offsets 0x0, 0x4, 0x8 ... d'une meme ligne.
        #
        # Le cas qui fait mal est donc exactement celui d'un bloc PLUS PETIT qu'une ligne: un
        # scalaire par work-item en met seize dans la meme. Un bloc deja gros, lui, ne gagne qu'un
        # arrondi.
        #
        # ---- et pourquoi c'est BORNE
        #
        # Le padding coute `( lead - inner ) * padded` elements, et cette depense n'a de sens que
        # devant ce qu'elle achete. Un axe de lot dimensionne sur les THREADS est minuscule (seize
        # items: ~1 Ko de padding, pour supprimer le faux partage). Un axe de lot dimensionne sur
        # les DONNEES ne l'est pas: un scalaire par germe a 1e6 germes passerait de 4 a 64 Mo. On
        # ne pade donc que tant que la depense reste NEGLIGEABLE EN ABSOLU -- une regle qui n'a pas
        # besoin de savoir ce que l'axe signifie, et qui retient d'elle-meme le scratch par
        # work-item sans toucher aux tableaux de donnees.
        #
        # CE QUE CA RAPPORTE, mesure : sur le `sdot` PowerDiagram a huit threads, `xsnp_hitm` baisse
        # de 21 % et le temps de 1.4 %. Le faux partage est donc REEL mais PAS le cout dominant --
        # raison de plus pour que ce soit un reglage par tenseur et non une politique.
        inner = math.prod( other_caps_phys ) if other_caps_phys else 1
        lead = inner
        if batch and item_alignment_bytes:
            item_multiple = items_per_alignment( item_alignment_bytes, itemsize )
            if inner % item_multiple:
                candidate = _round_up( inner, item_multiple )
                if ( candidate - inner ) * padded * int( itemsize ) <= MAX_PADDING_BYTES:
                    lead = candidate

        # physical buffer, row-major: [ flattened+padded batch ] (only if there IS a batch) then the
        # non-batch axes in physical order -- SAUF quand la foulee par item est padee, auquel cas le
        # bloc d'un item occupe `lead` elements dont `inner` seulement portent les axes non-lot.
        if lead != inner:
            buffer_shape = [ padded, lead ]
            inner_strides = _contiguous( other_caps_phys )
        else:
            buffer_shape = ( [ padded ] if batch else [] ) + other_caps_phys
            phys_strides = _contiguous( buffer_shape )
            inner_strides = phys_strides[ 1: ] if batch else phys_strides

        strides = [ 0 ] * rank
        # the batch axes decompose the single leading physical dim (row-major within it); each block
        # spans `lead` elements (the leading physical stride).
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
