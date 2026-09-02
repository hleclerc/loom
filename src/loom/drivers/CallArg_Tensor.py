import os
import weakref

from .CallArg import CallArg

class CallArg_Tensor( CallArg ):
    """A tensor attribute, and how it reaches the kernel.

    `inst` is the `Tensor` itself, so everything is read off it -- there is nothing to resolve
    from siblings. The shape depends on the direction, and that is the whole point:

    * INPUT   -> `inst.capacity`: the size the data ACTUALLY has (read off its buffer). An
                 output that wants to grow must not force us to inflate the input.
    * OUTPUT  -> the size THIS CALL asks for: the axes evaluated on the capacities the call was
                 given (see `CallArgsAnalysis`). Known to Python, as an XLA shape must be.
    * UNBOUND -> no buffer, and no `TensorView` either: the attribute lowers to a `NoneTensor`,
                 a distinct TYPE carrying the declared TF/Shape/AxisNames and no data. The
                 kernel discriminates at compile time; there is nothing to test at runtime.

    Codegen splits by concern: `cpp_*` emits the driver-agnostic C++ (identical for Jax or
    Torch), `jax_*` carries the Jax FFI ABI (buffer types, data pointer, result specs).
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( call_args_analysis.io_category( path, inst.raw is not None ), name )

        self.inst = inst
        self.dtype = inst.dtype
        # LAST line of defense on the dtype invariant (`Tensor._as_declared` is the first): below,
        # `cpp_scalar` spells this dtype as the element type of the buffer we are about to bind, so
        # a buffer that disagrees is REINTERPRETED by the kernel -- silent garbage, not a type error.
        if inst.raw is not None and not inst.is_fill:
            from ..tensor.Dtype import Dtype
            have = Dtype.of( inst.raw )
            assert self.dtype.same_as( have ), \
                f"tensor '{ name }' is declared { self.dtype.cpp_name } but its buffer holds { have.cpp_name }"
        # HOW the value is backed -- snapshotted here, like every other fact this node reads off the
        # tensor, so a later rebinding (this call's own write-back) cannot change what we emit. It is
        # what decides our C++ form: `cpp_type` / `cpp_view` / `_jax_buffer_shape` all ask IT, and
        # this class only supplies the spelling primitives (see `loom/tensor/storage.py`).
        self.storage = inst.storage
        # la SEULE référence remontante de tout l'arbre d'abaissement, et elle est FAIBLE : notre
        # analyse nous tient (`args` -> ... -> nous), donc la tenir en retour fermait l'anneau
        # `CallArgsAnalysis <-> CallArg_*`, que seul le ramasse-miettes cyclique défait. Ce n'est
        # pas qu'une question de nommage d'axes : cet anneau retenait aussi les agrégats de
        # l'appel, donc leurs tampons -- des tableaux du device -- bien après la fin de l'appel.
        # Faible est sûr : on ne s'en sert que pendant la génération de code, que l'analyse pilote.
        self._caa = weakref.ref( call_args_analysis )
        self.memory_space = call_args_analysis.cpp_memory_space

        if self.io_category.is_output:
            self.shape = [ int( s ) for s in call_args_analysis.output_shape( inst, path ) ]
        elif self.io_category.is_input:
            self.shape = [ int( s ) for s in inst.capacity ]
        else:
            self.shape = [ 0 ] * inst.rank

        # A TensorView must name each of its ARRAY dimensions. Each declared axis knows how it
        # unrolls into ordinary names (`cpp_dim_names`, the name analogue of `max_list`): a plain
        # `Axis` yields one; an unrolled `AxisList` yields several DISTINCT ones (`img_pos_0`,
        # `img_pos_1`, ...) since it only DEFINES several ordinary axes and changes nothing else
        # about the tensor. Concatenating gives one name per dimension (count == rank), each
        # `DEFINE_AXIS`'d by the aggregate (which folds in `axis_names`). No unrolling logic lives
        # here -- so nothing assumes a single, or any, `AxisList`.
        self.axis_names = [ n for index, axis in enumerate( inst.axes ) for n in axis.cpp_dim_names( index ) ]

        # Et, par dimension, l'extent que le TYPE peut porter : `None` quand il n'est connu qu'a
        # l'execution, l'entier quand il est fige a la COMPILATION -- c'est-a-dire quand l'extent de
        # l'axe ne depend que de `CtShapeVar`s (`RealTensor[ "num_vertex", "dim" ]` : `dim` oui,
        # `num_vertex` non, sa capacite double en cours de route). Le lowering le repand alors dans
        # le tuple de shape (`Ct<SI,2>` au lieu d'un `SI`), et `contiguous_strides` -- qui derive les
        # strides du TYPE de la shape -- en tire un stride de LIGNE compile-time : `vertex_positions(
        # v, d )` devient `base + v * 16 + d * 8` au lieu d'une multiplication par un `long long` lu
        # en memoire. Une capacite, elle, ne DOIT pas y aller : elle changerait a chaque doublement,
        # donc un noyau recompile a chaque fois.
        self._dim_ct_extent = _ct_extents_of( inst )

        # the PHYSICAL layout this buffer has (input: the one it already carries) or should get
        # (output: chosen from the device's batch alignment + this dtype's itemsize). `self.shape`
        # stays the LOGICAL extents (what the kernel iterates); the layout adds the physical
        # buffer_shape + per-axis strides. At alignment 1 (or no batch) it is IDENTITY -> the lowering
        # below is byte-identical to the contiguous one, so nothing changes for today's calls.
        #
        # `layout` is a LAZY property, not computed here: a `jax.vmap` prepends a leading dim by
        # calling `add_batch_axis` AFTER `__init__`, mutating `self.shape`; computing the layout now
        # would freeze a stale `buffer_shape`. The vmap path is also its own (contiguous) universe --
        # the framework handed us the extra dim -- so it never wants our batch-flatten policy.
        import numpy
        self.itemsize = int( numpy.dtype( self.dtype.driver_version ).itemsize )
        self._alignment_bytes = call_args_analysis.batch_alignment_bytes
        self._dim_is_batch = list( inst._dim_batch() )
        self._device = call_args_analysis.device
        self._has_vmap = False
        # the call's batch axes, to tell a SHARED output (accumulated into by every item) from a
        # per-item one -- see `cpp_seed_member`.
        self._call_batch_axes = list( call_args_analysis.batch_axes )

    @property
    def layout( self ):
        from ..tensor.PhysicalLayout import PhysicalLayout
        if self._has_vmap:
            return PhysicalLayout.contiguous( self.shape )      # vmap owns its leading dim -> contiguous
        if self.io_category.is_output:
            # the device's physical-order POLICY (None by default -> logical order, identity layout);
            # a device that prefers another order reorders the non-batch axes here (strides keep the
            # logical view). Only outputs choose a layout; inputs carry the one they already have.
            phys_num = self._device.physical_axis_num( self.axis_names ) if self._device else None
            return PhysicalLayout.of( self.shape, self._dim_is_batch,
                                      self._alignment_bytes, self.itemsize, phys_num )
        if self.io_category.is_input:
            return self.inst.buffer_layout
        return PhysicalLayout.contiguous( self.shape )          # unbound -> NoneTensor, unused

    # only the BUFFER binding is conditional on `is_bound`: an unbound tensor is a `NoneTensor`,
    # which still spells its axes in its type (`Tuple<_num_cut, _dim>`) -- so `cpp_axis_names`
    # answers them either way, and the analysis folds those in for their `DEFINE_AXIS`.
    def is_ffi_buffer( self ):
        return self.io_category.is_bound

    @property
    def is_differentiable( self ) -> bool:
        # deferred to the TENSOR: differentiability is a property of what the tensor is made of
        # (`IntTensor.is_differentiable`), not something re-derived from its dtype at each site.
        return self.inst.is_differentiable

    # -- the axes our type spells (see `CallArg.cpp_axis_names`) --
    def cpp_axis_names( self ):
        return self.axis_names

    # -- as a value a `vmap` maps over --
    def add_batch_axis( self, name, size ):
        """One more axis, in front -- a NAMED one, so the kernel selects it by name and a value
        that does not have it lets the index through. There is nothing more to it: a batch axis is
        an axis, and the buffer really did gain a leading dimension (that is what the framework
        handed us)."""
        self.axis_names = [ name ] + self.axis_names
        self.shape = [ int( size ) ] + self.shape
        self._dim_ct_extent = [ None ] + self._dim_ct_extent
        self._has_vmap = True   # the framework owns this leading dim -> stay contiguous (see `layout`)

    def batch_dim_expr( self, name ):
        # whether we can serve an extent at all depends on what backs us (a fill cannot: it has only
        # a scalar buffer), so the storage answers.
        return self.storage.batch_dim_expr( self, name )

    # -- driver-agnostic C++ (the same for every driver). Everything below the `cpp_*` helpers is
    # a SPELLING primitive: how one fragment is written. WHICH form this member takes is the
    # storage's call (`storage.cpp_type` / `cpp_view`), so a new way of being backed is a new
    # variant there, not another branch here. --
    def cpp_scalar( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "float", ( "f", 8 ): "double",
                 ( "i", 4 ): "std::int32_t", ( "i", 8 ): "std::int64_t",
                 ( "u", 4 ): "std::uint32_t", ( "u", 8 ): "std::uint64_t" }[ ( dt.kind, dt.itemsize ) ]

    def cpp_shape_tuple( self ):
        # the extents come from the BUFFER, not from `self.shape`: see `CallArg.jax_dim`. Except a
        # COMPILE-TIME one, which comes from the type instead -- reading it back off the buffer would
        # hand a `long long` to something the type already knows.
        return "tuple( " + ", ".join( self._cpp_extent( d ) or self.jax_dim( d )
                                      for d in range( len( self.shape ) ) ) + " )"

    def _cpp_extent( self, d ):
        """`Ct<SI,n>()` when dimension `d`'s extent is compile-time, else `None`."""
        n = self._dim_ct_extent[ d ]
        return None if n is None else f"Ct<SI, { n }>()"

    def cpp_logical_shape_tuple( self ):
        # the LOGICAL extents as literals -- used with a NON-contiguous layout, where the buffer's
        # physical dims (flattened/reordered) no longer match the logical axes, so `jax_dim` (which
        # reads the buffer) cannot serve them. Batch extents are prescribed and the rest are the
        # capacities this call allocates: all known at trace time.
        return "tuple( " + ", ".join( self._cpp_extent( d ) or f"SI( { int( e ) } )"
                                      for d, e in enumerate( self.shape ) ) + " )"

    def cpp_strides_tuple( self ):
        # the per-LOGICAL-axis BYTE strides of the physical layout (what `tensor_view`'s 4th arg wants).
        return "tuple( " + ", ".join( f"SI( { s } )" for s in self.layout.strides_bytes( self.itemsize ) ) + " )"

    def cpp_axis_tuple( self ):
        return "tuple( " + ", ".join( self.axis_names ) + " )"

    def cpp_shape_type( self ):
        # the *type* of the shape tuple: only the rank (extents are runtime `SI`s) -- a
        # statically known extent shows up here as a `Ct<SI,n>` -- see `_dim_ct_extent`.
        return "Tuple<" + ", ".join( "SI" if n is None else f"Ct<SI, { n }>"
                                     for n in self._dim_ct_extent ) + ">"

    def cpp_axis_names_type( self ):
        # `DEFINE_AXIS( num_vertex )` declares the type `_num_vertex` (and the value `num_vertex`).
        return "Tuple<" + ", ".join( "_" + n for n in self.axis_names ) + ">"

    def cpp_type( self ):
        """This member's C++ type -- asked of the STORAGE, since that is what it depends on: a real
        buffer is a `TensorView`, an absent value a `NoneTensor` (a compile-time fact, not a
        degenerate view to test at run time), a symbolic zero a `ZeroTensor`, a fill a
        `FillTensor`. Where the data lives is in the type too (`memory_space`): on a GPU, XLA has
        already put this buffer in device memory."""
        return self.storage.cpp_type( self )

    def cpp_view( self ):
        """How that type is initialized -- the storage's call for the same reason."""
        return self.storage.cpp_view( self )

    def sibling_dim_expr( self, name ):
        """Where the extent of axis `name` can be read at run time, off ANOTHER argument of this
        call that carries it. What a value with no extents of its own (a fill) builds its logical
        shape from -- so no extent is baked into the generated source."""
        caa = self._caa()
        if caa is None:
            raise RuntimeError( f"'{ self.name }' was asked for the extent of '{ name }' after its "
                                f"call's analysis was gone -- a lowering node only answers while "
                                f"the analysis driving the codegen is alive" )
        return caa.batch_dim_expr( name )

    def _rebind_analysis( self, caa ):
        self._caa = weakref.ref( caa )

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_member( self, owner_name ):
        """Zero a SHARED float OUTPUT of a BATCHED call, before the body runs.

        Such an output carries NONE of the call's batch axes, yet the call has some: every item
        writes the SAME buffer, so the kernel ACCUMULATES into it (e.g. a ProjectedSumOfDiracs points
        gradient, atomic-added by every angle) and it must start at zero. A per-item output (one that
        carries a batch axis) is written once per item -- no seed; and with no batch there is no
        accumulation at all. `fill_with( queue, 0 )` goes through the queue, so it is ordered before
        the body's kernel. Unbound (NoneTensor/ZeroTensor) and integer/int scratch have nothing to seed."""
        if not ( self.io_category.is_bound and self.io_category.is_output ):
            return ""

        # Toute sortie part à ZÉRO, sur toute sa CAPACITÉ, avant le corps.
        #
        # Un tampon de sortie que le corps n'écrit que PARTIELLEMENT laisse le reste tel que
        # l'allocateur l'a rendu -- et sur GPU ce n'est pas zéro. Deux façons d'en arriver là, et la
        # seconde est la règle, pas l'exception :
        #  * une boucle striée `for i = thread_index; i < n; i += nb_threads` ne touche rien quand
        #    `thread_index >= n`, ni un scratch déclaré en sortie que le forward n'écrit jamais ;
        #  * surtout, la dimension de BATCH est allouée à la capacité alignée (16 emplacements pour
        #    4 items), et le corps ne parcourt que le COMPTE -- depuis que la boucle se borne au
        #    compte et non plus à cette capacité (voir `CallArgsAnalysis.batch_dim_expr`), la queue
        #    rembourrée n'est plus écrite du tout.
        #
        # Ce que lit ensuite quelqu'un qui parcourt la capacité est alors indéterminé, et un COMPTE
        # indéterminé y borne une boucle : un accès des gigaoctets hors de toute allocation. Semer
        # rend cela inoffensif sans rien exiger des lecteurs.
        #
        # Ce n'est pas gratuit -- un remplissage par sortie et par appel -- d'où l'interrupteur, qui
        # sert maintenant à MESURER ce que coûte le semis, pas à décider s'il a lieu.
        if os.environ.get( "LOOM_ZERO_OUTPUTS", "" ).strip().lower() not in ( "0", "false", "no", "off" ):
            return f"{ owner_name }.{ self.name }.fill_with( queue, 0 );"

        # Le cas déjà couvert : une sortie flottante PARTAGÉE d'un appel batché. Elle ne porte aucun
        # axe de batch alors que l'appel en a, donc chaque item écrit le MÊME tampon : le kernel y
        # accumule (le gradient de points d'un `ProjectedSumOfDiracs`, ajouté atomiquement par chaque
        # angle) et il doit partir de zéro. Une sortie PAR ITEM est écrite une fois par item -- rien
        # à semer ; et sans batch il n'y a pas d'accumulation du tout. `fill_with( queue, 0 )` passe
        # par la queue, donc il est ordonné avant le kernel du corps.
        if not self.dtype.floating_point:
            return ""
        if not self._call_batch_axes or any( b in self.axis_names for b in self._call_batch_axes ):
            return ""
        return f"{ owner_name }.{ self.name }.fill_with( queue, 0 );"

    # -- as a member of an aggregate: one type parameter, spelled out at instantiation --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    # -- as a ROOT argument (a tensor needs no wrapper aggregate to be passed) --
    def cpp_root_decl( self, var_name ):
        return f"    auto { var_name } = { self.cpp_view() };"

    # -- Jax FFI ABI --
    def _jax_ffi_elem( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "ffi::F32", ( "f", 8 ): "ffi::F64",
                 ( "i", 4 ): "ffi::S32", ( "i", 8 ): "ffi::S64",
                 ( "u", 4 ): "ffi::U32", ( "u", 8 ): "ffi::U64" }[ ( dt.kind, dt.itemsize ) ]

    def jax_ffi_type( self ):
        return f"ffi::BufferR{ len( self._jax_buffer_shape() ) }<{ self._jax_ffi_elem() }>"

    def _jax_buffer_shape( self ):
        # the PHYSICAL buffer XLA allocates / binds -- a fill's is a single scalar, a laid-out one
        # is flattened + padded, the ordinary one is `self.shape`. The storage knows which it is.
        return self.storage.jax_buffer_shape( self )

    def jax_cpp_init( self ):
        return self.cpp_view()

    def jax_input_array( self ):
        return self.inst.raw

    def jax_out_spec( self ):
        import jax
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), self.dtype.driver_version )

    def jax_write_back( self, array ):
        # hand the result tensor its physical layout too, so downstream ops read it back logically
        # (via the storage's gather). Identity passes `None` -> the plain contiguous default.
        self.inst.set_raw( array, layout = None if self.layout.is_identity else self.layout )


def _ct_extents_of( inst ):
    """Per DIMENSION of `inst`, the extent frozen at C++ compile time, or `None`.

    An axis qualifies when its extent EXPRESSION (`hi - lo`, at step 1 -- a stepped window's size is
    a ceiling division, which no affine is) involves nothing but `CtShapeVar`s: their values are
    already baked into the generated source, so putting the extent in the type adds no key to the
    compile cache. An `AxisList` (several dimensions from one declaration) is left alone: it has one
    expression for several extents, and nothing here needs it yet.
    """
    from ..tensor.CtShapeVar import CtShapeVar
    res = []
    for index, axis in enumerate( inst.axes ):
        names = axis.cpp_dim_names( index )
        ct = None
        if len( names ) == 1 and axis.step == 1:
            symbols = ( axis.hi - axis.lo ).coeffs
            if all( isinstance( s, CtShapeVar ) for s in symbols ):
                n = axis.numeric_extent( lambda sv: None if sv.raw is None else int( sv.raw ) )
                ct = None if n is None else int( n )
        res += [ ct ] * len( names )
    return res
