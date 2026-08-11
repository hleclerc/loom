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
        # a symbolic-zero cotangent: it IS there (reads as 0), but no storage backs it -- so it
        # is unbound like a `NoneTensor`, only it lowers to a `ZeroTensor` (see `cpp_type`).
        self.symbolic_zero = getattr( inst, "is_symbolic_zero", False )
        # a symbolic FILL (`Tensor.full`): a single scalar backs the value; it BINDS a rank-0 buffer
        # but its C++ view is a storageless `FillTensor` over the LOGICAL shape (its extents read from
        # a sibling real buffer at emit time -- hence the back-ref to the analysis).
        self.is_fill = getattr( inst, "is_fill", False )
        self._caa = call_args_analysis
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
        self._has_vmap = True   # the framework owns this leading dim -> stay contiguous (see `layout`)

    def batch_dim_expr( self, name ):
        # a fill has only a scalar buffer -- it cannot answer a real extent, so it never resolves an
        # axis size for anyone (a fill's OWN extents are read FROM the real buffers, not the reverse).
        if self.is_fill or name not in self.axis_names:
            return None
        return self.jax_dim( self.axis_names.index( name ) )

    # -- driver-agnostic C++ (the same for every driver) --
    def _cpp_scalar( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "float", ( "f", 8 ): "double",
                 ( "i", 4 ): "std::int32_t", ( "i", 8 ): "std::int64_t",
                 ( "u", 4 ): "std::uint32_t", ( "u", 8 ): "std::uint64_t" }[ ( dt.kind, dt.itemsize ) ]

    def _cpp_shape_tuple( self ):
        # the extents come from the BUFFER, not from `self.shape`: see `CallArg.jax_dim`.
        return "tuple( " + ", ".join( self.jax_dim( d ) for d in range( len( self.shape ) ) ) + " )"

    def _cpp_logical_shape_tuple( self ):
        # the LOGICAL extents as literals -- used with a NON-contiguous layout, where the buffer's
        # physical dims (flattened/reordered) no longer match the logical axes, so `jax_dim` (which
        # reads the buffer) cannot serve them. Batch extents are prescribed and the rest are the
        # capacities this call allocates: all known at trace time.
        return "tuple( " + ", ".join( f"SI( { int( e ) } )" for e in self.shape ) + " )"

    def _cpp_strides_tuple( self ):
        # the per-LOGICAL-axis BYTE strides of the physical layout (what `tensor_view`'s 4th arg wants).
        return "tuple( " + ", ".join( f"SI( { s } )" for s in self.layout.strides_bytes( self.itemsize ) ) + " )"

    def _cpp_axis_tuple( self ):
        return "tuple( " + ", ".join( self.axis_names ) + " )"

    def _cpp_shape_type( self ):
        # the *type* of the shape tuple: only the rank (extents are runtime `SI`s) -- a
        # statically known extent would show up here as a `Ct<SI,n>`.
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_axis_names_type( self ):
        # `DEFINE_AXIS( num_vertex )` declares the type `_num_vertex` (and the value `num_vertex`).
        return "Tuple<" + ", ".join( "_" + n for n in self.axis_names ) + ">"

    def cpp_type( self ):
        """This member's C++ type. An unbound attribute is not a degenerate view: it is a
        `NoneTensor` (absent -- a compile-time fact) or a `ZeroTensor` (a symbolic zero, read as
        0). Where the data lives is in the type too (`memory_space`): on a GPU, XLA already put
        this buffer in device memory."""
        if self.is_fill:
            # storageless constant over the logical shape; every element reads one scalar (see FillTensor.h).
            return ( f"FillTensor<{ self._cpp_scalar() }, { self._cpp_shape_type() }, "
                     f"{ self._cpp_axis_names_type() }>" )
        if not self.io_category.is_bound:
            kind = "ZeroTensor" if self.symbolic_zero else "NoneTensor"
            return ( f"{ kind }<{ self._cpp_scalar() }, { self._cpp_shape_type() }, "
                     f"{ self._cpp_axis_names_type() }>" )
        # a NON-contiguous layout spells its Strides in the type too (a runtime `Tuple<SI,...>`);
        # the contiguous default leaves the template's default Strides (so today's type is unchanged).
        strides = "" if self.layout.is_identity else f", { self._cpp_shape_type() }"
        return ( f"TensorView<{ self._cpp_scalar() }, { self._cpp_shape_type() }, "
                 f"{ self.memory_space }, { self._cpp_axis_names_type() }{ strides }>" )

    def cpp_view( self ):
        # a `NoneTensor` has nothing to view: it value-initializes.
        if not self.io_category.is_bound:
            return f"{ self.cpp_type() }{{}}"
        if self.is_fill:
            # the scalar data ptr + the LOGICAL extents, each read from a sibling REAL buffer that
            # carries the axis (the analysis finds it, like a batch extent) -- so no extent is baked.
            extents = ", ".join( self._caa.batch_dim_expr( a ) for a in self.axis_names )
            return f"{ self.cpp_type() }{{ { self.jax_data_ptr() }, tuple( { extents } ) }}"
        ptr = self.jax_data_ptr()
        if self.layout.is_identity:
            return ( f"tensor_view<{ self.memory_space }>( { ptr }, { self._cpp_shape_tuple() }, "
                     f"{ self._cpp_axis_tuple() } )" )
        # non-contiguous: LOGICAL extents (literals) + physical BYTE strides -> the 4-arg overload.
        return ( f"tensor_view<{ self.memory_space }>( { ptr }, { self._cpp_logical_shape_tuple() }, "
                 f"{ self._cpp_axis_tuple() }, { self._cpp_strides_tuple() } )" )

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_member( self, owner_name ):
        """Zero a SHARED float OUTPUT of a BATCHED call, before the body runs.

        Such an output carries NONE of the call's batch axes, yet the call has some: every item
        writes the SAME buffer, so the kernel ACCUMULATES into it (e.g. a ProjectedSumOfDiracs points
        gradient, atomic-added by every angle) and it must start at zero. A per-item output (one that
        carries a batch axis) is written once per item -- no seed; and with no batch there is no
        accumulation at all. `fill_with( queue, 0 )` goes through the queue, so it is ordered before
        the body's kernel. Unbound (NoneTensor/ZeroTensor) and integer/int scratch have nothing to seed."""
        if not ( self.io_category.is_bound and self.io_category.is_output and self.dtype.floating_point ):
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
        # the PHYSICAL buffer XLA allocates / binds: the layout's `buffer_shape` (flattened+padded
        # batch when non-contiguous, else `self.shape`). The C++ view reinterprets it logically.
        if self.is_fill:
            return []   # a single scalar backs the whole (logical) fill -- a rank-0 buffer
        return self.layout.buffer_shape

    def jax_cpp_init( self ):
        return self.cpp_view()

    def jax_input_array( self ):
        return self.inst.raw

    def jax_out_spec( self ):
        import jax
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), self.dtype.driver_version )

    def jax_write_back( self, array ):
        self.inst.set_raw( array )
        # hand the result tensor its physical layout so downstream ops read it back logically (via
        # `Tensor.tensor`'s gather). Identity stays `None` -> the plain contiguous default.
        if not self.layout.is_identity:
            self.inst._layout = self.layout
