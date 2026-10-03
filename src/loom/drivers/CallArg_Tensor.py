import os
import weakref

from .. import env

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
        # the ONLY back-reference in the whole lowering tree, and it is WEAK: our analysis holds
        # us (`args` -> ... -> us), so holding it in return closed the ring
        # `CallArgsAnalysis <-> CallArg_*`, which only the cyclic garbage collector undoes. It is
        # not just a matter of axis naming: that ring also retained the call's aggregates, hence
        # their buffers -- device arrays -- long after the call ended.
        # Weak is safe: we only use it during code generation, which the analysis drives.
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
        # ... under their CANONICAL name: a batch axis is named by the call, not by the `Axis`
        # object ( see `CallArgsAnalysis.cpp_axis_name` ), so that two identical calls
        # produce the same source.
        self.axis_names = [ call_args_analysis.cpp_axis_name( n )
                            for index, axis in enumerate( inst.axes ) for n in axis.cpp_dim_names( index ) ]

        # And, per dimension, the extent the TYPE can carry: `None` when it is only known at run
        # time, the integer when it is frozen at COMPILE time -- that is, when the axis's extent
        # depends on `CtShapeVar`s only (`RealTensor[ "num_vertex", "dim" ]`: `dim` yes,
        # `num_vertex` no, its capacity doubles along the way). The lowering then spreads it into
        # the shape tuple (`Ct<SI,2>` instead of an `SI`), and `contiguous_strides` -- which derives
        # the strides from the shape's TYPE -- gets a compile-time ROW stride out of it:
        # `vertex_positions( v, d )` becomes `base + v * 16 + d * 8` instead of a multiplication by
        # a `long long` read from memory. A capacity, on the other hand, MUST NOT go in there: it
        # would change at every doubling, hence a kernel recompiled each time.
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
        self.itemsize = int( self.dtype.numpy_dtype.itemsize )
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
                                      self._alignment_bytes, self.itemsize, phys_num,
                                      getattr( self.inst, "item_alignment_bytes", 0 ) )
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
        dt = self.dtype.numpy_dtype
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
    def cpp_seed_root( self, var_name ):
        """The seed of a tensor passed BARE (not a member of an aggregate).

        It was missing: `JaxFfi._render_call` asks every root argument that has one for a
        `cpp_seed_root`, and only aggregates had one -- so a bare tensor output (the simplest
        case: `driver.call( ..., out = RealTensor[ ... ]() )`) was never seeded, even though
        `LOOM_ZERO_OUTPUTS` promises otherwise. Same rule as for a member, the view being here
        the variable itself."""
        return self._seed_of( var_name )

    def cpp_seed_member( self, owner_name ):
        return self._seed_of( f"{ owner_name }.{ self.name }" )

    def _seed_of( self, view ):
        """Zero a SHARED float OUTPUT of a BATCHED call, before the body runs.

        Such an output carries NONE of the call's batch axes, yet the call has some: every item
        writes the SAME buffer, so the kernel ACCUMULATES into it (e.g. a ProjectedSumOfDiracs points
        gradient, atomic-added by every angle) and it must start at zero. A per-item output (one that
        carries a batch axis) is written once per item -- no seed; and with no batch there is no
        accumulation at all. `fill_with( queue, 0 )` goes through the queue, so it is ordered before
        the body's kernel. Unbound (NoneTensor/ZeroTensor) and integer/int scratch have nothing to seed."""
        if not ( self.io_category.is_bound and self.io_category.is_output ):
            return ""

        # Every output starts at ZERO, over its whole CAPACITY, before the body.
        #
        # An output buffer that the body writes only PARTIALLY leaves the rest as the allocator
        # returned it -- and on GPU that is not zero. Two ways to get there, and the second is the
        # rule, not the exception:
        #  * a strided loop `for i = thread_index; i < n; i += nb_threads` touches nothing when
        #    `thread_index >= n`, nor does a scratch declared as an output that the forward never
        #    writes;
        #  * above all, the BATCH dimension is allocated at the aligned capacity (16 slots for
        #    4 items), and the body only walks the COUNT -- since the loop is bounded by the
        #    count and no longer by that capacity (see `CallArgsAnalysis.batch_dim_expr`), the
        #    padded tail is no longer written at all.
        #
        # What someone walking the capacity then reads is indeterminate, and an indeterminate
        # COUNT bounds a loop there: an access gigabytes outside any allocation. Seeding makes
        # that harmless without requiring anything of the readers.
        #
        # It is not free -- one fill per output and per call -- hence the switch, which now
        # serves to MEASURE what the seeding costs, not to decide whether it happens.
        #
        # THERE ARE TWO KINDS OF OUTPUTS, and only one can be poisoned.
        #
        # A SHARED output of a batched call carries no batch axis while the call has some: every
        # item writes the SAME buffer, so the kernel ACCUMULATES into it (the positions gradient of
        # a `PowerDiagram`, added by every cell; that of a `ProjectedSumOfDiracs`, by every
        # angle; the per-tile counter of `examples/splats`, by every splat). Starting from zero is
        # not a safety net there, it is the CONTRACT of the accumulation -- a poison would be
        # absorbed by the first addition and yield a perfectly legitimate-looking NaN. That one
        # starts at zero, whatever the mode.
        #
        # It was the poison mode that made the distinction visible: without it, the 14
        # DERIVATIVE tests of `test_PowerDiagram` returned `adjoint = nan` -- not a forgotten
        # write, just an accumulation poisoned in advance.
        #
        # The criterion is "SHARED", not "floating". It rested on the type until
        # `examples/splats` brought the counter-example: an INTEGER per-tile counter, accumulated
        # by all the splats. Run with `LOOM_ZERO_OUTPUTS=0`, it returns an indeterminate total that
        # becomes an ALLOCATION SIZE -- `overflow in static extent product:
        # dimensions=[2421069375325856419]`. A wrong gradient shows; so does a two-exabyte
        # allocation, but both come from the same omission.
        accumulated = ( self._call_batch_axes
                        and not any( b in self.axis_names for b in self._call_batch_axes ) )

        # AN ACCUMULATED OUTPUT STARTS AT ZERO, WHATEVER THE MODE: it is a contract, not a
        # setting. Nothing can disable it -- and `LOOM_ZERO_OUTPUTS=0` therefore does not mean
        # "nothing", it means "nothing MORE".
        if accumulated:
            return f"{ view }.fill_with( queue, 0 );"

        # The rest -- an ORDINARY output, one per item, written entirely within its logical
        # region -- is NO LONGER seeded by default, and that is where all the expense was: those
        # are the big ones. In `examples/splats`, the count is 4 KB and the list 2 MB of which 8%
        # is written; seeding the latter fills the CAPACITY, not the content.
        #
        # What that stops covering is the FORGOTTEN WRITE. But a zero did not cover it, it HID it
        # behind a plausible value -- the very argument that introduced `poison`.
        # What stays guaranteed without seeding anything: a count is always zero
        # (`CallArg_ShapeVar`, unconditional) and it is clamped to the capacity, so a read that
        # respects the count only touches written slots. The padding beyond -- the tail of an
        # aligned batch dimension, the slots after the count -- is read by nobody: `Tensor.value`
        # slices it off, and a chained call links the buffer to its LOGICAL size.
        #
        #   ( default ) what needs it: counts and accumulations
        #   poison      fills the rest with a NaN / an out-of-range integer -- the state of a test
        #               suite, where a forgotten write must FAIL
        #   all         fills the rest with zeros: the old default, to suspect an omission without
        #               catching NaNs, and to measure what the net costs
        # WARNING -- THE DEFAULT IS BACK TO "SEED EVERYTHING", and you should know why.
        #
        # An attempt to seed only what is necessary ( the counts, tiny, and the accumulations )
        # was written and then WITHDRAWN. Two things got it withdrawn:
        #
        #  * it gained nothing. The cost that grows with the capacity ( measured on
        #    `examples/splats`: 4.7 / 5.8 / 8.8 ms for a capacity x1 / x2 / x4, at CONSTANT useful
        #    work ) is not the fill but the ALLOCATION of the output buffer, which XLA redoes at
        #    every call. Not writing an overly generous capacity does not make it free.
        #  * and above all, two observations of the SAME code contradicted each other on whether
        #    an accumulated counter was still seeded: an instrumentation said yes, the generated
        #    source said no. As long as that gap is unexplained, shrinking the seeding would make
        #    correctness depend on a classification we cannot predict -- and the symptom would be
        #    invisible on Linux, which zeroes fresh pages, only to appear on GPU.
        #
        # What remains of the attempt, and is a net gain: the `accumulated` criterion no longer
        # rests on `dtype.floating_point` but on "shared", because an accumulated INTEGER counter
        # exists ( `examples/splats` ) and it was poisoned by mistake.
        #
        # What it would take to shrink for good: the caller DECLARES its accumulated outputs
        # ( `accumulated_outputs = [ ... ]` ), because "read-modify-write" is a property of the
        # BODY and not of the shapes -- `ids` and `counts` are both shared, one needs nothing,
        # the other absolutely needs it, and the analysis cannot tell them apart.

        mode = env.var( "ZERO_OUTPUTS", "" ).strip().lower()
        if mode == "poison":
            return f"{ view }.fill_with( queue, poison_value<DECAYED_TYPE_OF( { view } )::TF>() );"
        if mode not in ( "0", "false", "no", "off" ):
            return f"{ view }.fill_with( queue, 0 );"
        return ""

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
        dt = self.dtype.numpy_dtype
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

    def out_shape_dtype( self ):
        """`( shape, numpy dtype )` of the physical buffer a kernel writes -- what any driver allocates."""
        import numpy
        return tuple( int( s ) for s in self._jax_buffer_shape() ), self.dtype.numpy_dtype

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
