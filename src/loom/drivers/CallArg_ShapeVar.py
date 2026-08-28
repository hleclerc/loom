from ..tensor.Dtype import Dtype
from .CallArg_Errors import ERRORS_VAR_NAME
from .CallArg import CallArg

class CallArg_ShapeVar( CallArg ):
    """A `ShapeVar` crossing the FFI: its COUNTS, plus a `max` bound.

    It crosses in one of two ways, and which one is not a detail -- it is the difference between a
    dereference per read and a value in a register:

    * as a SCALAR (`as_scalar`): an XLA FFI attribute, landing in the kernel as a `ScalarValue<SI>`.
      This is the common case -- a plain, non-ragged count the host already knows. There is nothing
      per-element about it, so a buffer would buy nothing and cost an allocation, a transfer and an
      indirection on every read.
    * as a BUFFER of `std::int32_t`: what a count must be when it genuinely varies -- one count per
      segment of a RAGGED structure, or one per item of a batch -- or when the KERNEL is what writes
      it (an output), or when the host does not know it (a count a previous call wrote, which under
      a `jit` is a device value Python cannot read). A rank-0 count is then still bound as a
      1-element buffer (xla FFI rank-0 buffers are avoided), viewed as a rank-0 `TensorView`.

    What the count is NOT is a size: what sizes the buffers depending on this var is its
    CAPACITY, a decision made by the CALL (see `CallArgsAnalysis`). It crosses as the `max`
    bound of the `ShapeVarView`, so a written count can be capacity-checked on the C++ side --
    and a count that does not fit is recorded in the call's ERROR BUFFER, under the `error_id`
    given here, for the host to reserve more and run again.

    An unbound var wraps a `NoneTensor` instead of a view: there is no count anywhere, and the
    type says so -- writing it does not compile, rather than corrupting a null pointer.
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( call_args_analysis.io_category( path, _has_count( inst ) ), name )

        self.inst = inst
        self.path = path
        # the count Python actually holds, if it holds one -- `None` when it only lives on the
        # device (a count a kernel wrote, read back under a `jit`). Snapshotted here, like every
        # other fact this node reads off the attribute: only a count we KNOW can travel by value.
        self._static_count = inst.static_count()
        # a count is an `int32` buffer, never differentiable -- same role as `dtype` plays for a
        # `CallArg_Tensor`: it is what tells `JaxFfi._call_with_vjp` this is a trace CONSTANT, not
        # a primal to seek a gradient for (`t.dtype.floating_point`, uniform across both).
        self.dtype = Dtype.si( 32 )
        self.memory_space = call_args_analysis.cpp_memory_space
        # one count per cell of the ragged structure this var varies along (none -> a scalar).
        self.shape = [ int( s ) for axis in inst.dep_axes
                       for s in axis.capacity_list( lambda sv: call_args_analysis.capacity_of( sv, path ) ) ]
        # the batch axes, leading and NAMED (a count is per batch item too); the counts' own axes
        # stay positional -- they are cells of a ragged structure, not names. Two sources: the
        # aggregate we belong to (`Aggregate.apply_batch_axes`), taken here, and a `vmap`, which
        # prepends its own later (`add_batch_axis`).
        self.batch_axes = []

        # ... but only for a count a KERNEL WRITES. That is the whole criterion: two items can only
        # disagree about a count if a kernel is what put it there. A host-known count -- prescribed,
        # or solved off the shape of a tensor we were given (`nb_diracs`) -- is the same number for
        # every item of the batch, and giving it one slot per item would cost a buffer, a transfer
        # and an indirection to hold the value n times (it would also lose `as_scalar`, which is
        # exactly what such a count is for). `is_output` covers the count this call is about to
        # write; `is_kernel_written` the one an earlier call already did, which we are now READING
        # back -- both sides have to agree on the rank, being the same buffer.
        if self.io_category.is_output or inst.is_kernel_written():
            for axis in reversed( list( getattr( inst, "batch_axes", () ) ) ):
                self.add_batch_axis( axis.name, int( axis.max ) )

        # the bound a written count is checked against; -1 marks "unbounded" (nothing sizes
        # itself on this var, so this call had no reason to be given a capacity for it).
        try:
            self.max_bound = call_args_analysis.capacity_of( inst, path )
        except ValueError:
            self.max_bound = -1

        # a real id is handed out once the tree is built (see `wants_error_id`); -1 until then, and
        # for good if we need none.
        self.error_id = -1

    @property
    def as_scalar( self ):
        """Whether this count crosses BY VALUE rather than through a buffer (see the class doc).

        A lazy property, not a field: a `vmap` gives us a batch axis AFTER `__init__`, and one
        count per batch item is no longer a single value -- deciding in the constructor would
        freeze the answer before that is known.

        Three conditions, covering the four reasons a buffer is unavoidable: the kernel must not be
        WRITING it (`is_input`), it must have no dimensions -- neither RAGGED segments nor one cell
        per batch item (`not self.shape`) -- and the host must actually know the count.

        That last one is `static_count`, deliberately: a count PRESCRIBED in Python, or solved from
        the shape of a tensor we were given, is a fact that holds whether or not we are tracing. A
        count a KERNEL wrote is not -- eagerly Python could read it back, but under a `jit` it is a
        tracer, and an FFI attribute cannot be one. Keying on `static_count` keeps the generated
        signature identical in both, instead of compiling a second library for the eager case."""
        return ( self.io_category.is_input
                 and not self.shape                     # rank 0: not ragged, and no batch axis
                 and self._static_count is not None )

    def is_ffi_buffer( self ):
        return self.io_category.is_bound and not self.as_scalar

    def wants_error_id( self ):
        """Whether we need a slot in the error buffer: only a count this call WRITES, and that
        something is sized on, can overflow a capacity. The others carry a `NoErrorBuffer`, which
        compiles away -- so they never take an id."""
        return self.io_category.is_output and self.max_bound >= 0

    # -- as a value a `vmap` maps over: one count per batch item --
    def add_batch_axis( self, name, size ):
        self.batch_axes = [ name ] + self.batch_axes
        self.shape = [ int( size ) ] + self.shape

    def batch_dim_expr( self, name ):
        if name not in self.batch_axes:
            return None
        return self.jax_dim( self.batch_axes.index( name ) )

    # -- the axes our type spells (see `CallArg.cpp_axis_names`): only the NAMED batch ones -- a
    # count's own (ragged) dimensions are positional, spelled `UnnamedAxis` (see `_cpp_axis_tuple`).
    def cpp_axis_names( self ):
        return self.batch_axes

    # -- driver-agnostic C++ (the same for every driver) --
    def _cpp_shape_tuple( self ):
        # the extents come from the BUFFER, not from `self.shape`: see `CallArg.jax_dim`.
        return "tuple( " + ", ".join( self.jax_dim( d ) for d in range( len( self.shape ) ) ) + " )"

    def _cpp_axis_tuple( self ):
        # the batch axes are named (the kernel selects them by name); the count's own axes are not.
        names = self.batch_axes + [ "UnnamedAxis{}" ] * ( len( self.shape ) - len( self.batch_axes ) )
        return "tuple( " + ", ".join( names ) + " )"

    def _cpp_max_bound( self ):
        """The capacity a written count is checked against. A CALL parameter, so it reaches the
        kernel as an FFI attribute -- as a literal it would recompile the kernel for each
        capacity. Unbound: nothing crosses, so the literal is all there is.

        A SCALAR count is read-only (`set` is never reached on a value passed by copy), so there is
        no capacity to check and nothing to pass: `-1`, the "unbounded" marker, is a constant and
        therefore costs no recompile either."""
        if self.as_scalar:
            return "SI( -1 )"
        if not self.io_category.is_bound:
            return f"SI( { self.max_bound } )"
        return f"SI( { self._jax_attr_name() } )"

    def _cpp_shape_type( self ):
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_counts_type( self ):
        # counts use unnamed axes (they are positional).
        if self.as_scalar:
            return "ScalarValue<SI>"          # no pointer, no memory space: the value itself
        if not self.io_category.is_bound:
            return f"NoneTensor<std::int32_t, { self._cpp_shape_type() }, Tuple<>>"
        return f"TensorView<std::int32_t, { self._cpp_shape_type() }, { self.memory_space }>"

    def _cpp_errors( self ):
        """The error buffer this var records into -- the call's, the one and only. A var that
        cannot overflow gets a `NoErrorBuffer` instead: a type, so nothing crosses and the check
        compiles away."""
        if self.error_id < 0:
            return "NoErrorBuffer{}"
        return ERRORS_VAR_NAME

    def cpp_type( self ):
        return f"ShapeVarView<{ self._cpp_counts_type() }, NoErrorBuffer>"

    def cpp_view( self ):
        if self.as_scalar:
            # the count arrives as an FFI attribute (an int64 in the call frame) and is held by
            # value; nothing here dereferences anything.
            return ( f"make_shape_var_view( ScalarValue<SI>{{ SI( { self._jax_count_attr_name() } ) }}, "
                     f"{ self._cpp_max_bound() }, NoErrorBuffer{{}}, SI( -1 ) )" )
        if not self.io_category.is_bound:
            return ( f"{ self.cpp_type() }{{ { self._cpp_counts_type() }{{}}, "
                     f"{ self._cpp_max_bound() }, NoErrorBuffer{{}}, SI( -1 ) }}" )
        view = ( f"tensor_view<{ self.memory_space }>( { self.jax_data_ptr() }, "
                 f"{ self._cpp_shape_tuple() }, { self._cpp_axis_tuple() } )" )
        return ( f"make_shape_var_view( { view }, { self._cpp_max_bound() }, "
                 f"{ self._cpp_errors() }, SI( { self.error_id } ) )" )

    # -- as a member of an aggregate --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    def cpp_root_decl( self, var_name ):
        return f"    auto { var_name } = { self.cpp_view() };"

    # -- seeding: what an output must hold before the body runs --
    # A pure output starts at whatever XLA left in the buffer, and the body may only increment
    # it, so it has to be zeroed first. Through the QUEUE, not by a host loop: the buffer lives
    # where the kernel runs, which on a GPU is somewhere the host cannot write. A scalar count
    # broadcasts over all the cells of a ragged one, so this is right at any rank.
    def cpp_seed_member( self, owner_name ):
        if not self.io_category.is_output:
            return ""
        return f"{ owner_name }.{ self.name }.fill_with( queue, 0 );"

    def cpp_seed_root( self, var_name ):
        if not self.io_category.is_output:
            return ""
        return f"    { var_name }.fill_with( queue, 0 );"

    # -- Jax FFI ABI --
    def _jax_attr_name( self ):
        # the FFI attributes share one flat namespace, like the buffers: name it after ours.
        return f"max_{ self.ffi_name }"

    def _jax_count_attr_name( self ):
        # a scalar count has no `ffi_name` (it binds no buffer), so it is named after its PATH --
        # unique within a call by construction, which is what the flat namespace needs.
        return "count_" + self.path.replace( ".", "_" )

    def jax_attrs( self ):
        """The scalars this node needs at run time, but NOT through a buffer: an XLA FFI
        attribute is baked into the call, not into the kernel, so the same compiled kernel serves
        every capacity -- and every count.

        A SCALAR count is itself one of them (that is how it crosses). A BOUND buffer var passes
        its capacity bound. An unbound one passes nothing: it has no buffer, so its bound is a
        literal in the source (`_cpp_max_bound`), and it never got an `ffi_name` to name an attr
        after."""
        if self.as_scalar:
            return [ ( self._jax_count_attr_name(), "int64_t", int( self._static_count ) ) ]
        if not self.io_category.is_bound:
            return []
        return [ ( self._jax_attr_name(), "int64_t", int( self.max_bound ) ) ]

    def _jax_buffer_shape( self ):
        # rank-0 count -> a 1-element buffer (avoid xla FFI rank-0 buffers).
        return self.shape if self.shape else [ 1 ]

    def jax_ffi_type( self ):
        return f"ffi::BufferR{ len( self._jax_buffer_shape() ) }<ffi::S32>"

    def jax_cpp_init( self ):
        return self.cpp_view()

    def jax_input_array( self ):
        from ..drivers.driver import driver
        return driver.array( self.inst.raw, dtype = Dtype.si( 32 ) ).reshape( self._jax_buffer_shape() )

    def jax_out_spec( self ):
        import jax
        import jax.numpy as jnp
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), jnp.int32 )

    def jax_write_back( self, array ):
        # the count keeps the rank it was declared with (a scalar count came back as a
        # 1-element buffer). Nothing else moves: what the kernel wrote is a count, not a size.
        self.inst.set_count( array.reshape( self.shape ) )


def _has_count( inst ):
    # an unresolved ShapeVar (neither prescribed nor constrained by a tensor) has no count yet.
    return inst.raw is not None
