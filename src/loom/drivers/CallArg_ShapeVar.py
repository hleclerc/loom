from ..tensor.Dtype import Dtype
from .CallArg_Errors import ERRORS_VAR_NAME
from .CallArg import CallArg

class CallArg_ShapeVar( CallArg ):
    """A `ShapeVar` crossing the FFI: a buffer of `std::int32_t` COUNTS, plus a `max` bound.

    The count is a device value -- read by the kernel when the var has one, written by it when
    the var is an output. It is rank 0 for a plain count, rank > 0 for a ragged one (one count
    per segment, along its `dep_axes`). A rank-0 count is still bound as a 1-element buffer
    (xla FFI rank-0 buffers are avoided), viewed as a rank-0 `TensorView` over its pointer.

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
        # a count is an `int32` buffer, never differentiable -- same role as `dtype` plays for a
        # `CallArg_Tensor`: it is what tells `JaxFfi._call_with_vjp` this is a trace CONSTANT, not
        # a primal to seek a gradient for (`t.dtype.floating_point`, uniform across both).
        self.dtype = Dtype.si( 32 )
        self.memory_space = call_args_analysis.cpp_memory_space
        # one count per cell of the ragged structure this var varies along (none -> a scalar).
        self.shape = [ int( s ) for axis in inst.dep_axes
                       for s in axis.capacity_list( lambda sv: call_args_analysis.capacity_of( sv, path ) ) ]
        # the batch axes a `vmap` gave us, leading and NAMED (a count is per batch item too); the
        # counts' own axes stay positional -- they are cells of a ragged structure, not names.
        self.batch_axes = []

        # the bound a written count is checked against; -1 marks "unbounded" (nothing sizes
        # itself on this var, so this call had no reason to be given a capacity for it).
        try:
            self.max_bound = call_args_analysis.capacity_of( inst, path )
        except ValueError:
            self.max_bound = -1

        # a real id is handed out once the tree is built (see `wants_error_id`); -1 until then, and
        # for good if we need none.
        self.error_id = -1

    def is_ffi_buffer( self ):
        return self.io_category.is_bound

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
        capacity. Unbound: nothing crosses, so the literal is all there is."""
        if not self.io_category.is_bound:
            return f"SI( { self.max_bound } )"
        return f"SI( { self._jax_attr_name() } )"

    def _cpp_shape_type( self ):
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_counts_type( self ):
        # counts use unnamed axes (they are positional).
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

    def jax_attrs( self ):
        """The scalars this node needs at run time, but NOT through a buffer: an XLA FFI
        attribute is baked into the call, not into the kernel, so the same compiled kernel serves
        every capacity.

        Only a BOUND var crosses one: an unbound one has no buffer, so its bound is a literal in
        the source (`_cpp_max_bound`), and it never got an `ffi_name` to build an attr name from."""
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
