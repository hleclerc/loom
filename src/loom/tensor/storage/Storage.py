"""HOW a tensor's value is held.

A `Tensor` is three separable things: its AXES (the logical contract), its CLASS (what it is made
of -- `RealTensor` / `IntTensor` / ...), and its STORAGE (how the value is actually backed). This
module is the third.

There is one variant per way a value can be backed, and they map one-to-one onto the C++ types a
member lowers to:

    Unbound       -> NoneTensor    no value at all -- an absent, optional member
    Buffer        -> TensorView    a real backend buffer, sized at CAPACITY (padding included)
    Zero          -> ZeroTensor    shaped and typed, reads as 0, no storage behind it
    Fill          -> FillTensor    one scalar backing a whole logical shape

Everything that used to be a boolean on `Tensor` (`is_fill`, `is_symbolic_zero`, an explicit
`_layout`, `_raw is None`) is a property of the variant, so no caller tests for a kind: it asks,
and the variant answers.

That includes the C++ LOWERING: each variant spells its own type and view (`cpp_type` /
`cpp_view`), given a `CallArg_Tensor` that supplies the spelling primitives (the scalar type, the
shape/axis tuples, the data pointer, the layout). The choice of C++ form belongs to how a value is
backed; how each fragment is written belongs to the codegen. Adding a way of being backed -- a
`Range`, a strided window, a broadcast -- is then a new variant here, with nothing to change in the
lowering itself.

Two different things are called "raw", and keeping them apart is most of the point:

* `raw`    -- the object HELD, whatever its nature (a buffer, the framework's symbolic-zero
              object, a fill's single scalar). `None` only when there is genuinely nothing.
* `buffer` -- the MATERIALIZED buffer, what the FFI can bind. `None` for a value with no storage
              behind it, which is how a symbolic zero stays unbound without a special case.
"""

import copy

from ...drivers import framework_defaults

from ..PhysicalLayout import PhysicalLayout
from .cpp_spellings import absent_type, tensor_view_init, tensor_view_type


class Storage:
    """Base variant: holds nothing. Also the shared behaviour -- every variant carries a
    `reference_shape` (the LOGICAL, unpadded sizes read off the value it was built from), because
    that is what a `ShapeVar` pulls its count from and it outlives the buffer itself."""

    # -- what this variant IS, answered by the class rather than tested for --
    holds_value      = False    # is there a value here at all? (`Tensor.is_defined`)
    is_symbolic_zero = False
    is_fill          = False

    raw = None

    # WHICH framework holds our buffer (`None`: none does -- nothing, a zero, a fill's bare scalar), and
    # whether it TRACES (belongs to a machinery -- a jit, an autograd tape -- that a call must then
    # be a node of). The buffer classes answer them (see `buffers/`), so nothing above tests for a
    # framework by name.
    framework = None

    @property
    def traces( self ) -> bool:
        return False

    def device( self ):
        """`( "cpu", 0 )` or `( "cuda", index )`: where the buffer lives, or `None` when we cannot tell (no
        buffer, a value inside a trace)."""
        return None

    def is_strided( self ) -> bool:
        """Held otherwise than dense row-major: a view the kernel reads with its own strides. Nothing
        is, without a buffer."""
        return False

    def __init__( self, raw = None, reference_shape = None ) -> None:
        self.raw = raw
        self.reference_shape = reference_shape

    @property
    def buffer( self ):
        """The materialized buffer the FFI can bind, or `None` when nothing backs this value."""
        return None

    def retyped( self, coerce ):
        """The SAME kind of storage, holding `coerce( raw )` -- how a value keeps its nature when
        it is bound to another tensor (a fill stays a fill, a symbolic zero stays one), while that
        tensor's declared dtype is still enforced on whatever backs it."""
        res = copy.copy( self )
        res.raw = coerce( self.raw )
        return res

    def with_reference_shape( self, reference_shape ):
        res = copy.copy( self )
        res.reference_shape = reference_shape
        return res

    # -- physical questions, all asked with the tensor's LOGICAL rank / extents --
    def layout( self, rank ):
        """The physical arrangement of the buffer relative to the logical axes. With nothing
        allocated there are no extents to describe, so it is the empty contiguous one."""
        return PhysicalLayout.contiguous( [ 0 ] * rank )

    def capacity( self, rank ):
        """The allocated extents per LOGICAL dimension -- what our buffer IS. An input is bound at
        THIS size, so an output that wants to grow cannot force us to inflate it."""
        return tuple( self.layout( rank ).caps )

    def allocated_sizes( self, rank ):
        """The per-axis capacity a `ShapeVar` inverts to learn what it was allocated with, or
        `None` when there is no allocation to read it from."""
        return None

    def view( self, tensor ):
        """The LOGICAL data: the buffer's meaningful region, capacity padding cropped off.
        `None` when there is nothing to view."""
        return None

    # ---- C++ lowering: what this way of being backed spells in the kernel --------------------
    # `arg` is the `CallArg_Tensor` lowering us; it supplies the spelling primitives (see its
    # `cpp_*` helpers). Nothing here knows about Jax vs Torch.
    #
    # Two questions, and they compose: the CALL decides whether the value crosses at all, the
    # STORAGE decides what form it takes when it does. Holding a buffer does not make a member an
    # argument -- a call may exclude it (`input_exceptions`) -- and holding nothing does not make it
    # absent, since an OUTPUT is exactly a member this call is about to allocate a buffer for. So a
    # variant answers on both sides, and the default pair below is already right for both `Unbound`
    # (bound <=> output <=> a buffer we are given) and `Buffer` (bound <=> the buffer we have).
    def cpp_type( self, arg ):
        return self.bound_cpp_type( arg ) if arg.io_category.is_bound else self.absent_cpp_type( arg )

    def cpp_view( self, arg ):
        if arg.io_category.is_bound:
            return self.bound_cpp_view( arg )
        return f"{ self.cpp_type( arg ) }{{}}"      # nothing to view: it value-initializes

    def bound_cpp_type( self, arg ):
        return tensor_view_type( arg )

    def bound_cpp_view( self, arg ):
        return tensor_view_init( arg )

    def absent_cpp_type( self, arg ):
        """What we spell when this call does not take our value: a `NoneTensor` -- a TYPE carrying
        the declared scalar, rank and axis names and no data, discriminated at COMPILE time, not a
        degenerate view for the kernel to test at run time."""
        return absent_type( "NoneTensor", arg )

    def jax_buffer_shape( self, arg ):
        """The PHYSICAL buffer XLA allocates / binds for us -- the layout's `buffer_shape`
        (flattened + padded batch when non-contiguous). The C++ view reinterprets it logically."""
        return arg.layout.buffer_shape

    def batch_dim_expr( self, arg, name ):
        """Where the SIZE of axis `name` can be read at run time off our buffer, or `None` when we
        cannot answer for it."""
        if name not in arg.axis_names:
            return None
        return arg.jax_dim( arg.axis_names.index( name ) )

    @staticmethod
    def of( raw, reference_shape = None, layout = None ):
        """The variant `raw` calls for. The only place a value's nature is DECIDED; everywhere
        else it is carried. A fill is never inferred -- one scalar looks like any other rank-0
        buffer, so being a fill is something a construction site states (see `Tensor.full`)."""
        from .Buffer import Buffer
        from .buffers import buffer_class_for
        from .Zero import Zero
        from .Unbound import Unbound
        if raw is None:
            return Unbound( reference_shape = reference_shape )
        if framework_defaults.ops( raw ).is_symbolic_zero( raw ):
            return Zero( raw, reference_shape )
        return ( buffer_class_for( raw ) or Buffer )( raw, reference_shape, layout )

    def __repr__( self ) -> str:
        return f"{ type( self ).__name__ }()"
