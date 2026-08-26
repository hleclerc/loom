"""HOW a tensor's value is held.

A `Tensor` is three separable things: its AXES (the logical contract), its CLASS (what it is made
of -- `RealTensor` / `IntTensor` / ...), and its STORAGE (how the value is actually backed). This
module is the third.

There is one variant per way a value can be backed, and they map one-to-one onto the C++ types a
member lowers to:

    Unbound       -> NoneTensor    no value at all -- an absent, optional member
    Buffer        -> TensorView    a real backend buffer, sized at CAPACITY (padding included)
    SymbolicZero  -> ZeroTensor    shaped and typed, reads as 0, no storage behind it
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

import numpy

from .PhysicalLayout import PhysicalLayout


class Storage:
    """Base variant: holds nothing. Also the shared behaviour -- every variant carries a
    `reference_shape` (the LOGICAL, unpadded sizes read off the value it was built from), because
    that is what a `ShapeVar` pulls its count from and it outlives the buffer itself."""

    # -- what this variant IS, answered by the class rather than tested for --
    holds_value      = False    # is there a value here at all? (`Tensor.is_defined`)
    is_symbolic_zero = False
    is_fill          = False

    raw = None

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
        return _tensor_view_type( arg )

    def bound_cpp_view( self, arg ):
        return _tensor_view_init( arg )

    def absent_cpp_type( self, arg ):
        """What we spell when this call does not take our value: a `NoneTensor` -- a TYPE carrying
        the declared scalar, rank and axis names and no data, discriminated at COMPILE time, not a
        degenerate view for the kernel to test at run time."""
        return _absent_type( "NoneTensor", arg )

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
        from ..drivers.driver import driver
        if raw is None:
            return Unbound( reference_shape = reference_shape )
        if driver.is_symbolic_zero( raw ):
            return SymbolicZero( raw, reference_shape )
        return Buffer( raw, reference_shape, layout )

    def __repr__( self ) -> str:
        return f"{ type( self ).__name__ }()"


class Unbound( Storage ):
    """No value: the attribute exists but holds nothing, and lowers to a `NoneTensor` -- a TYPE
    carrying the declared scalar/shape/axis names and no data, discriminated at compile time.

    It still carries a `reference_shape`: unbinding a buffer (invalidating a computed cache, say)
    does not unlearn the sizes that were observed from it, and a `ShapeVar` may still be resolving
    its count through them.

    Its lowering needs no override: bound means this call ALLOCATES our buffer (we are its
    output), so the inherited view is exactly right; unbound means there is genuinely nothing, and
    the inherited `NoneTensor` says so. Both spell the same axes, so the aggregate still
    `DEFINE_AXIS`es them either way."""


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


class SymbolicZero( Storage ):
    """A shaped, typed, STORAGELESS value that reads as 0 -- the framework's own symbolic-zero
    cotangent, or one we mint. It lowers to a `ZeroTensor`, dropped at compile time.

    It holds a value (so it is not `Unbound`) but backs no buffer (so `buffer` is `None`, and it
    binds nothing across the FFI). That is the whole distinction, and it needs no flag anywhere
    else."""

    holds_value      = True
    is_symbolic_zero = True

    def absent_cpp_type( self, arg ):
        # like a `NoneTensor`, a TYPE with no data -- except it READS AS 0 wherever indexed, so the
        # kernel needs no branch and the compiler drops the arithmetic it feeds. Only this side is
        # overridden: backing no buffer, a symbolic zero is never on the bound one.
        return _absent_type( "ZeroTensor", arg )


class Fill( Storage ):
    """A symbolic constant (`Tensor.full`): a single scalar backs the whole logical shape, which
    lives in the axes. It lowers to a storageless `FillTensor` -- `CallArg_Tensor` binds the scalar
    and spells the logical extents in the view, read from a sibling real buffer at emit time.

    Its capacity is its own reference shape, NOT recomputed from the axes: a backward residual may
    not have resolved those yet, and no per-axis capacity of ours could be inverted by a `ShapeVar`
    anyway (a fill's extents are read FROM real buffers, never the reverse).

    NB this path is prototyped but not yet wired in -- `Tensor.full` materializes (see its note)."""

    holds_value = True
    is_fill     = True

    @property
    def buffer( self ):
        return self.raw

    def capacity( self, rank ):
        return tuple( self.reference_shape.capacities() )

    def view( self, tensor ):
        from ..drivers.driver import driver
        return driver.full( tensor.shape, self.raw, dtype = tensor.dtype )

    def bound_cpp_type( self, arg ):
        # a storageless constant over the logical shape: every element reads one scalar (FillTensor.h)
        return ( f"FillTensor<{ arg.cpp_scalar() }, { arg.cpp_shape_type() }, "
                 f"{ arg.cpp_axis_names_type() }>" )

    def bound_cpp_view( self, arg ):
        # the scalar's data ptr + the LOGICAL extents, each read from a sibling REAL buffer that
        # carries the axis (the call's analysis finds it, like a batch extent) -- so no extent is
        # baked into the source, and the same compiled library serves every size.
        extents = ", ".join( arg.sibling_dim_expr( a ) for a in arg.axis_names )
        return f"{ self.bound_cpp_type( arg ) }{{ { arg.jax_data_ptr() }, tuple( { extents } ) }}"

    def jax_buffer_shape( self, arg ):
        return []       # a single scalar backs the whole (logical) fill -- a rank-0 buffer

    def batch_dim_expr( self, arg, name ):
        # we have only a scalar buffer, so we can resolve no real extent for anyone: a fill's OWN
        # extents are read FROM the real buffers, never the reverse.
        return None


# ---- spellings shared by more than one variant -------------------------------------------------
# `Unbound` (as an output) and `Buffer` lower to the SAME view -- one has its buffer, the other is
# about to be given one -- so the spelling lives here rather than being inherited from either.

def _tensor_view_type( arg ):
    # a NON-contiguous layout spells its Strides in the type too (a runtime `Tuple<SI,...>`); the
    # contiguous default leaves the template's default Strides, so a plain view is unchanged.
    strides = "" if arg.layout.is_identity else f", { arg.cpp_shape_type() }"
    return ( f"TensorView<{ arg.cpp_scalar() }, { arg.cpp_shape_type() }, "
             f"{ arg.memory_space }, { arg.cpp_axis_names_type() }{ strides }>" )


def _tensor_view_init( arg ):
    ptr = arg.jax_data_ptr()
    if arg.layout.is_identity:
        return ( f"tensor_view<{ arg.memory_space }>( { ptr }, { arg.cpp_shape_tuple() }, "
                 f"{ arg.cpp_axis_tuple() } )" )
    # non-contiguous: LOGICAL extents (literals) + physical BYTE strides -> the 4-arg overload.
    return ( f"tensor_view<{ arg.memory_space }>( { ptr }, { arg.cpp_logical_shape_tuple() }, "
             f"{ arg.cpp_axis_tuple() }, { arg.cpp_strides_tuple() } )" )


def _absent_type( kind, arg ):
    """A value with no buffer behind it -- `NoneTensor` (absent) or `ZeroTensor` (reads as 0).
    Both still spell the declared scalar, shape rank and axis names in their type: what is absent
    is the data, not the contract."""
    return ( f"{ kind }<{ arg.cpp_scalar() }, { arg.cpp_shape_type() }, "
             f"{ arg.cpp_axis_names_type() }>" )
