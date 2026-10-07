from .Storage import Storage


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
        from ...drivers import framework_defaults
        return framework_defaults.full( tensor.shape, self.raw, dtype = tensor.dtype )

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
