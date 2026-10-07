"""The C++ spellings more than one storage variant shares."""


# ---- spellings shared by more than one variant -------------------------------------------------
# `Unbound` (as an output) and `Buffer` lower to the SAME view -- one has its buffer, the other is
# about to be given one -- so the spelling lives here rather than being inherited from either.

def tensor_view_type( arg ):
    # a NON-contiguous layout spells its Strides in the type too (a runtime `Tuple<SI,...>`); the
    # contiguous default leaves the template's default Strides, so a plain view is unchanged.
    if arg.runtime_strides:
        strides = f", { arg.cpp_runtime_strides_type() }"
    else:
        strides = "" if arg.layout.is_identity else f", { arg.cpp_shape_type() }"
    return ( f"TensorView<{ arg.cpp_scalar() }, { arg.cpp_shape_type() }, "
             f"{ arg.memory_space }, { arg.cpp_axis_names_type() }{ strides }>" )


def tensor_view_init( arg ):
    ptr = arg.jax_data_ptr()
    if arg.runtime_strides:
        return ( f"tensor_view<{ arg.memory_space }>( { ptr }, { arg.cpp_shape_tuple() }, "
                 f"{ arg.cpp_axis_tuple() }, { arg.cpp_runtime_strides_tuple() } )" )
    if arg.layout.is_identity:
        return ( f"tensor_view<{ arg.memory_space }>( { ptr }, { arg.cpp_shape_tuple() }, "
                 f"{ arg.cpp_axis_tuple() } )" )
    # non-contiguous: LOGICAL extents (literals) + physical BYTE strides -> the 4-arg overload.
    return ( f"tensor_view<{ arg.memory_space }>( { ptr }, { arg.cpp_logical_shape_tuple() }, "
             f"{ arg.cpp_axis_tuple() }, { arg.cpp_strides_tuple() } )" )


def absent_type( kind, arg ):
    """A value with no buffer behind it -- `NoneTensor` (absent) or `ZeroTensor` (reads as 0).
    Both still spell the declared scalar, shape rank and axis names in their type: what is absent
    is the data, not the contract."""
    return ( f"{ kind }<{ arg.cpp_scalar() }, { arg.cpp_shape_type() }, "
             f"{ arg.cpp_axis_names_type() }>" )
