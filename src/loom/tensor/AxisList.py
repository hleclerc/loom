import numpy

from ..util.Attribute import resolve_attribute
from .AbstractAxis import AbstractAxis


class AxisList( AbstractAxis ):
    """A *family* of axes indexed by a loop axis, meant to be UNROLLED.

    `AxisList[ loop_axis, expr ]`: `loop_axis` is the axis to unroll over
    (e.g. `dim`), `expr` the affine extent of each member (e.g. `extent`, with
    `extent : ShapeVar[ "dim" ]` holding one count per loop index).

    Used in a `Tensor` declaration with a trailing `...` (`Tensor[ "img_pos..." ]`)
    it expands into `nb_dims` separate static axes, giving the tensor a DYNAMIC
    rank. The count `nb_dims` is unknown at declaration time -- hence the split
    from `Axis` (a single, ragged-or-not, dimension needs no unrolling)."""

    def _init_axis( self, args, scope ):
        assert len( args ) == 2, "an AxisList takes a loop axis and an extent expression"
        self.loop_axis = resolve_attribute( args[ 0 ], scope, AbstractAxis )
        self._parse_expr( args[ 1 ], scope )

    def cpp_axis_names( self ):
        # A family declares NO single axis up front: its count (`nb_dims`) is unknown at declaration
        # time, so its members are named per-tensor, one DISTINCT axis per spanned dimension
        # (`img_pos_0`, `img_pos_1`, ... -- see `cpp_dim_names`). Nothing to `DEFINE_AXIS` here.
        return []

    def cpp_dim_names( self, index ):
        # Unroll into `loop_axis.max` DISTINCT ordinary names. The loop axis (e.g. `nb_dims`) IS a
        # real axis, so asking it for its max gives the member count -- the SAME source `max_list`
        # uses for the extents, and usually resolved because that axis is shared with other tensors.
        # An `AxisList` changes NOTHING about the tensor: it only DEFINES several ordinary axes, so
        # each unrolled dimension gets its own `_k`-suffixed name and is indexed positionally.
        base = self.name or f"a{ index }"
        return [ f"{ base }_{ k }" for k in range( self.loop_axis.max ) ]

    def array_dims( self, tensor ):
        # How many array dimensions this list spans on `tensor`. Prefer ASKING the loop axis its max:
        # `dim` (extent `nb_dims`) is a real, SHARED axis, so once any tensor -- or a prescription --
        # pins `nb_dims`, the width is known WITHOUT looking at this buffer. That is the general path,
        # and it assumes nothing about how many `AxisList`s a tensor holds.
        count = self.loop_axis.max
        if count is not None:
            return count

        # Last resort: this tensor is the ONLY witness of the loop count (e.g. just `values` set, so
        # `nb_dims` is read FROM it). The width is then the total array-dim count minus what the
        # siblings take -- from `_shape` (logical) if we have it, else the buffer rank (capacity).
        total = ( len( tensor._shape ) if tensor._shape is not None
                  else tensor._raw.ndim if tensor._raw is not None else None )
        return self._structural_width( tensor, total )

    def _structural_width( self, tensor, total ):
        # Our unroll width from STRUCTURE alone: `total` array dims minus what the siblings take,
        # WITHOUT consulting our own loop axis -- the loop resolvers use this to RESOLVE that axis, so
        # asking it would be circular. Each sibling is asked its own `array_dims` (a plain axis -> 1,
        # which never recurses back into us). `Tensor` guarantees at most one unrolled list, so this
        # remainder is unambiguous; the arithmetic stays HERE, never spread into `Tensor`.
        if total is None:
            return None
        others = 0
        for axis in tensor.axes:
            if axis is self:
                continue
            n = axis.array_dims( tensor )
            if n is None:
                return None
            others += n
        return total - others

    def max_list( self ):
        # one extent per loop index: `offset + sum( coeff * shape_var[k] )`, where
        # the loop count is the loop axis' extent (`nb_dims`) and each coeff's
        # ShapeVar is a rank-1 vector of that length.
        res = numpy.full( self.loop_axis.max, self.offset, dtype = int )
        for shape_var, m in self.coeffs.items():
            res = res + m * numpy.asarray( shape_var.raw, dtype = int )
        return [ int( x ) for x in res ]

    def capacity_list( self, capacity_of ):
        # an unrolled AxisList is dense: it holds no reservation, so its extents ARE its counts
        # (each of its ShapeVars is a vector, which a scalar capacity could not describe anyway).
        return self.max_list()

    def register_in( self, tensor ):
        # This family spans several array dimensions of `tensor` (we find our own position, so no
        # index is passed -- `_dim_index`, resolved at PULL time when `tensor.axes` is complete). Two
        # resolvers per ShapeVar, `logical` reading `_shape` (the value's unpadded sizes) and
        # `capacity` reading the buffer over the span:
        #  - the loop axis (`nb_dims`): its count IS our width, taken from STRUCTURE (`_structural_width`)
        #    -- NOT from `loop_axis.max`, which is the very axis we are resolving (that would recurse);
        #  - each member: logical by inverting its affine on every logical size over the span, capacity
        #    on the buffer sizes there. An unrolled tensor is dense (no padding), so the two agree --
        #    but we keep them distinct so capacity stays a `_raw` fact.
        for shape_var in self.loop_axis.coeffs:
            def loop_logical( t, axis = self.loop_axis, shape_var = shape_var, list_axis = self ):
                if t._shape is None:
                    return None
                width = list_axis._structural_width( t, len( t._shape ) )
                if width is None:
                    return None
                return axis.solve_single( shape_var, numpy.array( width, dtype = int ) )
            def loop_capacity( t, axis = self.loop_axis, shape_var = shape_var, list_axis = self ):
                if t._raw is None:
                    return None
                width = list_axis._structural_width( t, t._raw.ndim )
                if width is None:
                    return None
                return axis.solve_single( shape_var, numpy.array( width, dtype = int ) )
            shape_var.add_usage( tensor, loop_logical, loop_capacity )

        for shape_var in self.coeffs:
            def member_logical( t, axis = self, shape_var = shape_var ):
                if t._shape is None:
                    return None
                span = t._unroll_span( t._dim_index( axis ) )
                if span is None:
                    return None
                start, count = span
                vals = [ axis.solve_single( shape_var, t._shape.sizes( start + k ) ) for k in range( count ) ]
                if any( v is None for v in vals ):
                    return None
                return numpy.array( vals, dtype = int )
            def member_capacity( t, axis = self, shape_var = shape_var ):
                span  = t._unroll_span( t._dim_index( axis ) )
                sizes = t.allocated_sizes
                if span is None or sizes is None:
                    return None
                start, count = span
                vals = [ axis.solve_single( shape_var, sizes[ start + k ] ) for k in range( count ) ]
                if any( v is None for v in vals ):
                    return None
                return numpy.array( vals, dtype = int )
            shape_var.add_usage( tensor, member_logical, member_capacity )
