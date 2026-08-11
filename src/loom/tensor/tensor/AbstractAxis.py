from ..util.Attribute import Attribute, resolve_attribute


class AbstractAxis( Attribute ):
    """Common base for `Axis` and `AxisList`.

    Both name a tensor dimension whose extent is an affine expression of
    `ShapeVar`s. The shared part -- factored here -- is the *parsing* of that
    expression into `coeffs` (a `{ ShapeVar: int }` map) and an integer
    `offset`, plus the single-variable inversion used to solve a `ShapeVar`
    from an observed size.

    The expression is given either as a string (`Axis[ "2 * nb_dims + 1" ]`,
    whose names an aggregate resolves) or, outside of any aggregate, as the
    `ShapeVar` itself (`Axis( nb_dims )`).

    They differ in what is known at declaration time (virtual `register_in`):
    an `Axis` is a single dimension (ragged when one of its `ShapeVar`s has
    rank > 0); an `AxisList` is a *family* of dimensions to be unrolled, whose
    count (`nb_dims`) is unknown when the class is declared.
    """

    def __init__( self, *exprs, template_args = (), template_kwargs = {}, scope = None, name = None ) -> None:
        from .ShapeVar import ShapeVar

        self.coeffs: dict[ ShapeVar, int ] = {}
        self.offset = 0
        self.name = name

        # a batch axis sorts BEFORE ordinary ones in the logical layout an elementwise op produces
        # (see `Tensor._binary`). A plain axis is not one; `new_batch_axis` sets this. Physical axis
        # order (e.g. a leading batch on GPU) is a separate, later concern -- it will not touch this.
        self.is_batch = False

        # declared (`Axis[ "nb_dims + 1" ]`) or built directly (`Axis( nb_dims )`): same args,
        # two ways in.
        self._init_axis( list( template_args ) + list( exprs ), scope )

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        # An axis lowers to NOTHING: it is a declaration, not data. Its extent is already baked
        # into the shape of every tensor that uses it, and its name is registered by those
        # tensors (a tensor may borrow an axis from an object that is not even an argument).
        return None

    def cpp_axis_names( self ):
        """The axis name(s) this declaration needs `DEFINE_AXIS`'d in C++. An aggregate collects
        these so every axis it declares is spelled in its header -- even one no tensor of the call
        references (`num_edge`): a body may still name it, and the C++ type must exist for it."""
        return [ self.name ]

    def cpp_dim_names( self, index ):
        """The C++ name(s) for the ARRAY dimension(s) this axis expands into: the NAME analogue of
        `max_list` (which does the same for extents). One entry for a plain `Axis`; several for an
        unrolled `AxisList`. Keeping the unrolling HERE (and in the overrides) lets a caller merely
        concatenate over a tensor's axes -- it needs no notion of how many `AxisList`s there are or
        how wide each unrolls. `index` is the axis' position, used only for the nameless fallback."""
        return [ self.name or f"a{ index }" ]

    @staticmethod
    def cpp_shared_header( name ):
        """The shared header that DECLARES the axis `name`: `DEFINE_AXIS` (behind AxisNames.h),
        which spells the type `_name` a tensor references and the `name` object a body indexes
        with. Returns its include path. It lives here because the C++ facet of an axis is the
        axis's business, not the call's -- the call only asks for it by name."""
        from ..compilation.generated_headers import shared_header
        content = ( "#pragma once\n\n"
                    '#include "sdot/support/containers/AxisNames.h"\n\n'
                    f"DEFINE_AXIS( { name } );\n" )
        return shared_header( f"sdot/generated/axes/{ name }.h", content )

    # ---- shared affine parser ----
    @staticmethod
    def parse_affine( expr ):
        """Pure parse of an affine expression like "2 * nb_dims + 3 * nb_xs + 1" into
        `( { var_name: coeff }, offset )`. No `ShapeVar` resolution -- names stay strings, so
        this is usable without a parent instance (e.g. shape resolution in `CallArg_Tensor`).
        Spaces are dropped and subtraction becomes the addition of a negative term."""
        coeffs = {}
        offset = 0
        expr = expr.replace( " ", "" ).replace( "-", "+-" )
        for term in ( t for t in expr.split( "+" ) if t ):
            if term.lstrip( "-" ).isdigit():
                offset += int( term )
            elif "*" in term:
                coeff_str, var_name = term.split( "*", 1 )
                coeffs[ var_name ] = coeffs.get( var_name, 0 ) + int( coeff_str )
            else:
                name = term.lstrip( "+-" )
                coeffs[ name ] = coeffs.get( name, 0 ) + ( -1 if term[ 0 ] == "-" else 1 )
        return coeffs, offset

    def _parse_expr( self, expr, scope ):
        """Instance-side parse: fill `coeffs` (each name resolved to its `ShapeVar` in `scope`)
        and `offset`. A `ShapeVar` handed over directly is the degenerate expression `1 * var`,
        and needs no scope."""
        from .ShapeVar import ShapeVar
        if isinstance( expr, ShapeVar ):
            self._add_coeff( expr, 1, scope )
            return

        coeffs, offset = self.parse_affine( expr )
        self.offset += offset
        for var_name, coeff in coeffs.items():
            self._add_coeff( var_name, coeff, scope )

    def _add_coeff( self, var, coeff, scope ):
        from .ShapeVar import ShapeVar
        shape_var = resolve_attribute( var, scope, ShapeVar )
        self.coeffs[ shape_var ] = self.coeffs.get( shape_var, 0 ) + coeff

    def solve_single( self, shape_var, size ):
        """Invert the single-variable affine `size = coeff * shape_var + offset`.

        Returns `(size - offset) // coeff` (a scalar or a per-segment array,
        following the shape of `size`), or `None` if `shape_var` is not the sole
        variable of this axis (the multi-variable case is a TODO linear solve)."""
        if list( self.coeffs.keys() ) != [ shape_var ]:
            return None
        return ( size - self.offset ) // self.coeffs[ shape_var ]

    def set( self, value ):
        raise RuntimeError( "An axis cannot be set" )

    # ---- extents (virtual) ----
    def max_list( self ):
        """The member's extents as a list to be concatenated into a tensor shape:
        one entry for an `Axis`, `nb_dims` entries for an unrolled `AxisList`."""
        raise NotImplementedError

    def array_dims( self, tensor ):
        """How many ARRAY dimensions this axis occupies on `tensor`: one for a plain `Axis`, its
        unroll width for an `AxisList` (which overrides). The count lives on the AXIS, so a tensor
        never has to do `ndim - n_plain` arithmetic itself, nor know how many lists it holds."""
        return 1

    # ---- usage registration (virtual) ----
    def register_in( self, tensor ):
        """Record on our `ShapeVar`s that `tensor` (which declares us among its axes) constrains them,
        so they can be SOLVED (pulled) from it on demand -- from its logical sizes for the count, from
        its buffer for the allocated capacity. We find our OWN position in `tensor` (`_dim_index`) so
        the caller passes no index. A plain axis is one dimension; an `AxisList` overrides (it spans
        several)."""
        self._register_dense( tensor )

    def _register_dense( self, tensor ):
        """One declared axis <-> its own array dimension. Two resolvers per ShapeVar, both mapping our
        axis to its array dimension (`_spec_dims`, which accounts for an unrolled sibling): `logical`
        inverts our affine on the LOGICAL size there (`_shape.sizes(...)`, a 0-d scalar for a dense
        axis or a per-segment array for a ragged one -- unpadded); `capacity` on the ALLOCATED buffer
        size (`allocated_sizes` at that dimension). Our position is resolved at PULL time (`_dim_index`),
        by then `tensor.axes` is complete."""
        for shape_var in self.coeffs:
            def logical( t, axis = self, shape_var = shape_var ):
                if t._shape is None:
                    return None
                dim = t._spec_dims()[ t._dim_index( axis ) ]
                return axis.solve_single( shape_var, t._shape.sizes( dim ) )
            def capacity( t, axis = self, shape_var = shape_var ):
                sizes = t.allocated_sizes
                if sizes is None:
                    return None
                dim = t._spec_dims()[ t._dim_index( axis ) ]
                return axis.solve_single( shape_var, sizes[ dim ] )
            shape_var.add_usage( tensor, logical, capacity )

    # ---- extents (virtual) ----
    def capacity_list( self, capacity_of ):
        """The member's extents for an allocation, as a list to be concatenated into a tensor
        shape -- `capacity_of( shape_var )` being what the CALL decided to allocate for each of
        our variables (a capacity is never our own state, see `ShapeVar`)."""
        raise NotImplementedError

    # ---- virtual ----
    def _init_axis( self, args, scope ):
        raise NotImplementedError
