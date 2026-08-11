from .AbstractAxis import AbstractAxis


class Axis( AbstractAxis ):
    """A single named tensor dimension whose extent is an affine expression of
    `ShapeVar`s (`nb_dims + 1`). A single `ShapeVar` can drive several axes,
    which is why `ShapeVar` and `Axis` are distinct.

    The axis is RAGGED when its extent depends on a `ShapeVar` of rank > 0
    (e.g. `Axis[ "extent + 1" ]` with `extent : ShapeVar[ "dim" ]`): each
    row/col/... then has a different length and there is no single extent.

    Stateless: the concrete extent is a pure function of the `ShapeVar` cells,
    so `c.dim` reads back the evaluated extent (an `int`, or `None` while
    still unsolved)."""

    def _init_axis( self, args, scope ):
        assert len( args ) == 1, "an Axis takes exactly one extent expression"
        self._parse_expr( args[ 0 ], scope )

    @property
    def max( self ):
        res = self.offset
        for shape_var, m in self.coeffs.items():
            # an unresolved ShapeVar leaves the extent unsolved: there is no int to hand back yet.
            count = shape_var.raw
            if count is None:
                return None
            res += m * count.max()
        return int( res )

    def max_list( self ):
        return [ self.max ]

    def capacity_list( self, capacity_of ):
        res = self.offset
        for shape_var, m in self.coeffs.items():
            res += m * capacity_of( shape_var )
        return [ int( res ) ]
