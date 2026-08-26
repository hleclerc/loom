"""Affine expressions over SYMBOLS -- the arithmetic the whole shape system is written in.

An `Affine` is `offset + sum( coeff * symbol )`, with integer coefficients. It is a VALUE: immutable,
comparable, hashable. Two of them are equal when they are the same expression, not when they happen
to evaluate to the same number -- which is what lets a window be compared without being resolved.

A **symbol** is an unknown integer, and there are two kinds. They are the same arithmetic and very
different meanings, so keeping both admissible in one type is the point:

* a `ShapeVar` -- a **count**: how many items there are. `max = nb_vertices`.
* a `Coord`    -- a **position** along another dimension. `max = num_i`, which is how a triangular
  structure (`j` running over `0..i`) says what it is, and how a ragged extent could be stated by
  FORMULA rather than by a materialized array of counts.

Only `ShapeVar` symbols are produced today. `Coord` exists so that admitting the other kind later
costs no change of representation -- the one thing that is expensive to retrofit.

Inversion (`solve`) is what turns an OBSERVED size back into a count, and it is the operation that
constrains the design: it is single-variable, and it is meaningless over a `Coord` (one does not
"solve" a position). Both cases return `None` rather than guessing.
"""


class Coord:
    """A POSITION along a dimension, as an affine symbol -- what a bound depends on when a window is
    triangular or ragged-by-formula. Identified by its `AxisId`, so two `Coord`s over the same
    dimension are the same symbol.

    Not produced yet (see the module docstring); it is here so the expression type does not have to
    change when it is."""

    __slots__ = ( "axis_id", )

    def __init__( self, axis_id ) -> None:
        self.axis_id = axis_id

    def __eq__( self, other ):
        return isinstance( other, Coord ) and other.axis_id is self.axis_id

    def __hash__( self ):
        return hash( ( "Coord", id( self.axis_id ) ) )

    @property
    def name( self ):
        return getattr( self.axis_id, "name", None )

    def __repr__( self ) -> str:
        return f"Coord( { self.name } )"


class Affine:
    """`offset + sum( coeff * symbol )`. Immutable; every operation returns a new one."""

    __slots__ = ( "coeffs", "offset" )

    def __init__( self, coeffs = None, offset = 0 ) -> None:
        # zero coefficients are dropped, so equality is a property of the EXPRESSION and not of how
        # it was built: `n - n + 3` and `3` are the same affine.
        self.coeffs = { s: c for s, c in ( coeffs or {} ).items() if c }
        self.offset = int( offset )

    # ---- building ----
    @staticmethod
    def constant( value ) -> "Affine":
        return Affine( {}, value )

    @staticmethod
    def of( symbol ) -> "Affine":
        """The expression that IS `symbol` (`1 * symbol + 0`)."""
        return Affine( { symbol: 1 }, 0 )

    @staticmethod
    def factory( value ) -> "Affine":
        """`value` as an affine: an `Affine` (as is), an int (a constant), or a symbol."""
        if isinstance( value, Affine ):
            return value
        if isinstance( value, int ):
            return Affine.constant( value )
        return Affine.of( value )

    @staticmethod
    def parse( expr, resolve ) -> "Affine":
        """Parse `"2 * nb_dims + 1"` into an affine, each NAME turned into a symbol by `resolve`.

        Spaces are dropped and subtraction becomes the addition of a negative term. Parsing and
        resolution are split on purpose: the same text is read whether or not there is a scope to
        resolve names in (see `AbstractAxis.parse_affine`)."""
        names, offset = parse_terms( expr )
        res = Affine.constant( offset )
        for name, coeff in names.items():
            res = res + Affine.of( resolve( name ) ) * coeff
        return res

    # ---- arithmetic ----
    def __add__( self, other ) -> "Affine":
        other = Affine.factory( other )
        coeffs = dict( self.coeffs )
        for symbol, coeff in other.coeffs.items():
            coeffs[ symbol ] = coeffs.get( symbol, 0 ) + coeff
        return Affine( coeffs, self.offset + other.offset )

    __radd__ = __add__

    def __neg__( self ) -> "Affine":
        return Affine( { s: -c for s, c in self.coeffs.items() }, -self.offset )

    def __sub__( self, other ) -> "Affine":
        return self + ( -Affine.factory( other ) )

    def __rsub__( self, other ) -> "Affine":
        return Affine.factory( other ) + ( -self )

    def __mul__( self, k: int ) -> "Affine":
        return Affine( { s: c * int( k ) for s, c in self.coeffs.items() }, self.offset * int( k ) )

    __rmul__ = __mul__

    # ---- reading ----
    @property
    def symbols( self ):
        return list( self.coeffs.keys() )

    @property
    def is_constant( self ) -> bool:
        return not self.coeffs

    def value( self, of ):
        """Evaluate, `of( symbol )` giving each symbol's value -- an int, or an ARRAY for a count
        that varies (a ragged `ShapeVar` holds one per segment, and the arithmetic follows it).
        `None` as soon as one symbol has no value yet: an expression is known or it is not."""
        res = self.offset
        for symbol, coeff in self.coeffs.items():
            v = of( symbol )
            if v is None:
                return None
            res = res + coeff * v
        return res

    def solve( self, symbol, result ):
        """Invert `result = coeff * symbol + offset` for `symbol`, i.e. `( result - offset ) //
        coeff`, following the shape of `result` (a scalar, or one per segment).

        `None` when it cannot be done, and the two cases are worth telling apart:
        * `symbol` is not our SOLE variable -- a multi-variable solve is not attempted;
        * ... which includes an expression mentioning a `Coord`: a position is not something one
          solves for, so an expression that depends on where you are cannot be inverted at all."""
        if list( self.coeffs.keys() ) != [ symbol ]:
            return None
        return ( result - self.offset ) // self.coeffs[ symbol ]

    # ---- identity as a VALUE ----
    def __eq__( self, other ):
        if not isinstance( other, Affine ):
            if isinstance( other, int ):
                return self.is_constant and self.offset == other
            return NotImplemented
        return self.offset == other.offset and self.coeffs == other.coeffs

    def __hash__( self ):
        return hash( ( self.offset, frozenset( ( id( s ), c ) for s, c in self.coeffs.items() ) ) )

    def __repr__( self ) -> str:
        terms = []
        for symbol, coeff in self.coeffs.items():
            name = getattr( symbol, "name", None ) or "?"
            terms.append( name if coeff == 1 else f"{ coeff } * { name }" )
        if self.offset or not terms:
            terms.append( str( self.offset ) )
        return " + ".join( terms ).replace( "+ -", "- " )


def parse_terms( expr ):
    """Pure text parse of `"2 * nb_dims + 3 * nb_xs + 1"` into `( { name: coeff }, offset )`. No
    resolution -- names stay strings, so this is usable with no scope at all."""
    coeffs = {}
    offset = 0
    expr = str( expr ).replace( " ", "" ).replace( "-", "+-" )
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
