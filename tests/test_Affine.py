from loom.tensor import Affine, Coord
from loom import ShapeVar
from loom.testing import test
import numpy


class FakeAxisId:
    """Stands in for the axis identity a `Coord` names -- all a `Coord` needs of it is to be a
    stable object with a name."""
    def __init__( self, name ):
        self.name = name


if test( "an_affine_is_a_value" ):
    # Two affines are equal when they are the SAME EXPRESSION, not when they happen to evaluate to
    # the same number -- which is what lets a window be compared before anything is resolved.
    n, m = ShapeVar(), ShapeVar()
    n.name, m.name = "nb_n", "nb_m"

    assert Affine.constant( 3 ) == Affine.constant( 3 )
    assert Affine.of( n ) == Affine.of( n )
    assert Affine.of( n ) != Affine.of( m )
    assert Affine.of( n ) + 1 != Affine.of( n )

    # a zero coefficient is not part of the expression, so how it was BUILT does not show
    assert Affine.of( n ) - Affine.of( n ) + 3 == Affine.constant( 3 )
    assert ( Affine.of( n ) * 0 ).is_constant

    # hashable, so an expression can key a lookup (a window coordinate is compared this way)
    assert len( { Affine.of( n ) + 1, Affine.of( n ) + 1, Affine.of( m ) } ) == 2

    # an int compares as the constant it is
    assert Affine.constant( 5 ) == 5 and Affine.of( n ) != 5


if test( "affine_arithmetic" ):
    n, m = ShapeVar(), ShapeVar()
    n.name, m.name = "nb_n", "nb_m"
    a = Affine.of( n )

    assert ( a + 1 ).offset == 1
    assert ( 1 + a ).offset == 1                  # radd
    assert ( a - 1 ).offset == -1
    assert ( 5 - a ) == Affine( { n: -1 }, 5 )    # rsub
    assert ( a * 3 ) == Affine( { n: 3 }, 0 )
    assert ( 3 * a ) == Affine( { n: 3 }, 0 )     # rmul
    assert ( -a ) == Affine( { n: -1 }, 0 )
    assert ( a + Affine.of( m ) * 2 - 1 ) == Affine( { n: 1, m: 2 }, -1 )

    # the operands are untouched: an affine is immutable
    assert a == Affine.of( n )


if test( "affine_evaluation" ):
    n, m = ShapeVar(), ShapeVar()
    n.name, m.name = "nb_n", "nb_m"
    expr = Affine.of( n ) * 2 + Affine.of( m ) + 1        # 2n + m + 1

    assert expr.value( { n: 3, m: 4 }.get ) == 11

    # unresolved anywhere -> unresolved as a whole: an expression is known, or it is not
    assert expr.value( { n: 3 }.get ) is None
    assert Affine.constant( 7 ).value( lambda s: None ) == 7

    # a RAGGED count holds one value per segment, and the arithmetic simply follows it
    per_segment = numpy.array( [ 1, 2, 3 ] )
    ragged = Affine.of( n ) + 1
    assert ragged.value( { n: per_segment }.get ).tolist() == [ 2, 3, 4 ]


if test( "affine_inversion" ):
    # Inverting is what turns an OBSERVED size back into a count -- the operation the whole shape
    # inference rests on, and the one that constrains what an expression may contain.
    n, m = ShapeVar(), ShapeVar()
    n.name, m.name = "nb_n", "nb_m"

    assert ( Affine.of( n ) * 2 + 1 ).solve( n, 7 ) == 3           # 2n + 1 = 7 -> n = 3
    assert ( Affine.of( n ) ).solve( n, 5 ) == 5
    assert ( Affine.of( n ) - 1 ).solve( n, numpy.array( [ 2, 4 ] ) ).tolist() == [ 3, 5 ]

    # not our sole variable: not attempted rather than guessed
    assert ( Affine.of( n ) + Affine.of( m ) ).solve( n, 7 ) is None
    assert ( Affine.of( m ) ).solve( n, 7 ) is None
    assert Affine.constant( 4 ).solve( n, 4 ) is None


if test( "a_position_is_not_something_one_solves_for" ):
    # An affine admits two kinds of symbol, and they are not interchangeable:
    #   * a ShapeVar -- a COUNT: how many items there are.
    #   * a Coord    -- a POSITION along another dimension. What a triangular window depends on
    #                   (`j` running over `0..i`), and what a ragged extent could be stated with.
    # Both are unknown integers and share the arithmetic. But one does not INVERT a position, so an
    # expression that mentions one cannot be solved -- and says so rather than inventing an answer.
    n = ShapeVar(); n.name = "nb_n"
    i = Coord( FakeAxisId( "num_i" ) )

    triangular = Affine.of( i )                    # max = num_i
    assert triangular.symbols == [ i ]
    assert triangular.value( { i: 4 }.get ) == 4   # it EVALUATES fine, wherever we are

    assert triangular.solve( n, 4 ) is None        # ... but it does not invert to a count
    assert ( Affine.of( n ) + Affine.of( i ) ).solve( n, 9 ) is None

    # two Coords over the SAME dimension are the same symbol; over another one, they are not
    same_dim = i.axis_id
    assert Affine.of( Coord( same_dim ) ) == triangular
    assert Affine.of( Coord( FakeAxisId( "num_i" ) ) ) != triangular   # same NAME, other dimension


if test( "affine_parsing" ):
    n, m = ShapeVar(), ShapeVar()
    n.name, m.name = "nb_dims", "nb_xs"
    resolve = { "nb_dims": n, "nb_xs": m }.get

    assert Affine.parse( "nb_dims + 1", resolve ) == Affine.of( n ) + 1
    assert Affine.parse( "2 * nb_dims + 3 * nb_xs + 1", resolve ) == Affine( { n: 2, m: 3 }, 1 )
    assert Affine.parse( "nb_dims - 1", resolve ) == Affine.of( n ) - 1
    assert Affine.parse( "  nb_dims  +  nb_dims ", resolve ) == Affine.of( n ) * 2
    assert Affine.parse( "4", resolve ) == 4

    # the TEXT parse is separable: usable with no scope to resolve names in at all
    from loom.tensor.Affine import parse_terms
    assert parse_terms( "2 * a + b - 3" ) == ( { "a": 2, "b": 1 }, -3 )


if test( "affine_reads_back" ):
    n = ShapeVar(); n.name = "nb_n"
    assert repr( Affine.of( n ) ) == "nb_n"
    assert repr( Affine.of( n ) + 1 ) == "nb_n + 1"
    assert repr( Affine.of( n ) * 2 - 1 ) == "2 * nb_n - 1"
    assert repr( Affine.constant( 3 ) ) == "3"
