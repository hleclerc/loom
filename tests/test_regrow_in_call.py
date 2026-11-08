"""Under the pointer drivers ( numpy, torch, cupy ), a call GROWS ITS OUTPUTS ITSELF: when a capacity overflows, the handler asks
for larger outputs ( `regrow`, see the shim's `XLA_FFI_CallFrame` ) and runs the body again in the same call -- Python is only
asked to analyse the call again with the capacities grown ( `CallArgsAnalysis.grown` ), not to launch it again."""
import loom
from loom import ShapeVar, CtShapeVar, Axis, Aggregate, RealTensor
from loom.compilation.FfiCode import FfiCode
from errand import test


class CellR( Aggregate ):
    vertex_positions : RealTensor[ "num_vertex", "dim" ]

    num_vertex       : Axis[ "nb_vertices" ]
    dim              : Axis[ "nb_dims" ]

    nb_vertices      : ShapeVar
    nb_wanted        : ShapeVar
    nb_dims          : CtShapeVar


KERNEL = FfiCode.per_item( code = """
auto c = outputs.cell( batch_index );
c.nb_vertices.set( c.nb_wanted );
for ( SI n = 0; n < SI( c.nb_vertices ); ++n )
    c.vertex_positions( num_vertex = n, dim = 0 ) = n;
""" )


def cell_of( nb_wanted, capacity ):
    cell = CellR( nb_dims = 2, nb_wanted = nb_wanted )
    loom.ffi_call( "test_regrow_in_call", KERNEL,
                   cell = loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ), capacities = { "nb_vertices": capacity } ) )
    return cell


def counted( f ):
    """`f()`, with the launches from Python ( `PointerFfi.run` ) and the regrowths in the call ( `grown` ) counted"""
    from loom.drivers import PointerFfi
    from loom.drivers.CallArgsAnalysis import CallArgsAnalysis
    counts = dict( runs = 0, grown = 0 )
    run, grown = PointerFfi.run, CallArgsAnalysis.grown

    def counting_run( *a, **k ):
        counts[ "runs" ] += 1
        return run( *a, **k )

    def counting_grown( self ):
        res = grown( self )
        counts[ "grown" ] += res is not None
        return res

    PointerFfi.run, CallArgsAnalysis.grown = counting_run, counting_grown
    try:
        res = f()
    finally:
        PointerFfi.run, CallArgsAnalysis.grown = run, grown
    return res, counts


if test( "an_overflow_grows_in_the_call" ):
    from loom.drivers import framework_defaults
    if str( framework_defaults.framework() ) not in ( "numpy", "torch", "cupy" ):
        from errand import skip
        skip( "the outputs are XLA's under Jax: they cannot grow in the call" )

    # it fits: one launch, nothing grown
    cell, counts = counted( lambda: cell_of( 2, 8 ) )
    assert counts == dict( runs = 1, grown = 0 ), counts
    assert cell.nb_vertices.value == 2

    # 5 into 1: `max( 5, 2 * 1 ) = 5`, grown in the call -- still ONE launch from Python
    cell, counts = counted( lambda: cell_of( 5, 1 ) )
    assert counts == dict( runs = 1, grown = 1 ), counts
    assert cell.nb_vertices.value == 5
    assert cell.vertex_positions.capacity == ( 5, 2 )
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ] == [ 0, 1, 2, 3, 4 ]

    # 3 into 2: `max( 3, 4 ) = 4`
    cell, counts = counted( lambda: cell_of( 3, 2 ) )
    assert counts == dict( runs = 1, grown = 1 ), counts
    assert cell.vertex_positions.capacity == ( 4, 2 )
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ][ :3 ] == [ 0, 1, 2 ]


if test( "an_overflow_grows_in_a_batched_call" ):
    from loom.testing import need
    need( "vmap" )
    from loom.drivers import framework_defaults
    if str( framework_defaults.framework() ) not in ( "numpy", "torch", "cupy" ):
        from errand import skip
        skip( "the outputs are XLA's under Jax: they cannot grow in the call" )
    from loom import IntTensor

    class CellB( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]
        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]
        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

    kernel = FfiCode.per_item( code = """
    auto c = outputs.cell( batch_index );
    c.nb_vertices.set( SI( inputs.wanted( batch_index ) ) );
    for ( SI n = 0; n < SI( c.nb_vertices ); ++n )
        c.vertex_positions( num_vertex = n, dim = 0 ) = n;
    """ )

    def one( w ):
        cell = CellB( nb_dims = 2 )
        wanted = IntTensor()
        wanted.set( w )
        loom.ffi_call( "test_regrow_in_batched_call", kernel, wanted = wanted,
                       cell = loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ), capacities = { "nb_vertices": 2 } ) )
        return cell.vertex_positions.raw

    # one item asks for 6 in a batch of capacity 2: the batched call grows to `max( 6, 4 ) = 6`, in the call
    res, counts = counted( lambda: loom.vmap( one )( loom.array( [ 1, 6, 3 ], dtype = int ) ) )
    assert counts[ "grown" ] == 1 and counts[ "runs" ] == 1, counts
    rows = [ [ v[ 0 ] for v in item ] for item in res.tolist() ]
    assert [ r[ :w ] for r, w in zip( rows, ( 1, 6, 3 ) ) ] == [ [ 0 ], [ 0, 1, 2, 3, 4, 5 ], [ 0, 1, 2 ] ], rows
