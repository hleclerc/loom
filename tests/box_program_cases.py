"""A small program per box for the tests of `loom.BoxProgram`: TRAVELLERS hopping between boxes.

Box `b` starts `L` travellers; traveller `( b, i )` makes `hops[ b, i ]` hops of stride `strides[ b, i ]` ( box `b + s`, then `b + 2 s`,
modulo `K` ), adding the weight of each box it arrives at to its load, which starts at `values[ b, i ]`. Back home -- an add to its
owner, wherever it ends -- the load is `res[ b, i ]`; every arrival adds 1 to `visits[ box, 0 ]`. What a numpy loop gives too
( `expected` ), with rows that travel a variable number of rounds, some boxes more visited than others, and nothing left at the end."""
import numpy as np


def case( K = 8, L = 16, seed = 0, max_hops = 5 ):
    rng = np.random.default_rng( seed )
    values = rng.random( ( K, L ) )
    hops = rng.integers( 0, max_hops + 1, ( K, L ) ).astype( np.float64 )
    strides = rng.integers( 1, K, ( K, L ) ).astype( np.float64 ) if K > 1 else np.ones( ( K, L ) )
    weights = rng.random( K )
    return values, hops, strides, weights


def expected( values, hops, strides, weights ):
    K, L = values.shape
    res, visits = values.copy(), np.zeros( ( K, 1 ) )
    for b in range( K ):
        for i in range( L ):
            for k in range( 1, int( hops[ b, i ] ) + 1 ):
                c = int( ( b + k * strides[ b, i ] ) % K )
                res[ b, i ] += weights[ c ]
                visits[ c, 0 ] += 1
    return res, visits, int( hops.max() ) + 1


def program( K, L, **capacities ):
    """the `BoxProgram`: rows `[ owner box, owner index, hops left, stride, load ]`"""
    import loom
    from loom.tensor.Dtype import Dtype
    i32 = Dtype.si( 32 )

    def start( box, values, hops, strides, weights ):
        o = loom.ops( values )
        fp = o.dtype_of( values )
        L = int( values.shape[ 0 ] )
        idx = o.arange( L, i32 )
        here = idx * 0 + box.index                                        # [ L ], this box
        none = idx * 0 - 1
        rows = o.stack( [ o.astype( x, fp ) for x in ( here, idx, hops, strides, values ) ], axis = 1 )
        going = hops > 0
        dest = o.where( going, ( here + o.astype( strides, i32 ) ) % K, none )
        return loom.Outbox( send = ( dest, rows ), add = { "res": ( o.where( going, none, here ), idx, values ) } )

    def step( box, inbox, values, hops, strides, weights ):
        o = loom.ops( values )
        fp = o.dtype_of( values )
        r, ok = inbox.rows, inbox.valid
        here = o.astype( r[ :, 0 ], i32 ) * 0 + box.index
        none = here * 0 - 1
        load = r[ :, 4 ] + o.astype( weights, fp )[ box.index ]
        left = r[ :, 2 ] - 1
        go, home = ok & ( left > 0 ), ok & ( left <= 0 )
        dest = o.where( go, ( here + o.astype( r[ :, 3 ], i32 ) ) % K, none )
        rows = o.stack( [ r[ :, 0 ], r[ :, 1 ], left, r[ :, 3 ], load ], axis = 1 )
        return loom.Outbox( send = ( dest, rows ), add = {
            "res": ( o.where( home, o.astype( r[ :, 0 ], i32 ), none ), o.astype( r[ :, 1 ], i32 ), load ),
            "visits": ( o.where( ok, here, none ), here * 0, load * 0 + 1 ),
        } )

    return loom.BoxProgram( start, step, row_width = 5, outputs = { "res": L, "visits": 1 }, **capacities )


def run_and_check( prog_args, call, *, tol = 1e-12, K = 8, L = 16, **case_args ):
    """`call( prog, values, hops, strides, weights )` -> the outputs as host arrays; compared to `expected`"""
    values, hops, strides, weights = case( K, L, **case_args )
    prog = program( K, L, **prog_args )
    res, visits = call( prog, values, hops, strides, weights )
    ref_res, ref_visits, rounds = expected( values, hops, strides, weights )
    assert res.shape == ( K, L ) and visits.shape == ( K, 1 ), ( res.shape, visits.shape )
    assert np.abs( res - ref_res ).max() < tol, np.abs( res - ref_res ).max()
    assert np.abs( visits - ref_visits ).max() == 0, ( visits.ravel(), ref_visits.ravel() )
    assert prog.last_rounds == rounds, ( prog.last_rounds, rounds )
    return prog


def expected_grads( values, hops, strides, weights, C ):
    """the gradients of `sum( C * res )`: wrt `values` ( `C` ), `weights` ( `C` of each traveller, at each box it arrives at ), and
    0 wrt `hops` and `strides` ( which only say where the rows go )"""
    K, L = values.shape
    gw = np.zeros( K )
    for b in range( K ):
        for i in range( L ):
            for k in range( 1, int( hops[ b, i ] ) + 1 ):
                gw[ int( ( b + k * strides[ b, i ] ) % K ) ] += C[ b, i ]
    return C.copy(), gw


def check_grads( got_values, got_weights, values, hops, strides, weights, C, tol = 1e-12 ):
    ref_v, ref_w = expected_grads( values, hops, strides, weights, C )
    assert np.abs( got_values - ref_v ).max() < tol, np.abs( got_values - ref_v ).max()
    assert np.abs( got_weights - ref_w ).max() < tol, ( got_weights, ref_w )
