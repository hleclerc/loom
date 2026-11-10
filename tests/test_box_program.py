"""`loom.BoxProgram` ( `box_programs.py` ): a program per box, rows travelling between boxes in rounds. The travellers of
`box_program_cases.py` on one rank ( the boxes in turn, any framework ) and on MPI ranks ( numpy ). The Jax devices and the Torch
processes have files of their own ( `test_box_program_jax.py`, `box_program_torch_worker.py` )."""
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from errand import test


def _host( x ):
    import numpy as np
    import loom
    return np.asarray( loom.gather_to_host( x ) )


if test( "BoxProgram on one rank: the boxes in turn give what a numpy loop gives" ):
    import loom
    from box_program_cases import run_and_check
    from loom.tensor.Dtype import Dtype
    tol = 1e-12 if Dtype.fp().size != 32 else 1e-4

    def call( prog, values, hops, strides, weights ):
        out = prog( loom.array( values ), loom.array( hops ), loom.array( strides ), shared = ( loom.array( weights ), ) )
        return _host( out[ "res" ] ), _host( out[ "visits" ] )

    for K, L in [ ( 1, 4 ), ( 3, 5 ), ( 8, 16 ) ]:
        run_and_check( {}, call, K = K, L = L, tol = tol )
    # small capacities: the overflows of the rows and of the adds, the runs again, and what they taught kept
    prog = run_and_check( { "send_capacity": 1, "add_capacity": 1 }, call, tol = tol )
    assert prog.send_capacity.value > 1 and prog.add_capacity.value > 1, ( prog.send_capacity.value, prog.add_capacity.value )

if test( "BoxProgram adjoint on one rank: the rounds replayed in reverse give the exact gradient ( Jax, Torch )" ):
    import numpy as np
    import loom
    from box_program_cases import case, check_grads, program
    from loom.tensor.Dtype import Dtype
    from errand import has_tag
    fw = next( ( n for n in ( "jax", "torch" ) if has_tag( f"driver={ n }" ) ), None )
    if fw is None:
        print( "numpy: forward only" )
    else:
        if fw == "jax":
            import jax
            jax.config.update( "jax_enable_x64", True )
        old_fw, loom.default_framework = loom.default_framework, fw
        o = loom.ops( None )
        tol = 1e-12
        for K, L, caps in [ ( 1, 4, {} ), ( 3, 5, {} ), ( 8, 16, {} ), ( 8, 16, { "send_capacity": 1, "add_capacity": 1 } ) ]:
            values, hops, strides, weights = case( K, L )
            C = np.random.default_rng( 1 ).random( ( K, L ) )
            prog = program( K, L, **caps )
            prog.tape_rounds.value = 2            # ( an overflow of the tape: run again with more rounds )
            arr = lambda a: o.array( a, Dtype.fp( 64 ) )

            def loss( v, w ):
                out = prog( v, arr( hops ), arr( strides ), shared = ( w, ) )
                return o.sum( out[ "res" ].value * arr( C ) )

            gv, gw = o.grad( loss, argnums = ( 0, 1 ) )( arr( values ), arr( weights ) )
            check_grads( _host( gv ), _host( gw ), values, hops, strides, weights, C, tol = tol )
            assert prog.tape_rounds.value >= prog.last_rounds, ( prog.tape_rounds.value, prog.last_rounds )
        loom.default_framework = old_fw

if test( "BoxProgram out of core: boxes read from a store when they have mail, at most `max_resident` kept, what a numpy loop gives" ):
    import numpy as np
    import loom
    from box_program_cases import case, run_and_check
    from loom.tensor.Dtype import Dtype
    tol = 1e-12 if Dtype.fp().size != 32 else 1e-4
    reads = []

    def call( prog, values, hops, strides, weights ):
        # one store for the three blocks of a box, counting what it reads
        store = loom.BoxStore( lambda b: ( reads.append( b ), values[ b ] )[ 1 ], len( values ) )
        out = prog( store, loom.BoxStore.of( hops ), strides, shared = ( weights, ), max_resident = 2 )
        return _host( out[ "res" ] ), _host( out[ "visits" ] )

    for K, L in [ ( 1, 4 ), ( 8, 16 ) ]:
        reads.clear()
        prog = run_and_check( {}, call, K = K, L = L, tol = tol )
        # every box read in the first round, then again only when it has mail and was not kept
        assert len( reads ) >= K and prog.last_loads == len( reads ), ( len( reads ), prog.last_loads )
        assert len( reads ) <= K * prog.last_rounds

if test( "BoxProgram on MPI ranks ( numpy ): spread blocks, or whole arrays and the ranks found from the launcher", ranks = 4 ):
    import importlib.util
    from errand import skip
    if importlib.util.find_spec( "mpi4py" ) is None:
        skip( "needs mpi4py", hint = "micromamba install -c conda-forge mpi4py mpich" )
    import numpy as np
    from mpi4py import MPI
    import loom
    from box_program_cases import run_and_check
    loom.default_framework = "numpy"
    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    assert size > 1, "one process: not started by the launcher"
    assert isinstance( loom.detect_ranks(), loom.MpiRanks )
    ranks = loom.MpiRanks()

    def spread_call( prog, values, hops, strides, weights ):
        B = len( values ) // size
        mine = lambda a: loom.spread( a[ rank * B:( rank + 1 ) * B ].copy(), ranks )
        out = prog( mine( values ), mine( hops ), mine( strides ), shared = ( weights, ) )
        assert out[ "res" ].value.shape == ( B, values.shape[ 1 ] )         # ( this rank's boxes, in place )
        return _host( out[ "res" ] ), _host( out[ "visits" ] )

    def whole_call( prog, values, hops, strides, weights ):
        out = prog( values, hops, strides, shared = ( weights, ) )      # ( nothing spread: the ranks of the launcher )
        assert loom.spread_of( out[ "res" ] ).ranks.size == size
        return _host( out[ "res" ] ), _host( out[ "visits" ] )

    for call in ( spread_call, whole_call ):
        run_and_check( {}, call, K = 2 * size, L = 8 )
        run_and_check( {}, call, K = size, L = 16, seed = 3 )
        prog = run_and_check( { "send_capacity": 1, "add_capacity": 1 }, call, K = 2 * size, L = 8 )
        assert prog.send_capacity.value > 1
