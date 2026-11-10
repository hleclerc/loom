"""`loom.BoxProgram` on Jax devices ( 4 fake CPU devices: the device count must be set before the first `import jax` of the process,
hence a file of its own ): one `shard_map` under `jit`, the rounds in a `lax.while_loop`."""
import os
import sys

os.environ[ "XLA_FLAGS" ] = os.environ.get( "XLA_FLAGS", "" ) + " --xla_force_host_platform_device_count=4"
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from errand import has_tag, test

if not has_tag( "driver=jax" ):
    sys.exit( 0 )

if test( "BoxProgram on Jax devices: boxes spread by loom or sharded by Jax, several boxes per device, what a numpy loop gives" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import numpy as np
    import loom
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from box_program_cases import run_and_check
    from loom.testing import need
    need( "cpu" )
    assert len( jax.devices() ) == 4, jax.devices()

    for nd in [ 2, 4 ]:
        mesh = Mesh( np.array( jax.devices()[ :nd ] ), ( "p", ) )
        ranks = loom.JaxRanks( mesh )
        for how in ( "spread", "sharded" ):
            def call( prog, values, hops, strides, weights ):
                if how == "spread":
                    args = [ loom.spread( a, ranks ) for a in ( values, hops, strides ) ]
                else:
                    args = [ jax.device_put( a, NamedSharding( mesh, P( "p", None ) ) ) for a in ( values, hops, strides ) ]
                out = prog( *args, shared = ( jax.numpy.asarray( weights ), ) )
                assert loom.spread_of( out[ "res" ] ).ranks == ranks
                return np.asarray( loom.gather_to_host( out[ "res" ] ) ), np.asarray( loom.gather_to_host( out[ "visits" ] ) )

            for K in [ nd, 2 * nd, 4 * nd ]:
                run_and_check( {}, call, K = K, L = 8 )
            prog = run_and_check( { "send_capacity": 1, "add_capacity": 1 }, call, K = 2 * nd, L = 8 )
            assert prog.send_capacity.value > 1 and prog.add_capacity.value > 1

if test( "BoxProgram adjoint on Jax devices: jax.grad through the rounds, wrt spread blocks and a whole shared value" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    import numpy as np
    import loom
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from box_program_cases import case, check_grads, program
    from loom.testing import need
    need( "cpu" )

    for nd in [ 2, 4 ]:
        mesh = Mesh( np.array( jax.devices()[ :nd ] ), ( "p", ) )
        ranks = loom.JaxRanks( mesh )
        for how in ( "spread", "sharded" ):
            for K, caps in [ ( nd, {} ), ( 2 * nd, { "send_capacity": 1, "add_capacity": 1 } ) ]:
                L = 8
                values, hops, strides, weights = case( K, L )
                C = np.random.default_rng( 1 ).random( ( K, L ) )
                prog = program( K, L, **caps )
                prog.tape_rounds.value = 2
                sh = lambda a: jax.device_put( jnp.asarray( a ), NamedSharding( mesh, P( "p", None ) ) )

                def loss( v, w ):
                    if how == "spread":       # ( traced, a sharded array says nothing: `loom.spread` or `ranks =` )
                        out = prog( loom.spread( v, ranks ), loom.spread( sh( hops ), ranks ), loom.spread( sh( strides ), ranks ), shared = ( w, ) )
                    else:
                        out = prog( v, sh( hops ), sh( strides ), shared = ( w, ), ranks = ranks )
                    return jnp.sum( out[ "res" ].value * C )

                gv, gw = jax.grad( loss, argnums = ( 0, 1 ) )( sh( values ), jnp.asarray( weights ) )
                check_grads( np.asarray( gv ), np.asarray( gw ), values, hops, strides, weights, C )
                assert prog.tape_rounds.value >= prog.last_rounds
