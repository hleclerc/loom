"""`loom.jit` under Jax WATCHES the capacities of its calls: an overflow leaves the compiled program as an output ( `checkify` ),
and the function runs again, whole, with that call site's capacity grown -- and kept for the next calls. No run-time callback,
and through the user's own `lax.while_loop`, `vmap` and `grad` too."""
import loom
from loom import ShapeVar, CtShapeVar, Axis, Aggregate, RealTensor, IntTensor
from loom.compilation.FfiCode import FfiCode
from loom.testing import need
from errand import test


class CellW( Aggregate ):
    vertex_positions : RealTensor[ "num_vertex", "dim" ]

    num_vertex       : Axis[ "nb_vertices" ]
    dim              : Axis[ "nb_dims" ]

    nb_vertices      : ShapeVar   # written by the kernel: what it produced
    nb_dims          : CtShapeVar


KERNEL = FfiCode.per_item( code = """
auto c = outputs.cell( batch_index );
c.nb_vertices.set( SI( inputs.wanted( batch_index ) ) );
for ( SI n = 0; n < SI( c.nb_vertices ); ++n )
    c.vertex_positions( num_vertex = n, dim = 0 ) = n;
""" )


def cell_of( nb_wanted, capacity, name = "test_jit_watch" ):
    cell = CellW( nb_dims = 2 )
    wanted = IntTensor()       # how many the kernel is going to produce: a traced value under a `jit`
    wanted.set( nb_wanted )
    loom.ffi_call( name, KERNEL, wanted = wanted, cell = loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ), capacities = { "nb_vertices": capacity } ) )
    return cell


def total( nb_wanted, capacity, name = "test_jit_watch" ):
    """the sum of the positions written: 0 + 1 + ... + nb_wanted - 1, if nothing was truncated"""
    import jax.numpy as jnp
    cell = cell_of( nb_wanted, capacity, name )
    # under a trace, `.value` cannot slice at a count it does not know: the buffer, masked past the count the kernel wrote
    pos = cell.vertex_positions.raw[ :, 0 ]
    return jnp.where( jnp.arange( pos.shape[ 0 ] ) < jnp.max( cell.nb_vertices.raw ), pos, 0 ).sum()


if test( "overflow_reruns_under_jit" ):
    need( "jax" )
    import jax.numpy as jnp
    f = loom.jit( lambda w: total( w, 2 ) )
    assert float( f( jnp.int32( 2 ) ) ) == 1
    assert f.watch.learned == {}
    # 5 into 2: the program runs again with `max( 5, 2 * 2 ) = 5`, and the result is whole
    assert float( f( jnp.int32( 5 ) ) ) == 10
    assert f.watch.learned == { ( "test_jit_watch", 0, "cell.nb_vertices" ): 5 }
    # kept: a smaller count does not trace again
    version = f.watch.version
    assert float( f( jnp.int32( 4 ) ) ) == 6
    assert f.watch.version == version

if test( "two_sites_of_one_kernel_learn_apart" ):
    need( "jax" )
    import jax.numpy as jnp
    f = loom.jit( lambda a, b: total( a, 2 ) + 100 * total( b, 2 ) )
    assert float( f( jnp.int32( 2 ), jnp.int32( 7 ) ) ) == 1 + 100 * 21
    assert f.watch.learned == { ( "test_jit_watch", 1, "cell.nb_vertices" ): 7 }

if test( "overflow_inside_a_while_loop" ):
    need( "jax" )
    import jax
    import jax.numpy as jnp

    def f( n ):
        # the count grows with the iterations: the capacity of 2 is exceeded on the 3rd one, inside the loop
        def body( c ):
            i, acc = c
            return i + 1, acc + total( i + 1, 2 )
        return jax.lax.while_loop( lambda c: c[ 0 ] < n, body, ( jnp.int32( 0 ), jnp.float64( 0 ) if jax.config.x64_enabled else jnp.float32( 0 ) ) )[ 1 ]

    g = loom.jit( f )
    # sum over i = 1..4 of i ( i - 1 ) / 2 = 0 + 1 + 3 + 6
    assert float( g( jnp.int32( 4 ) ) ) == 10
    assert g.watch.learned[ ( "test_jit_watch", 0, "cell.nb_vertices" ) ] >= 4

if test( "overflow_under_vmap" ):
    need( "jax" )
    import jax
    import jax.numpy as jnp
    g = loom.jit( jax.vmap( lambda w: total( w, 2 ) ) )
    assert [ float( v ) for v in g( jnp.array( [ 1, 3, 6 ], dtype = jnp.int32 ) ) ] == [ 0, 3, 15 ]

if test( "a_failure_raises_under_loom_jit" ):
    need( "jax" )
    need( "cpu" )
    import jax.numpy as jnp
    from loom.drivers.CallArg_Errors import KernelFailure

    def run( wanted ):
        res = RealTensor[ Axis( ShapeVar( 4 ), name = "num_fail" ) ]()
        loom.ffi_call(
            "test_jit_watch_failure",
            FfiCode.inline( "if ( SI( args.inputs.wanted ) ) args.errors.record( SI( args.inputs.wanted ), 7, 42 );\n" ),
            wanted = IntTensor( wanted ), res = loom.out( res ), failures = { 7: "seven: {value}" } )
        return res.value.sum()

    f = loom.jit( run )
    f( jnp.int32( 0 ) )
    try:
        f( jnp.int32( 2 ) )
        raise AssertionError( "no failure raised" )
    except KernelFailure as e:
        assert "seven: 42" in str( e ), str( e )

if test( "overflow_under_grad" ):
    need( "jax" )
    import jax
    import jax.numpy as jnp
    # the call sits in the primal of a gradient: its check goes through the differentiation too
    g = loom.jit( jax.grad( lambda x, w: x * total( w, 2 ) ) )
    assert float( g( jnp.float32( 3 ), jnp.int32( 6 ) ) ) == 15
    assert g.watch.learned == { ( "test_jit_watch", 0, "cell.nb_vertices" ): 6 }

if test( "under_an_outer_transform_the_callback_stays" ):
    need( "jax" )
    import jax
    import jax.numpy as jnp
    # `jax.grad` OF the `loom.jit` function: the error would be a tracer the host cannot read, so the plain `jax.jit` runs,
    # with the run-time callback -- the values are the same when nothing overflows
    f = loom.jit( lambda x, w: x * total( w, 8 ) )
    assert float( jax.grad( f )( jnp.float32( 3 ), jnp.int32( 6 ) ) ) == 15

if test( "a_differentiable_call_under_loom_jit" ):
    need( "jax" )
    import jax
    import jax.numpy as jnp
    # a call with a backward kernel ( a `custom_vjp` ): its backward is traced inside the watched program too
    forward = FfiCode.per_item( code = "outputs.out( batch_index ) = 2 * inputs.inp( batch_index ) + 100;" )
    backward = FfiCode.per_item( """
            if ( ! grad_of_outputs.out.surely_null && grad_of_inputs.inp.is_valid )
                grad_of_inputs.inp( batch_index ) = 2 * grad_of_outputs.out( batch_index );
        """, computed_grads = "computed_grads.inp_from_out = true;" )

    def fwd_of( x ):
        inp = RealTensor()
        inp.set( x )
        out = RealTensor()
        loom.ffi_call( "test_jit_watch_der", forward, backward, out = loom.out( out ), inp = inp )
        return out.raw

    g = loom.jit( jax.grad( lambda v: jax.vmap( fwd_of )( v ).sum() ) )
    assert [ float( x ) for x in g( jnp.array( [ 1.0, 2.0, 3.0 ] ) ) ] == [ 2, 2, 2 ]
