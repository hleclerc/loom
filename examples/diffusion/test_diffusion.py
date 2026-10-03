"""The tests of the outside user. Nothing imports sdot; everything goes through `loom`.

    errand test_diffusion
"""
import math
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import loom
from errand import test
from loom.testing import need
from loom.testing import check_grad

from diffusion import evolve, step


# THE STARTING FIELDS ARE BUILT ON THE DEVICE. A C++ expression of the coordinates, compiled
# once: no Python `n * n` loop, no host -> device transfer for data that
# the device knows how to build. The convention fits in two names -- `i_<axis>` the index along that
# axis, `n_<axis>` its extent -- and an anonymous axis is called `axis_0`, `axis_1`, so all of
# this is written without declaring a single axis.
_BORDER = "i_axis_0 == 0 || i_axis_1 == 0 || i_axis_0 + 1 == n_axis_0 || i_axis_1 + 1 == n_axis_1"


def _eigenmode( n ):
    """`sin( pi x ) sin( pi y )`: an EXACT eigenvector of the five-point stencil, zero on the
    boundary. Its decay factor per step is computed by hand."""
    return loom.RealTensor[ n, n ].expr(
        "sin( M_PI * i_axis_1 / ( n_axis_1 - 1 ) ) * sin( M_PI * i_axis_0 / ( n_axis_0 - 1 ) )" ).raw


def _bump( n, x0 = 0.35, y0 = 0.4, s = 0.15 ):
    return loom.RealTensor[ n, n ].expr(
        f"{ _BORDER } ? TF( 0 ) : exp( - ( ( TF( i_axis_1 ) / ( n_axis_1 - 1 ) - x0 )"
        " * ( TF( i_axis_1 ) / ( n_axis_1 - 1 ) - x0 )"
        " + ( TF( i_axis_0 ) / ( n_axis_0 - 1 ) - y0 )"
        " * ( TF( i_axis_0 ) / ( n_axis_0 - 1 ) - y0 ) ) / ( 2 * s * s ) )",
        x0 = x0, y0 = y0, s = s ).raw


if test( "the_eigenmode_decays_by_the_exact_factor" ):
    # `sin( pi x ) sin( pi y )` is an eigenvector of the stencil: one step must multiply it by
    #   1 + coef * ( 4 cos( pi h ) - 4 ),   h = 1 / ( n - 1 )
    # to machine precision. It is the stencil itself that is tested, not a trend.
    n = 17
    coef = 0.2                                     # dt / h^2, under the stability limit ( 0.25 )
    u = _eigenmode( n )

    expected = 1 + coef * ( 4 * math.cos( math.pi / ( n - 1 ) ) - 4 )
    v = step( u, coef )

    for j in range( 1, n - 1 ):
        for i in range( 1, n - 1 ):
            a, b = float( v[ j ][ i ] ), expected * float( u[ j ][ i ] )
            assert abs( a - b ) <= 1e-12 + 1e-10 * abs( b ), ( j, i, a, b )

    print( f"eigenmode: factor {expected:.6f} recovered on {(n-2)**2} cells" )


if test( "the_border_stays_fixed" ):
    # the boundary cells carry an imposed temperature: a step must not touch them, whatever
    # the interior does.
    n = 12
    u = loom.RealTensor[ n, n ].random( seed = 3 ).raw
    v = step( u, 0.15 )

    for j in range( n ):
        for i in range( n ):
            if j in ( 0, n - 1 ) or i in ( 0, n - 1 ):
                assert float( v[ j ][ i ] ) == float( u[ j ][ i ] ), ( j, i )
    assert any( float( v[ j ][ i ] ) != float( u[ j ][ i ] ) for j in range( 1, n - 1 ) for i in range( 1, n - 1 ) )


if test( "the_adjoint_is_the_one_of_the_solver" ):
    need( "grad" )
    # THE test that matters: the adjoint, checked against the finite difference, across A CHAIN of steps
    # ( it must travel back up it ).
    n = 9
    coef = 0.18
    u0 = _bump( n )

    ad, df = check_grad( lambda u: evolve( u, coef, 3 ), u0, seed = 11 )
    print( f"d/du  : adjoint {float( ad ):+.9f}   finite diff. {float( df ):+.9f}" )


if test( "we_go_back_in_time" ):
    need( "grad" )
    # WHAT we made the solver differentiable FOR: an inversion. We observe the temperature after
    # `nb_steps` diffusion steps, and we recover the INITIAL state by gradient descent through
    # the whole chain -- all compiled once ( `loom.driver.jit` ).
    n, nb_steps, coef = 14, 6, 0.2

    true_state = _bump( n )
    observed = evolve( true_state, coef, nb_steps )

    def loss( u ):
        diff = evolve( u, coef, nb_steps ) - observed
        return ( diff * diff ).sum()

    u = loom.RealTensor[ n, n ].zeros().raw
    loss_jit = loom.driver.jit( loss )
    gradient = loom.driver.jit( loom.driver.grad( loss ) )

    # the step size: `evolve` is CONTRACTING ( diffusion only smooths ), so the singular
    # values of its Jacobian are <= 1 and the Hessian of the loss has its eigenvalues
    # <= 2. A step beyond ~0.5 diverges -- that is what makes the inversion slow, not a
    # setting to be found by trial and error.
    loss_start = float( loss_jit( u ) )
    for _ in range( 400 ):
        u = u - 0.4 * gradient( u )
    loss_end = float( loss_jit( u ) )

    # it is the LOSS we assert, not `u`: going back in time is ill-posed ( diffusion erases
    # the high frequencies, which nothing can restore ), so without regularization we recover a
    # state that explains the data, not the true state. What is tested here is that the gradient
    # does go through the six calls.
    err_u = float( ( ( u - true_state ) ** 2 ).sum() ) ** 0.5
    print( f"loss {loss_start:.3e} -> {loss_end:.3e}   ( x{loss_start / max( loss_end, 1e-30 ):.0f} ),"
           f"   || u - u_true || = {err_u:.3f}" )
    assert loss_end < loss_start / 20


if test( "the_same_body_batches_without_knowing_it" ):
    need( "vmap" )
    need( "jax" )       # the test below maps with `jax.vmap` itself
    # WHAT THE AXES BUY. The stencil is written once, without counting dimensions:
    # `coords.axes - batch_axes` gives it its OWN axes, `for_each` unrolls them, and
    # `coords + axis` only shifts the named axis. A `vmap` therefore adds an axis WITHOUT the body
    # changing -- and without it knowing that it exists.
    import jax
    import numpy

    n, nb = 8, 3
    rng = numpy.random.default_rng( 0 )
    u = rng.normal( size = ( nb, n, n ) )

    # the reference: one call per batch item, by hand
    ref = numpy.stack( [ numpy.asarray( step( loom.driver.array( u[ b ] ), 0.1 ) ) for b in range( nb ) ] )

    # the same, vmapped over the first axis of `u`
    batched = jax.vmap( lambda uu: step( uu, 0.1 ), in_axes = 0 )
    got = numpy.asarray( batched( loom.driver.array( u ) ) )

    assert got.shape == ref.shape, ( got.shape, ref.shape )
    diff = float( numpy.abs( ref - got ).max() )
    assert diff == 0.0, diff
    print( f"vmap : { nb } batches, error exactly { diff }" )
