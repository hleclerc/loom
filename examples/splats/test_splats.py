"""Tests of loom's second foreign user. Nothing here imports sdot.

    errand test_splats
"""
import math
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import numpy

from errand import Param, bench, test
from loom.testing import check_grad

from splats import TILE_SIDE, Splats, touching_gaussians, render, render_scene


def _scene( nb, shape, seed = 0, scale = 3.0, clusters = 6 ):
    """A reproducible scene, and above all a REALISTIC one on the single point that matters here:
    inequality.

    Two things produce it, and they are the ones a real scene has. The centers come in CLUSTERS (a
    reconstruction puts Gaussians where there is matter, not uniformly), and the sizes span a DECADE
    (fine detail and blurry background). Hence empty tiles and overloaded ones -- a worst case very
    far from the mean, which is exactly what a single bound has to pay for.

    `centers[ :, d ]` is the coordinate along axis `d` of the image: one convention, no `x` and no
    `y`. The covariances below are 2D, like `splats.h::half_box`.
    """
    rng = numpy.random.default_rng( seed )
    nb_dims = len( shape )

    # the centers: a few tight clusters, plus a scattered background
    nb_bg = max( 1, nb // 5 )
    foci = numpy.stack( [ rng.uniform( 0, n, clusters ) for n in shape ], axis = 1 )
    which = rng.integers( 0, clusters, nb - nb_bg )
    tight = foci[ which ] + rng.normal( 0, min( shape ) / 25, ( nb - nb_bg, nb_dims ) )
    background = numpy.stack( [ rng.uniform( 0, n, nb_bg ) for n in shape ], axis = 1 )
    centers = numpy.concatenate( [ tight, background ] )

    # the sizes: log-uniform over a decade
    sigmas = scale * numpy.exp( rng.uniform( 0, math.log( 10 ), nb ) )

    # an anisotropic covariance, inverted: ( a, b, c ) = inv( R S R^T ), packed upper triangle
    angles = rng.uniform( 0, math.pi, nb )
    ratio = rng.uniform( 0.3, 1.0, nb )
    s0, s1 = sigmas, sigmas * ratio
    ca, sa = numpy.cos( angles ), numpy.sin( angles )
    i00, i11 = 1.0 / s0 ** 2, 1.0 / s1 ** 2
    a = ca * ca * i00 + sa * sa * i11
    c = sa * sa * i00 + ca * ca * i11
    b = ca * sa * ( i00 - i11 )
    cov_inv = numpy.stack( [ a, b, c ], axis = 1 )

    # nothing to prescribe but the two compile-time counts; `num_coeff` COMPUTES itself from
    # `nb_dims` ( 2 * 3 / 2 = 3 )
    splats = Splats( nb_dims = nb_dims, nb_channels = 3 )
    splats.centers = centers
    splats.cov_inv = cov_inv
    splats.colors = rng.uniform( 0.1, 1.0, ( nb, 3 ) )
    splats.opacities = rng.uniform( 0.2, 1.0, nb )
    return splats


def _nb_tiles( shape ):
    return tuple( ( n + TILE_SIDE - 1 ) // TILE_SIDE for n in shape )


def _per_tile( touching, shape ):
    """How many Gaussians each tile holds, back in the shape of the tile GRID.

    `offsets` carries `nb_rows + 1` BOUNDS and not counts, so a row's length is a subtraction of
    neighbours -- and that is also why the last bound IS the total."""
    offsets = numpy.asarray( touching.offsets.value )
    return numpy.diff( offsets ).reshape( _nb_tiles( shape ) )


if test( "the_lists_are_sized_exactly" ):
    # THE test of the structure: the per-tile counts are written by the kernel, read back on the
    # HOST, and the flat list allocated at exactly their sum. Nothing is guessed and nothing is
    # padded -- which is the one thing a JIT cannot do.
    shape, nb = ( 256, 256 ), 2000
    splats = _scene( nb, shape )
    touching = touching_gaussians( splats, shape )

    counts = _per_tile( touching, shape )
    assert counts.shape == _nb_tiles( shape ), counts.shape
    assert counts.sum() > nb, "every splat touches at least one tile"

    # EXACTLY: the list is as long as its content, to the integer
    assert touching.total == int( counts.sum() )
    ids = numpy.asarray( touching.values )
    assert ids.shape == ( touching.total, ), ids.shape

    # the structure really is UNEVEN: that is what makes a single bound expensive
    print( f"tiles {counts.shape}, {nb} splats : per tile min {counts.min()}, "
           f"mean {counts.mean():.1f}, max {counts.max()}, total {counts.sum()}" )
    assert counts.min() > 0 and counts.max() >= 2 * counts.min()

    # and every row holds nothing but splats
    offsets = numpy.asarray( touching.offsets.value )
    flat = counts.reshape( -1 )
    for t in range( 0, flat.size, 7 ):
        row = ids[ offsets[ t ] : offsets[ t + 1 ] ]
        assert len( row ) == flat[ t ]
        assert ( ( 0 <= row ) & ( row < nb ) ).all()
    print( "lists consistent" )


if test( "the_rendering_matches_the_direct_sum" ):
    # the reference: the sum over EVERY splat, no lists at all. If one of them lost a splat the gap
    # would show -- which is what makes this a test of the structure, not of the stencil alone.
    from reference import dense_render

    shape, nb = ( 64, 96 ), 150
    splats = _scene( nb, shape )
    image = render( splats, touching_gaussians( splats, shape ), shape )

    expected = dense_render( numpy.asarray( splats.centers ), numpy.asarray( splats.cov_inv ),
                             numpy.asarray( splats.colors ), numpy.asarray( splats.opacities ),
                             shape )
    got = numpy.asarray( image )
    gap = numpy.abs( got - expected ).max()
    print( f"max gap to the dense rendering : {gap:.3e}   ( image {got.shape} )" )
    assert gap < 1e-10, gap


if test( "an_image_that_is_not_a_multiple_of_the_tile" ):
    # THE IMAGE SIZE IS NOT A COMPILE-TIME PARAMETER: nothing forces its sides onto the tile grid,
    # and the last row of tiles simply hangs over the edge.
    from reference import dense_render

    shape, nb = ( 37, 53 ), 80
    splats = _scene( nb, shape, seed = 5 )
    image = render( splats, touching_gaussians( splats, shape ), shape )

    expected = dense_render( numpy.asarray( splats.centers ), numpy.asarray( splats.cov_inv ),
                             numpy.asarray( splats.colors ), numpy.asarray( splats.opacities ),
                             shape )
    gap = numpy.abs( numpy.asarray( image ) - expected ).max()
    print( f"image {shape} ( tile {TILE_SIDE} ) : gap to the dense rendering {gap:.3e}" )
    assert gap < 1e-10, gap


if test( "the_atomic_adjoint_is_the_one_of_the_rendering" ):
    # the adjoint ACCUMULATES: a splat is hit by every pixel it covers, hence by different
    # work-items. The opposite of `examples/diffusion`, which was a pure gather.
    shape, nb = ( 48, 48 ), 30
    splats = _scene( nb, shape, seed = 3, scale = 5.0 )

    for name in ( "colors", "opacities", "centers", "cov_inv" ):
        def rend( value, name = name ):
            s = _scene( nb, shape, seed = 3, scale = 5.0 )
            setattr( s, name, value )
            return render_scene( s, shape )

        start = getattr( splats, name ).raw
        adjoint, finite = check_grad( rend, start, seed = 11 )
        print( f"d/d{name:<10} : adjoint {float( adjoint ):+.9f}   finite diff. {float( finite ):+.9f}" )


if test( "what_a_single_bound_costs" ):
    # WHAT XLA COSTS, in numbers, and without a straw man.
    #
    # XLA can build these lists -- provided the worst case is BOUNDED before tracing. The bound it
    # would have to choose is priced here against what loom spends: the radius of a FIXED WINDOW per
    # splat, dictated by the largest splat of the scene and paid by all. That is the expensive one,
    # because the sizes span a decade, hence the footprints two orders of magnitude.
    from reference import window_work, real_work

    for nb, shape in ( ( 500, ( 256, 256 ) ), ( 2000, ( 256, 256 ) ), ( 2000, ( 512, 512 ) ) ):
        splats = _scene( nb, shape )
        cov = numpy.asarray( splats.cov_inv )
        counts = _per_tile( touching_gaussians( splats, shape ), shape ).reshape( -1 )

        bound, radius = window_work( cov )
        useful = real_work( cov )
        dense = nb * int( numpy.prod( shape ) )

        print( f"\n{nb} splats, {shape} :" )
        print( f"  per tile       : mean {counts.mean():7.1f}   max {counts.max():5d}"
               f"   -> a single capacity would waste x{counts.max() / counts.mean():.1f}" )
        print( f"  useful pairs   : {useful:12d}   ( the sum of the real footprints )" )
        print( f"  fixed window   : {bound:12d}   ( R = {radius}, set by the largest splat )"
               f"   -> x{bound / useful:.1f}" )
        print( f"  dense sum      : {dense:12d}   ( every pixel sees every splat )"
               f"   -> x{dense / useful:.1f}" )

        # a fixed window's bound costs at least an order of magnitude: that is the argument
        assert bound > 5 * useful


if test( "a_padded_storage_would_cost_more_memory" ):
    # WHY THERE IS ONLY ONE STORAGE LEFT. The example used to carry both, and the padded rectangle
    # was measured at 2.4 to 5.1 times the memory of the CSR -- so it was dropped ( see the README ).
    #
    # The finding stays CHECKED, and it needs no second implementation to stay alive: what a padded
    # rectangle would cost is arithmetic on the row lengths. It is priced AT BEST, with the capacity
    # exactly equal to the longest row -- which a real padded version could not even know in advance.
    for nb, shape in ( ( 500, ( 256, 256 ) ), ( 2000, ( 256, 256 ) ), ( 2000, ( 512, 512 ) ) ):
        splats = _scene( nb, shape )
        touching = touching_gaussians( splats, shape )
        counts = _per_tile( touching, shape ).reshape( -1 )

        nb_tiles = counts.size
        padded = nb_tiles * int( counts.max() )             # the BEST possible capacity
        csr = touching.total + nb_tiles + 1                 # the list, plus the bounds

        print( f"\n{nb} splats, {shape} :" )
        print( f"  padded, best capacity ( {counts.max()} ) : {padded:8d} ints" )
        print( f"  CSR, exact size                   : {csr:8d} ints" )
        print( f"  -> even at best, a padded rectangle costs x{padded / csr:.2f} the memory" )

        assert padded > csr


if p := bench( "where_the_time_goes",
               nb     = Param( 2000, help = "number of splats" ),
               size   = Param( 512, help = "image side" ),
               reps   = Param( 5, help = "timed repetitions ( the minimum is kept )" ),
               seed   = Param( 0, help = "scene seed" ) ):
    # THE MEMORY IS MEASURED ABOVE; here it is TIME, and the question it settles is where it goes.
    # Building the lists costs two passes over the splats AND a round trip to the host ( reading the
    # total ); rendering costs one pass over the pixels. Which dominates is not a matter of opinion.
    import time

    def done( x ):
        """Really wait: jax is asynchronous, and a stopwatch around a call that has not finished
        only measures the time to enqueue it."""
        raw = x.raw if hasattr( x, "raw" ) else x
        if hasattr( raw, "block_until_ready" ):
            raw.block_until_ready()
        else:
            numpy.asarray( raw )
        return x

    shape = ( p.size, p.size )
    splats = _scene( p.nb, shape, seed = p.seed )

    # one warm-up per kernel: the first call of each is still compiling
    touching = touching_gaussians( splats, shape )
    done( touching.values )
    done( render( splats, touching, shape ) )

    def chrono( action ):
        best = float( "inf" )
        for _ in range( p.reps ):
            t = time.perf_counter()
            done( action() )
            best = min( best, time.perf_counter() - t )
        return best

    t_lists = chrono( lambda: touching_gaussians( splats, shape ).values )
    t_render = chrono( lambda: render( splats, touching, shape ) )

    counts = _per_tile( touching, shape ).reshape( -1 )
    print( f"\n{p.nb} splats, {shape}, {touching.total} ( splat, tile ) pairs"
           f" over {counts.size} tiles ( max {counts.max()} )" )
    print( f"  building the lists ( 2 passes + a host prefix ) : {t_lists * 1e3:8.2f} ms" )
    print( f"  rendering                                       : {t_render * 1e3:8.2f} ms" )
    print( f"  TOTAL : {( t_lists + t_render ) * 1e3:8.2f} ms"
           f"   -> the rendering is {100 * t_render / ( t_lists + t_render ):.0f} % of it" )

    p.results.update( lists_ms = t_lists * 1e3, render_ms = t_render * 1e3,
                      total_pairs = touching.total, max_per_tile = int( counts.max() ) )
