"""The references: what a competent user would write WITHOUT loom, and what it costs.

The example's argument would be dishonest against a straw man. XLA is not unable to do this -- it is
unable to do it WITHOUT BOUNDING THE WORST CASE AHEAD OF TIME. So the two references below are the
two correct shapes one actually writes, and what gets measured is the price of the bound, not a
clumsiness.

  `dense_render` : the sum over EVERY splat, for every pixel. Simple, exact, and the first thing
                   one writes. Costs `pixels x splats` -- it is the TRUTH the tests compare
                   against, and a scale marker.

  a fixed window : each splat writes only into an `R x R` patch around its center, by scatter-add.
                   That is the right XLA implementation, and it is fast. But `R` is an INTEGER OF
                   THE SHAPE, so it has to cover the LARGEST splat of the scene: everyone pays the
                   worst footprint. That is exactly the bound loom does without -- `window_work`
                   prices it.

What loom does instead: the kernel writes the per-tile count, and a capacity that did not hold is
reported and retried. The cost is the real sum of the footprints, not `n x worst^2`.
"""
import numpy

SIGMA_RADIUS = 3.0          # the same threshold as `splats.h`


def _weights( centers, cov_inv, opacities, shape ):
    """`opacity * ( exp( -q/2 ) - E )`, truncated past `SIGMA_RADIUS`, for every pixel of `shape`.

    Dimension-free, like the kernel: `shape` has one entry per axis, and axis `d` of the image is
    coordinate `d` of a center."""
    nb, nb_dims = centers.shape
    ones = ( 1, ) * len( shape )

    grid = numpy.indices( shape ).astype( float )                      # [ D, *shape ]
    delta = grid[ None ] - centers.reshape( nb, nb_dims, *ones )       # [ n, D, *shape ]

    # q = sum_ab A_ab dx_a dx_b, the upper triangle being stored once -- so off-diagonal twice
    q, k = 0.0, 0
    for a in range( nb_dims ):
        for b in range( a, nb_dims ):
            coeff = cov_inv[ :, k ].reshape( nb, *ones )
            q = q + ( 1 if a == b else 2 ) * coeff * delta[ :, a ] * delta[ :, b ]
            k += 1

    # the SAME continuous truncation as `splats.h`: we subtract the value at the threshold,
    # otherwise this test would be comparing two different models
    threshold = numpy.exp( -0.5 * SIGMA_RADIUS ** 2 )
    w = opacities.reshape( nb, *ones ) * ( numpy.exp( -0.5 * q ) - threshold )
    return numpy.where( q > SIGMA_RADIUS ** 2, 0.0, w )


def dense_render( centers, cov_inv, colors, opacities, shape ):
    """The truth: every pixel sums over every splat. `pixels x splats` operations, and an
    intermediate `[ n, *shape ]` array -- 500 MB for 2000 splats at 256x256."""
    w = _weights( centers, cov_inv, opacities, shape )      # [ n, *shape ]
    return numpy.tensordot( w, colors, axes = ( 0, 0 ) )    # [ *shape, nb_channels ]


def _half_extents( cov_inv ):
    """Per splat, the half-extent of its footprint along each axis. 2D only, exactly like
    `splats.h::half_box` -- and for the same reason: in nD this is the diagonal of an inverse."""
    a, b, c = cov_inv[ :, 0 ], cov_inv[ :, 1 ], cov_inv[ :, 2 ]
    det = a * c - b * b
    return SIGMA_RADIUS * numpy.sqrt( c / det ), SIGMA_RADIUS * numpy.sqrt( a / det )


def required_radius( cov_inv ):
    """The half-side `R` a fixed window must have to contain the LARGEST splat.

    That is the number the XLA version has to choose before tracing, and one single splat dictates
    it -- hence its cost: `n x ( 2R+1 )^2` for everybody."""
    half_0, half_1 = _half_extents( cov_inv )
    return int( numpy.ceil( max( half_0.max(), half_1.max() ) ) )


def window_work( cov_inv ):
    """How many ( splat, pixel ) pairs the fixed-window version goes through."""
    r = required_radius( cov_inv )
    return cov_inv.shape[ 0 ] * ( 2 * r + 1 ) ** 2, r


def real_work( cov_inv ):
    """How many ( splat, pixel ) pairs are actually useful: the sum of the real footprints."""
    half_0, half_1 = _half_extents( cov_inv )
    return int( numpy.sum( ( 2 * half_0 + 1 ) * ( 2 * half_1 + 1 ) ) )
