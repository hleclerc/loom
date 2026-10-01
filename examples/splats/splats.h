#pragma once

// THE C++ HALF OF THE TUTORIAL. `splats.py` tells the story; this file is the four kernels it
// launches, in the order it launches them. Read it top to bottom.
//
// Nothing here knows about Python, and nothing here knows how many dimensions it is working in:
// every loop below runs over `d`, and no line names an `x` or a `y`. The one exception is
// `half_box`, which says so.

#include <loom/support/algorithms/box_indices.h>      // for_each_in_box, to_array, to_tuple
#include <loom/support/containers/Coords.h>           // for_each_indexed
#include <loom/support/common_macros.h>               // HD
#include <loom/support/common_types.h>                // SI
#include <loom/support/atomic_add.h>

#include <type_traits>
#include <cmath>

namespace splats {

using namespace sdot;

/// the scalar a view holds, without its constness
template<class V>
using Scalar = std::remove_const_t<typename std::remove_reference_t<V>::TF>;

/// Past `SIGMA_RADIUS` standard deviations a splat no longer contributes. That is what gives it a
/// BOUNDED footprint, hence a box of tiles -- without which there would be nothing to index.
static constexpr double SIGMA_RADIUS = 3.0;

/// The truncation is CONTINUOUS: we subtract the value AT the threshold rather than cut.
///
///     w = opacity * max( exp( -q/2 ) - exp( -k²/2 ), 0 )
///
/// Cutting would make the weight jump from `opacity * 1.1e-2` to zero across the boundary. And the
/// boundary MOVES when a center moves or a covariance changes: the derivative then acquires
/// impulses, a finite difference measures a jump divided by 2 eps, and a gradient descent takes
/// noise at every step. Measured: with the hard cut, the adjoint was exact to eleven digits on
/// `colors` and `opacities` -- which do not enter `q` -- and wrong by 5 % on `centers`. With this
/// one, all four are exact.
template<class TF>
HD TF threshold() {
    return TF( std::exp( -0.5 * SIGMA_RADIUS * SIGMA_RADIUS ) );
}

// ── THE MODEL, written once for every dimension ──────────────────────────────────────────────────
//
//     image( p ) = Σ_i  opacity_i · ( exp( -q_i( p ) / 2 ) - E ) · color_i
//     q_i( p )   = Σ_ab  A_ab  dx_a  dx_b,        dx = p - center_i
//
// `A` is the INVERSE covariance, and it is symmetric: only its upper triangle is stored, which is
// why `num_coeff` is `D ( D + 1 ) / 2` -- 3 in 2D, 6 in 3D, and `splats.py` computes it rather than
// stating it.

/// `A_ab` of splat `i`, read out of the packed upper triangle.
///
/// Row-major over the triangle: `( 0,0 ) ( 0,1 ) … ( 0,D-1 ) ( 1,1 ) …`, so row `a` starts at
/// `a D - a ( a - 1 ) / 2`. In 2D that is the familiar `( a, b, c )`.
HD decltype( auto ) precision( const auto &splats, SI i, SI a, SI b, SI nb_dims ) {
    if ( a > b ) {
        const SI t = a; a = b; b = t;
    }
    return splats.cov_inv( i, a * nb_dims - ( a * ( a - 1 ) ) / 2 + ( b - a ) );
}

/// `q( dx ) = Σ_ab A_ab dx_a dx_b` -- the squared Mahalanobis distance.
template<class S,class TF>
HD TF quadratic_form( const S &splats, SI i, const TF *dx, SI nb_dims ) {
    TF res = 0;
    for ( SI a = 0; a < nb_dims; ++a )
        for ( SI b = 0; b < nb_dims; ++b )
            res += TF( precision( splats, i, a, b, nb_dims ) ) * dx[ a ] * dx[ b ];
    return res;
}

/// What splat `i` adds at pixel `x`, accumulated into `acc` (one entry per channel).
template<int D,class S,class TF>
HD void contribution( const S &splats, SI i, const SI *x, SI nb_channels, TF *acc ) {
    TF dx[ D ];
    for ( SI d = 0; d < D; ++d )
        dx[ d ] = TF( x[ d ] ) - TF( splats.centers( i, d ) );

    const TF q = quadratic_form( splats, i, dx, D );
    if ( q > TF( SIGMA_RADIUS * SIGMA_RADIUS ) )
        return;

    const TF w = TF( splats.opacities( i ) ) * ( std::exp( -TF( 0.5 ) * q ) - threshold<TF>() );
    for ( SI c = 0; c < nb_channels; ++c )
        acc[ c ] += w * TF( splats.colors( i, c ) );
}

// ── THE FOOTPRINT: which tiles a splat reaches ───────────────────────────────────────────────────

/// The half-extent of the box containing the ellipsoid `q <= SIGMA_RADIUS²`, per axis, in pixels.
///
/// THE ONE PLACE THIS EXAMPLE IS STILL TWO-DIMENSIONAL, and it says so rather than pretending. The
/// half-extent along axis `k` is `SIGMA_RADIUS · sqrt( ( A⁻¹ )_kk )`; with `A = ( a b ; b c )` the
/// inverse is `( c -b ; -b a ) / det`, which is the closed form below. In D dimensions it is the
/// diagonal of an inverse, i.e. a Cholesky -- perfectly doable, and deliberately left out: it is
/// the SINGLE function to replace, everything else above and below already runs in any dimension.
///
/// A precision matrix that is not positive definite has no footprint: we report an empty box rather
/// than return NaNs.
template<int D,class S,class TF>
HD bool half_box( const S &splats, SI i, TF *half ) {
    static_assert( D == 2, "splats: half_box is written for D = 2 -- in nD it needs the diagonal "
                           "of the inverse precision matrix (a Cholesky). Everything else here is "
                           "dimension-free." );
    const TF a = TF( precision( splats, i, 0, 0, D ) );
    const TF b = TF( precision( splats, i, 0, 1, D ) );
    const TF c = TF( precision( splats, i, 1, 1, D ) );

    const TF det = a * c - b * b;
    if ( ! ( det > 0 ) || ! ( a > 0 ) || ! ( c > 0 ) )
        return false;

    half[ 0 ] = TF( SIGMA_RADIUS ) * std::sqrt( c / det );
    half[ 1 ] = TF( SIGMA_RADIUS ) * std::sqrt( a / det );
    return true;
}

/// The tiles splat `i` reaches, as a box `[ lo, hi [` of tile multi-indices, already clipped to the
/// grid. False when it reaches none -- outside the image, or a degenerate precision matrix.
///
/// `nb_tiles` is read off the tensor the caller is about to write into: the tile grid IS one of its
/// dimensions, so neither an image size nor a tile side has to travel as an argument.
template<int D,SI SIDE,class S>
HD bool touched_tiles( const S &splats, SI i, const SI *nb_tiles, SI *lo, SI *hi ) {
    using TF = Scalar<decltype( splats.centers )>;

    TF half[ D ];
    if ( ! half_box<D>( splats, i, half ) )
        return false;

    for ( SI d = 0; d < D; ++d ) {
        const TF center = TF( splats.centers( i, d ) );
        lo[ d ] = SI( std::floor( ( center - half[ d ] ) / SIDE ) );
        hi[ d ] = SI( std::floor( ( center + half[ d ] ) / SIDE ) ) + 1;

        lo[ d ] = lo[ d ] < 0 ? 0 : lo[ d ];
        hi[ d ] = hi[ d ] > nb_tiles[ d ] ? nb_tiles[ d ] : hi[ d ];
        if ( lo[ d ] >= hi[ d ] )
            return false;
    }
    return true;
}

// ── THE TILE GRID AND THE CSR ROWS ───────────────────────────────────────────────────────────────
//
// THE ROWS OF A CSR ARE A SEQUENCE, not a grid: the offsets are a prefix sum along ONE order. So the
// tile grid is read row-major here, and the two functions below are the only places in this example
// where anything is flattened.

/// the row a tile multi-index stands for
template<int D>
HD SI row_of( const SI *tile, const SI *nb_tiles ) {
    SI res = 0;
    for ( SI d = 0; d < D; ++d )
        res = res * nb_tiles[ d ] + tile[ d ];
    return res;
}

/// The pixel this item IS -- as plain numbers, in `x` -- and the row of the tile it belongs to.
///
/// These few lines are the whole bridge between the axes, which exist only at compile time, and a
/// geometry written as `for ( d = 0; d < D; ++d )`, which is how one writes one. The axes a `vmap`
/// added are not ours, and `coords.axes - batch_axes` is the one line that says so.
template<int D,SI SIDE>
HD SI pixel_and_row( auto coords, auto batch_axes, const auto &image, SI *x ) {
    const auto main_axes = coords.axes - batch_axes;
    static_assert( DECAYED_TYPE_OF( main_axes )::ct_size == D,
                   "splats: the image must have one axis per dimension of the splats" );

    SI tile[ D ], nb_tiles[ D ];
    for_each_indexed( main_axes, [&]( auto axis, int d ) {
        x[ d ] = SI( coords[ axis ] );
        nb_tiles[ d ] = ( SI( image.size( axis ) ) + SIDE - 1 ) / SIDE;
    } );
    for ( SI d = 0; d < D; ++d )
        tile[ d ] = x[ d ] / SIDE;

    return row_of<D>( tile, nb_tiles );
}

/// How many dimensions the splats live in. `nb_dims` is a `CtShapeVar`, so the answer is in the
/// TYPE -- which is what lets the bodies below size their little arrays on the stack.
#define SPLATS_NB_DIMS( args ) DECAYED_TYPE_OF( args.inputs.splats.centers.size( num_dim ) )::value

// ── PASS 1a: COUNT, writing no list at all ───────────────────────────────────────────────────────

template<int D,SI SIDE>
struct Count {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto &counts = args.outputs.counts;

        SI nb_tiles[ D ], lo[ D ], hi[ D ];
        to_array<D>( counts.shape(), nb_tiles );
        if ( ! touched_tiles<D,SIDE>( args.inputs.splats, SI( coords[ num_splat ] ),
                                      nb_tiles, lo, hi ) )
            return;

        for_each_in_box<D>( lo, hi, [&]( auto tile ) {
            auto &cell = counts( tile ).ref();
            atomic_add( cell, std::remove_reference_t<decltype( cell )>( 1 ) );
        } );
    }
};

template<SI SIDE>
void count( auto &&queue, auto &&batch_axes, auto &&args ) {
    // ONE ITEM PER SPLAT, and the domain is read off the splats themselves.
    constexpr int D = SPLATS_NB_DIMS( args );
    queue.run_parallel( Count<D,SIDE>{},
                        batch_axes + args.inputs.splats.opacities.domain( num_splat ),
                        args, batch_axes );
}

// ── PASS 1b: FILL, the offsets being known ───────────────────────────────────────────────────────

template<int D,SI SIDE>
struct Fill {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const SI i = SI( coords[ num_splat ] );
        const auto &cursors = args.outputs.cursors;

        SI nb_tiles[ D ], lo[ D ], hi[ D ];
        to_array<D>( cursors.shape(), nb_tiles );
        if ( ! touched_tiles<D,SIDE>( args.inputs.splats, i, nb_tiles, lo, hi ) )
            return;

        for_each_in_box<D>( lo, hi, [&]( auto tile ) {
            // the cursor says where in its row the next splat goes. Several work-items write into
            // the SAME row, so the ticket an `atomic_fetch_add` returns is this one's slot.
            auto &cursor = cursors( tile ).ref();
            using TC = std::remove_reference_t<decltype( cursor )>;
            const SI k = SI( atomic_fetch_add( cursor, TC( 1 ) ) );

            SI at[ D ];
            to_array<D>( tile, at );
            // `touching( row, slot )`: the first index looks the row up in the offsets.
            args.outputs.touching( row_of<D>( at, nb_tiles ), k ) = i;
        } );
    }
};

template<SI SIDE>
void fill( auto &&queue, auto &&batch_axes, auto &&args ) {
    constexpr int D = SPLATS_NB_DIMS( args );
    queue.run_parallel( Fill<D,SIDE>{},
                        batch_axes + args.inputs.splats.opacities.domain( num_splat ),
                        args, batch_axes );
}

// ── PASS 2: the image, and its adjoint ───────────────────────────────────────────────────────────

template<int D,SI SIDE>
struct RenderPixel {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        using TF = Scalar<decltype( args.inputs.splats.centers )>;
        constexpr SI C = DECAYED_TYPE_OF( args.outputs.image.size( num_channel ) )::value;

        SI x[ D ];
        const SI row = pixel_and_row<D,SIDE>( coords, batch_axes, args.outputs.image, x );

        // `touching( row, k )`: the `row` looks the offsets up, the `k` indexes inside the row.
        // Neither a bound nor a count to pass alongside -- the tensor carries them.
        TF acc[ C ] = {};
        for ( SI k = 0; k < args.inputs.touching.row_size( row ); ++k )
            contribution<D>( args.inputs.splats, SI( args.inputs.touching( row, k ) ), x, C, acc );

        // `coords` carries only the pixel axes, so what is left is the channel one: a rank-1 view.
        auto out = args.outputs.image( coords );
        for ( SI c = 0; c < C; ++c )
            out( c ) = acc[ c ];
    }
};

template<SI SIDE>
void render( auto &&queue, auto &&batch_axes, auto &&args ) {
    // ONE ITEM PER PIXEL -- not per image scalar: leaving the channel axis out of the domain is
    // what keeps each item from recomputing the same Gaussian once per channel.
    constexpr int D = SPLATS_NB_DIMS( args );
    queue.run_parallel( RenderPixel<D,SIDE>{},
                        batch_axes + args.outputs.image.domain( args.outputs.image.axes - num_channel ),
                        args, batch_axes );
}

/// THE ADJOINT of pass 2, for one pixel: every splat of the tile gets its share.
///
/// All contributions are ACCUMULATED ATOMICALLY: a splat is hit by every pixel it covers, hence by
/// different work-items. That is the opposite of `examples/diffusion`'s adjoint, which was a pure
/// gather -- and it is what exercises the zero-seeding of shared outputs (see
/// `CallArg_Tensor.cpp_seed_member`).
///
/// With `L` the loss and `g` the pixel's cotangent, and `<g,color>` written `gc`:
///     dL/dcolor_i,c = w g_c                               with w = opacity ( e - E )
///     dL/dopacity_i = ( e - E ) gc
///     dL/dcenter_ia = opacity e gc  Σ_b A_ab dx_b
///     dL/dA_i,ab    = opacity e gc  ( -½ dx_a dx_b ) , doubled off the diagonal (packed storage)
///
/// The VALUE goes through `( e - E )`, the GEOMETRIC derivative through `e` alone: the constant the
/// continuous truncation subtracts does not depend on `q`, so it vanishes on differentiating.
template<int D,SI SIDE>
struct RenderPixelBwd {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto splats = args.inputs.splats;
        const auto grad = args.grad_of_inputs.splats;
        using TF = Scalar<decltype( splats.centers )>;
        constexpr SI C = DECAYED_TYPE_OF( args.grad_of_outputs.image.size( num_channel ) )::value;

        SI x[ D ];
        const SI row = pixel_and_row<D,SIDE>( coords, batch_axes, args.grad_of_outputs.image, x );

        const auto cotangent = args.grad_of_outputs.image( coords );
        TF g[ C ];
        for ( SI c = 0; c < C; ++c )
            g[ c ] = TF( cotangent( c ) );

        for ( SI k = 0; k < args.inputs.touching.row_size( row ); ++k ) {
            const SI i = SI( args.inputs.touching( row, k ) );

            TF dx[ D ];
            for ( SI d = 0; d < D; ++d )
                dx[ d ] = TF( x[ d ] ) - TF( splats.centers( i, d ) );

            const TF q = quadratic_form( splats, i, dx, D );
            if ( q > TF( SIGMA_RADIUS * SIGMA_RADIUS ) )
                continue;

            const TF opacity = TF( splats.opacities( i ) );
            const TF e = std::exp( -TF( 0.5 ) * q );
            const TF w = opacity * ( e - threshold<TF>() );   // the value
            const TF de = opacity * e;                        // what the derivative in q goes through

            // <g, color>: everything geometric passes through this single number
            TF gc = 0;
            for ( SI c = 0; c < C; ++c )
                gc += g[ c ] * TF( splats.colors( i, c ) );

            if constexpr ( grad.colors.is_valid )
                for ( SI c = 0; c < C; ++c )
                    atomic_add( grad.colors( i, c ).ref(), w * g[ c ] );

            if constexpr ( grad.opacities.is_valid )
                atomic_add( grad.opacities( i ).ref(), ( e - threshold<TF>() ) * gc );

            if constexpr ( grad.centers.is_valid )
                for ( SI a = 0; a < D; ++a ) {
                    TF s = 0;
                    for ( SI b = 0; b < D; ++b )
                        s += TF( precision( splats, i, a, b, D ) ) * dx[ b ];
                    atomic_add( grad.centers( i, a ).ref(), de * gc * s );
                }

            if constexpr ( grad.cov_inv.is_valid )
                for ( SI a = 0, packed = 0; a < D; ++a )
                    for ( SI b = a; b < D; ++b, ++packed )
                        atomic_add( grad.cov_inv( i, packed ).ref(),
                                    de * gc * ( a == b ? -TF( 0.5 ) : -TF( 1 ) ) * dx[ a ] * dx[ b ] );
        }
    }
};

template<SI SIDE>
void render_bwd( auto &&queue, auto &&batch_axes, auto &&args ) {
    // A NULL COTANGENT HAS NOTHING TO PROPAGATE: the test is on the LAUNCH, not inside the body --
    // a symbolic zero is another TYPE, with neither a domain nor values to read.
    constexpr int D = SPLATS_NB_DIMS( args );
    if constexpr ( ! DECAYED_TYPE_OF( args.grad_of_outputs.image )::surely_null )
        queue.run_parallel( RenderPixelBwd<D,SIDE>{},
                            batch_axes + args.grad_of_outputs.image.domain(
                                args.grad_of_outputs.image.axes - num_channel ),
                            args, batch_axes );
}

#undef SPLATS_NB_DIMS

} // namespace splats
