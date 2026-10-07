"""Fits a few hundred Gaussians to a cartoon face by gradient descent THROUGH loom's renderer, and
records the optimisation as an animation. The face is drawn here, in code: nothing to download. Used by the splats tutorial; run it from the repository root:

    python docs/scripts/splats_fit.py [--out docs/public/anim/splats_fit.gif]

The model is ordinary Jax code around ONE loom call: the parameters ( centers, log-sigmas, an angle,
colours, opacities ) are turned into the packed inverse covariance by plain `jnp`, then
`examples/splats` renders them. `jax.grad` goes through both.
"""
import argparse, math, sys
from pathlib import Path

ROOT = Path( __file__ ).resolve().parents[ 2 ]
sys.path.insert( 0, str( ROOT / "examples" / "splats" ) )

import jax, jax.numpy as jnp
import numpy
jax.config.update( "jax_enable_x64", True )

import loom
from splats import Splats, render_scene


def cov_inv_of( log_sigma, angle ):
    """Packed upper triangle ( a, b, c ) of R diag( 1/s0², 1/s1² ) Rᵀ."""
    i0, i1 = jnp.exp( -2 * log_sigma[ :, 0 ] ), jnp.exp( -2 * log_sigma[ :, 1 ] )
    ca, sa = jnp.cos( angle ), jnp.sin( angle )
    return jnp.stack( [ ca * ca * i0 + sa * sa * i1, ca * sa * ( i0 - i1 ), sa * sa * i0 + ca * ca * i1 ], axis = 1 )


def make_splats( p, nb_dims = 2 ):
    s = Splats( nb_dims = nb_dims, nb_channels = 3 )
    s.centers   = p[ "centers" ]
    s.cov_inv   = cov_inv_of( p[ "log_sigma" ], p[ "angle" ] )
    s.colors    = p[ "colors" ]
    s.opacities = p[ "opacities" ]
    return s


def image_of( p, shape ):
    """( H, W, 3 ): what `render_scene` returns is already the backend array."""
    return render_scene( make_splats( p ), shape )


def draw_face( size ):
    """A friendly cartoon face, ( size, size, 3 ) floats in [ 0, 1 ]. Drawn at 4x and shrunk, so the
    edges are smooth."""
    from PIL import Image, ImageDraw
    k = 4
    S = size * k
    im = Image.new( "RGB", ( S, S ), ( 24, 78, 92 ) )
    d = ImageDraw.Draw( im )
    u = lambda v: v * S                                           # unit square -> pixels
    box = lambda cx, cy, rx, ry: [ u( cx - rx ), u( cy - ry ), u( cx + rx ), u( cy + ry ) ]

    d.ellipse( box( .50, .53, .31, .33 ), fill = ( 255, 205, 70 ) )                       # head
    d.ellipse( box( .20, .55, .045, .075 ), fill = ( 255, 190, 60 ) )                     # ears
    d.ellipse( box( .80, .55, .045, .075 ), fill = ( 255, 190, 60 ) )
    d.pieslice( box( .50, .33, .31, .20 ), 180, 360, fill = ( 120, 62, 30 ) )             # hair
    d.ellipse( box( .50, .15, .06, .05 ), fill = ( 120, 62, 30 ) )                        # a tuft
    for sx in ( -1, 1 ):
        d.ellipse( box( .50 + sx * .125, .50, .062, .075 ), fill = ( 255, 255, 255 ) )    # eyes
        d.ellipse( box( .50 + sx * .115, .51, .032, .042 ), fill = ( 40, 30, 30 ) )       # pupils
        d.ellipse( box( .50 + sx * .105, .495, .011, .014 ), fill = ( 255, 255, 255 ) )   # glints
        d.ellipse( box( .50 + sx * .20, .625, .05, .032 ), fill = ( 255, 140, 120 ) )     # cheeks
    d.chord( box( .50, .64, .15, .12 ), 15, 165, fill = ( 150, 40, 50 ) )                 # smile
    d.chord( box( .50, .655, .10, .06 ), 20, 160, fill = ( 255, 120, 120 ) )              # tongue
    d.ellipse( box( .50, .57, .022, .017 ), fill = ( 235, 150, 60 ) )                     # nose
    return numpy.asarray( im.resize( ( size, size ), Image.LANCZOS ), dtype = float ) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--size", type = int, default = 96 )
    ap.add_argument( "--nb", type = int, default = 500 )
    ap.add_argument( "--steps", type = int, default = 500 )
    ap.add_argument( "--out", default = str( ROOT / "docs" / "public" / "anim" / "splats_fit.gif" ) )
    a = ap.parse_args()
    shape = ( a.size, a.size )

    rng = numpy.random.default_rng( 7 )
    target = jnp.asarray( draw_face( a.size ) )

    # the start: small Gaussians scattered at random, each taking the colour of the target under its
    # centre -- the usual initialisation. Everything else ( where they go, how big, how opaque ) is
    # for the descent to find.
    centers = rng.uniform( 0, a.size, ( a.nb, 2 ) )
    colors = numpy.asarray( target )[ centers[ :, 0 ].astype( int ), centers[ :, 1 ].astype( int ) ]
    p = dict(
        centers   = jnp.asarray( centers ),
        log_sigma = jnp.log( jnp.full( ( a.nb, 2 ), 3.0 ) ),
        angle     = jnp.zeros( a.nb ),
        colors    = jnp.asarray( colors ),
        opacities = jnp.full( a.nb, 0.4 ),
    )

    def loss( p ):
        return jnp.mean( ( image_of( p, shape ) - target ) ** 2 )

    value_and_grad = jax.value_and_grad( loss )
    lr = dict( centers = 0.12, log_sigma = 0.01, angle = 0.01, colors = 0.01, opacities = 0.01 )
    m = { k: jnp.zeros_like( v ) for k, v in p.items() }
    v = { k: jnp.zeros_like( x ) for k, x in p.items() }
    b1, b2, eps = 0.9, 0.99, 1e-12

    frames, losses = [], []
    for t in range( a.steps + 1 ):
        l, g = value_and_grad( p )
        losses.append( float( l ) )
        frames.append( numpy.asarray( image_of( p, shape ) ) )
        if t % 20 == 0:
            print( f"step {t:4d}  loss {float( l ):.3e}", flush = True )
        for k in p:
            m[ k ] = b1 * m[ k ] + ( 1 - b1 ) * g[ k ]
            v[ k ] = b2 * v[ k ] + ( 1 - b2 ) * g[ k ] ** 2
            mh, vh = m[ k ] / ( 1 - b1 ** ( t + 1 ) ), v[ k ] / ( 1 - b2 ** ( t + 1 ) )
            p[ k ] = p[ k ] - lr[ k ] * mh / ( jnp.sqrt( vh ) + eps )
        p[ "log_sigma" ] = jnp.clip( p[ "log_sigma" ], math.log( 0.9 ), math.log( 12 ) )
        p[ "opacities" ]  = jnp.clip( p[ "opacities" ], 0.02, 1.5 )

    write_gif( Path( a.out ), numpy.stack( frames ), numpy.asarray( target ), numpy.asarray( losses ) )
    print( "saved", a.out )


def write_gif( path, frames, target, losses ):
    """target | current | loss curve, one GIF frame per kept optimisation step. Early steps are
    kept densely ( that is where the picture changes ), later ones every few steps."""
    import io
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from PIL import Image

    keep = sorted( set( list( range( 0, 60, 3 ) ) + list( range( 60, 200, 5 ) ) + list( range( 200, len( frames ), 10 ) ) + [ len( frames ) - 1 ] ) )
    shown = lambda im: numpy.clip( im, 0, 1 )
    out = []
    for t in keep:
        fig, ax = plt.subplots( 1, 3, figsize = ( 9, 3.1 ), dpi = 80, gridspec_kw = dict( width_ratios = [ 1, 1, 1.25 ] ) )
        ax[ 0 ].imshow( shown( target ), interpolation = "nearest" );   ax[ 0 ].set_title( "target" )
        ax[ 1 ].imshow( shown( frames[ t ] ), interpolation = "nearest" ); ax[ 1 ].set_title( f"splats, step {t}" )
        for x in ax[ :2 ]:
            x.axis( "off" )
        ax[ 2 ].semilogy( losses[ : t + 1 ], color = "#0f766e" )
        ax[ 2 ].set_xlim( 0, len( losses ) ); ax[ 2 ].set_ylim( losses.min() / 2, losses.max() * 2 )
        ax[ 2 ].set_title( "loss" ); ax[ 2 ].set_xlabel( "step" )
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig( buf, format = "png" ); plt.close( fig )
        out.append( Image.open( buf ).convert( "P", palette = Image.ADAPTIVE ) )
    path.parent.mkdir( parents = True, exist_ok = True )
    out[ 0 ].save( path, save_all = True, append_images = out[ 1: ], duration = [ 90 ] * ( len( out ) - 1 ) + [ 2000 ], loop = 0, optimize = True )


if __name__ == "__main__":
    main()
