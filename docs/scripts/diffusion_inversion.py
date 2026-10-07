"""Records the diffusion tutorial's inversion as a GIF: gradient descent THROUGH `evolve`, from the
temperature observed after `nb_steps` steps back to a state that explains it.

    python docs/scripts/diffusion_inversion.py [--out docs/public/anim/diffusion_inversion.gif]
"""
import argparse, io, sys
from pathlib import Path

ROOT = Path( __file__ ).resolve().parents[ 2 ]
sys.path.insert( 0, str( ROOT / "examples" / "diffusion" ) )

import jax, jax.numpy as jnp
import numpy
jax.config.update( "jax_enable_x64", True )

from diffusion import evolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 40 )
    ap.add_argument( "--nb-steps", type = int, default = 25 )
    ap.add_argument( "--iters", type = int, default = 600 )
    ap.add_argument( "--out", default = str( ROOT / "docs" / "public" / "anim" / "diffusion_inversion.gif" ) )
    a = ap.parse_args()
    n, coef = a.n, 0.2

    # the truth: two bumps and a ridge, zero on the border
    x = jnp.linspace( 0, 1, n )
    X, Y = jnp.meshgrid( x, x )
    bump = lambda cx, cy, s: jnp.exp( -( ( X - cx ) ** 2 + ( Y - cy ) ** 2 ) / ( 2 * s * s ) )
    true = bump( .3, .35, .06 ) + 0.7 * bump( .68, .6, .08 ) + 0.4 * jnp.exp( -( ( X - .5 ) ** 2 ) / ( 2 * .02 ** 2 ) ) * ( Y > .15 ) * ( Y < .5 )
    mask = jnp.ones( ( n, n ) ).at[ 0, : ].set( 0 ).at[ -1, : ].set( 0 ).at[ :, 0 ].set( 0 ).at[ :, -1 ].set( 0 )
    true = true * mask
    observed = evolve( true, coef, a.nb_steps )

    loss = lambda u: ( ( evolve( u, coef, a.nb_steps ) - observed ) ** 2 ).sum()
    loss_j, grad_j = jax.jit( loss ), jax.jit( jax.grad( loss ) )

    u = jnp.zeros( ( n, n ) )
    frames, losses = [], []
    for t in range( a.iters + 1 ):
        frames.append( numpy.asarray( u ) )
        losses.append( float( loss_j( u ) ) )
        if t % 100 == 0:
            print( f"iter {t:4d}  loss {losses[ -1 ]:.3e}", flush = True )
        u = u - 0.4 * grad_j( u )

    write_gif( Path( a.out ), frames, numpy.asarray( true ), numpy.asarray( observed ), numpy.asarray( losses ) )
    print( "saved", a.out )


def write_gif( path, frames, true, observed, losses ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from PIL import Image

    vmax = float( max( true.max(), observed.max() ) )
    keep = sorted( set( list( range( 0, 40, 2 ) ) + list( range( 40, 200, 8 ) ) + list( range( 200, len( frames ), 20 ) ) + [ len( frames ) - 1 ] ) )
    out = []
    for t in keep:
        fig, ax = plt.subplots( 1, 4, figsize = ( 12, 3.1 ), dpi = 72, gridspec_kw = dict( width_ratios = [ 1, 1, 1, 1.3 ] ) )
        for x, im, title in ( ( ax[ 0 ], observed, "observed after the steps" ), ( ax[ 1 ], frames[ t ], f"estimate, iteration {t}" ), ( ax[ 2 ], true, "initial state ( unknown )" ) ):
            x.imshow( im, vmin = 0, vmax = vmax, cmap = "inferno" ); x.set_title( title, fontsize = 9 ); x.axis( "off" )
        ax[ 3 ].semilogy( losses[ : t + 1 ], color = "#0f766e" )
        ax[ 3 ].set_xlim( 0, len( losses ) ); ax[ 3 ].set_ylim( max( losses[ -1 ] / 2, 1e-12 ), losses[ 0 ] * 2 )
        ax[ 3 ].set_title( "loss", fontsize = 9 ); ax[ 3 ].set_xlabel( "iteration" )
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig( buf, format = "png" ); plt.close( fig )
        out.append( Image.open( buf ).convert( "P", palette = Image.ADAPTIVE ) )
    path.parent.mkdir( parents = True, exist_ok = True )
    out[ 0 ].save( path, save_all = True, append_images = out[ 1: ], duration = [ 90 ] * ( len( out ) - 1 ) + [ 2000 ], loop = 0, optimize = True )


if __name__ == "__main__":
    main()
