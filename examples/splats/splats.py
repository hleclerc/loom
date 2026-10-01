"""Gaussian splatting: a RAGGED structure whose length the kernel discovers.

The second deliberately FOREIGN user of loom (the first is `examples/diffusion`). It imports only
`loom`, its C++ lives next door in `splats.h`, and it exercises exactly what `diffusion` did not:

  * a structure the host sizes from a count A KERNEL WROTE -- so a shape that is not known at trace
    time, and the host read-back that is loom's one irreplaceable trick;
  * a PIPELINE of several passes sharing aggregates;
  * an adjoint in ATOMIC ACCUMULATION (`diffusion`'s was a pure gather).

    image( p ) = sum_i  opacity_i * exp( -q_i( p ) / 2 ) * color_i

ADDITIVE blending: no order, hence no sort -- alpha compositing comes later.

= Why the ragged structure is not a convenience here

A pixel must only look at the splats that reach it, otherwise rendering is O( pixels x splats ). So
every tile needs the list of the Gaussians TOUCHING it, and the length of that list DEPENDS ON THE
DATA.

What XLA can and cannot do, exactly -- because the argument has to be honest: it CAN build such a
structure, provided the total is BOUNDED ahead of time (a count, a prefix sum, a fixed-size
scatter). What it cannot do is DISCOVER it. The difference is paid for: too small a bound loses
splats silently, a safe bound makes everyone pay the worst case. Here the kernel COUNTS, the host
READS the total, and the list is allocated at exactly that -- no bound anywhere. `reference.py`
measures what a bound would have cost.

= One storage, and why this one

The lists are kept in CSR: the bounds of each row, plus one flat list of everything. It was a
measured choice and not a taste -- the obvious alternative, a PADDED rectangle `tiles x
max_per_tile`, cost 2.4 to 5.1 times the memory on the same scenes, for the same images. The
comparison is in the README, and `test_splats.py` still checks the ratio on every run (it is
arithmetic on the row lengths, so it needs no second implementation to keep alive).

= How to read this

    splats.py     this file: the data, the calls, and what each pass does
    splats.h      the C++ of the four kernels, in the same order
    reference.py  what one would write without loom, and what the bound costs
    test_splats.py

= Nothing is flat, and nothing is named `x`

No flat rank, no division by an image side, no `int( ... .value )`, and no axis called `x` or `y`.
Every kernel asks for THE DOMAIN IT WANTS and reads its coordinates BY NAME; the C++ then copies
them into a small array and loops over `d`. The whole example therefore has no dimension baked in
-- `splats.h` has exactly one function that is still 2D, and it says so.
"""
# `loom.Thing` and never a bare `Thing`: in an example one must see where every name comes from.
import loom

TILE_SIDE = 16          # a tile is TILE_SIDE pixels along every axis


class Splats( loom.Aggregate ):
    """The Gaussians: center, inverse covariance, color, opacity.

    `num_coeff` is a COMPUTED extent: the inverse covariance is SYMMETRIC, so it has
    `D ( D + 1 ) / 2` independent coefficients -- 3 in 2D, 6 in 3D. That is a formula, not an
    affine expression, and loom evaluates it because `nb_dims` is known at compile time (see
    `ShapeVar.set_formula`). Nothing to prescribe, hence nothing to keep in agreement.
    """
    centers     : loom.RealTensor[ "num_splat", "num_dim" ]
    cov_inv     : loom.RealTensor[ "num_splat", "num_coeff" ]
    colors      : loom.RealTensor[ "num_splat", "num_channel" ]
    opacities   : loom.RealTensor[ "num_splat" ]

    num_splat   : loom.Axis[ "nb_splats" ]
    num_dim     : loom.Axis[ "nb_dims" ]
    num_coeff   : loom.Axis[ "nb_dims * ( nb_dims + 1 ) / 2" ]
    num_channel : loom.Axis[ "nb_channels" ]

    nb_splats   : loom.ShapeVar
    nb_dims     : loom.CtShapeVar
    nb_channels : loom.CtShapeVar


# `shape` below is ALWAYS the image shape, axis by axis: `( 480, 640 )` for a 2D image, and axis `d`
# of the image is coordinate `d` of a center. That single convention is what lets everything here be
# written without ever naming a dimension.

def _tile_axes( shape ):
    """The axes of the tile grid -- one per dimension of the image, and that is all there is to it.
    `num_tile_0`, `num_tile_1`, ... on the C++ side."""
    return [ loom.Axis( ( n + TILE_SIDE - 1 ) // TILE_SIDE, name = f"num_tile_{ k }" )
             for k, n in enumerate( shape ) ]


def _empty_image( splats, shape ):
    """An image of `shape`, plus the channel axis -- which the splats already own."""
    pixels = [ loom.Axis( n, name = f"num_pixel_{ k }" ) for k, n in enumerate( shape ) ]
    return loom.RealTensor[ *pixels, splats.num_channel ]()


# Each kernel is written where it is LAUNCHED, right below. `loom.FfiCode.inline` takes the body of
# the handler -- loom calls `kernel( queue, batch_axes, args )` -- and ours forwards to `splats.h`,
# which holds the C++. `TILE_SIDE` crosses as a TEMPLATE argument, so it stays a compile-time
# constant on the C++ side without being written there: the knob lives here, where one starts
# reading.


def touching_gaussians( splats, shape ):
    """PASS 1: for each tile, which Gaussians reach it -- a `loom.CsrTensor` sized EXACTLY.

    TWO PASSES, and that is what the exactness costs: COUNT, then -- the offsets being known --
    FILL. In between, `CsrTensor.from_counts` does the prefix sum ON THE DEVICE and allocates
    `values` at exactly the total.

    THE ONE THING LOOM CAN DO THAT A JIT CANNOT is the line in the middle: sizing `values` means
    READING the total, which is a count a kernel has just written. Under `jit` that count is a
    tracer, so the total would have to be bounded before tracing -- and a bound is precisely what
    this example does without.

    NOT differentiable: it returns integers, and its link to the centers is discontinuous -- a splat
    either enters a tile or it does not.
    """
    tiles = _tile_axes( shape )

    # 1. count. Nothing is written but one integer per tile, whose size is known: no capacity to
    #    guess, no list to allocate yet.
    counts = loom.ffi_call(
        "splats_count",
        loom.FfiCode.inline( f"splats::count< { TILE_SIDE } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        counts = loom.out( loom.IntTensor[ *tiles ]() ),
    )

    # 2. the storage. The GRID of counts becomes a SEQUENCE of rows here, read row-major -- a CSR's
    #    rows are a sequence by construction, its offsets being a prefix sum along one order.
    touching = loom.CsrTensor.from_counts( counts )

    # 3. fill -- the per-tile cursor gives the place within a row. Two outputs, so the call hands
    #    back both, in the order they were given; only the second interests us.
    _, touching = loom.ffi_call(
        "splats_fill",
        loom.FfiCode.inline( f"splats::fill< { TILE_SIDE } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        cursors = loom.out( loom.IntTensor[ *tiles ]() ),
        touching = loom.out( touching, writes = ( "values", ) ),
    )
    return touching


def render( splats, touching, shape ):
    """PASS 2: the image. Differentiable with respect to everything the splats carry.

    Two kernels, and the second is the ADJOINT -- a kernel in its own right, running on other
    buffers with its own launch domain. The call takes both and names the pair.

    `.value` because what a caller wants here is the BACKEND array, not loom's tensor: it is what
    `jax.vjp` differentiates through, and what `numpy.asarray` reads."""
    return loom.ffi_call(
        "splats_render",
        loom.FfiCode.inline( f"splats::render< { TILE_SIDE } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        loom.FfiCode.inline( f"splats::render_bwd< { TILE_SIDE } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        touching = touching,
        image = loom.out( _empty_image( splats, shape ) ),
    ).value


def render_scene( splats, shape ):
    """The whole thing, differentiable: the lists are built on centers whose gradient is CUT (which
    tile a splat lands in is discontinuous, and that is not where the derivative goes -- a 3DGS does
    the same), then the image is rendered."""
    return render( splats, touching_gaussians( _detached( splats ), shape ), shape )


def _detached( splats ):
    """The same splats, detached: what pass 1 reads. The counts are SHARED rather than restated --
    passing a `ShapeVar` makes both aggregates reference the same object."""
    other = Splats( nb_dims = splats.nb_dims, nb_channels = splats.nb_channels )
    other.centers   = loom.stop_gradient( splats.centers )
    other.cov_inv   = loom.stop_gradient( splats.cov_inv )
    other.colors    = loom.stop_gradient( splats.colors )
    other.opacities = loom.stop_gradient( splats.opacities )
    return other
