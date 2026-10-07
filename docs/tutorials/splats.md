# 2 · Gaussian splatting

A renderer for anisotropic Gaussians, differentiable with respect to everything they carry. By the
end you will have a pipeline of **three kernels and an adjoint** that fits a handful of Gaussians to
an image by gradient descent:

![A cartoon face rebuilt from 500 Gaussians: starting from scattered coloured blobs, the splats on the left converge to the target face while the loss falls over 500 steps](/anim/splats_fit.gif)

*500 small Gaussians, scattered at random, pulled onto a cartoon face by gradient descent through the
renderer. Left to right: the target, the current rendering, the loss. 500 steps, about 25 seconds on a
laptop CPU, a 96×96 image. The face is drawn in code, so there is nothing to download; the script is
[`docs/scripts/splats_fit.py`](https://github.com/hleclerc/loom/blob/main/docs/scripts/splats_fit.py).*

The first tutorial, [Diffusion](./diffusion), had a fixed-shape problem: the output was the same size
as the input. This one does not. A pixel must only look at the Gaussians that reach it, which means an
**index whose length depends on the data**, and that is exactly where a Jax or Torch program gets
stuck, and where loom has something to offer.

The example lives in [`examples/splats/`](https://github.com/hleclerc/loom/tree/main/examples/splats):

```text
splats.py        the data, the calls, and what each pass does
splats.h         the C++ of the four kernels, in the same order
reference.py     what one would write without loom, and what its bound costs
test_splats.py   seven tests, one of them a benchmark
```

```bash
cd examples/splats
errand test_splats            # the tests
errand -k bench test_splats   # the timings ( takes the machine for itself )
```

## The case

Additive Gaussian splatting: every Gaussian adds its colour to the pixels it covers, no sorting, no
occlusion.

```text
image( p ) = Σ_i  opacity_i · ( exp( −q_i( p ) / 2 ) − E ) · color_i          q_i( p ) = Σ_ab A_ab dx_a dx_b,   dx = p − center_i
```

`A` is the *inverse* covariance, so it is symmetric: only its upper triangle is stored, which is
`D ( D + 1 ) / 2` numbers (3 in 2-D, 6 in 3-D). A Gaussian has a bounded footprint: past 3 standard
deviations it contributes nothing.

Rendering this the obvious way costs `pixels × splats`. So rendering is organised by **tiles** of
16×16 pixels: for each tile, the list of the Gaussians that touch it. A pixel loops over its tile's
list and nothing else.

## The question

> **Who decides the size of that list, and when?**

It depends on where the Gaussians are and how large they are. It is not known until someone has
looked. Tiles are uneven in the real world: in the test scene below, the number of Gaussians per tile
ranges from 18 to 381 around a mean of 146.

What can Jax do? It *can* build such a structure, provided the total is **bounded ahead of time**:
a count, a prefix sum, a fixed-size scatter. What it cannot do is *discover* the total. The
difference is paid for: a bound too small loses Gaussians silently, a safe bound makes everyone pay
the worst case.

What loom adds is one thing, and it exists nowhere else: **the host can read a count a kernel has just
written, and size the next allocation with it.** The list is then exactly as long as its content.

## The data, with its axes written once

```python
class Splats( loom.Aggregate ):
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
```

A `loom.Aggregate` is a group of tensors that travel together, with the axes they share. Three ideas
in these few lines:

- **An axis is a reference.** `num_dim` is the same object for `centers` and for the Gaussian's
  dimension everywhere else, so two tensors that name it cannot disagree about its length.
- **An extent can be computed.** `num_coeff` is `nb_dims * ( nb_dims + 1 ) / 2`: a formula, evaluated
  by loom, so there is no number to keep in agreement with the dimension. `nb_dims = 2` gives 3.
- **`CtShapeVar` is a compile-time count**: `nb_dims` and `nb_channels` are engraved into the
  generated C++, so the kernel's inner loops are sized on the stack.

`Screen` plays the same game for the image and its tile grid. The tile grid is
`( nb_pixels + tile_side - 1 ) // tile_side` per dimension: one formula, declared once, evaluated by
loom, so the grid follows the image by construction. An image whose sides are not multiples of the
tile (37×53, say) is fine; the last row of tiles hangs over the edge.

→ [Axes and coordinates](/guide/axes)

## Pass 1: which Gaussians touch which tile

Three steps, and the middle one is the one that matters.

```python
def touching_gaussians( splats, screen ):
    side = int( screen.tile_side )

    # 1. count
    counts = loom.ffi_call(
        "splats_count",
        loom.FfiCode.inline( f"splats::count< { side } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        counts = loom.out( loom.IntTensor[ screen.num_tile ]() ),
    )

    # 2. the storage, sized by what the kernel wrote
    touching = loom.CsrTensor.from_counts( counts )

    # 3. fill
    _, touching = loom.ffi_call(
        "splats_fill",
        loom.FfiCode.inline( f"splats::fill< { side } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        cursors  = loom.out( loom.IntTensor[ screen.num_tile ]() ),
        touching = loom.out( touching, writes = ( "values", ) ),
    )
    return touching
```

**1 · Count.** One item per Gaussian. It finds the box of tiles its footprint reaches, and adds 1 to
the count of each:

```cpp
template<int D,SI SIDE>
struct Count {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto &counts = args.outputs.counts;

        SI nb_tiles[ D ], lo[ D ], hi[ D ];
        to_array<D>( counts.shape(), nb_tiles );
        if ( ! touched_tiles<D,SIDE>( args.inputs.splats, SI( coords[ num_splat ] ), nb_tiles, lo, hi ) )
            return;

        for_each_in_box<D>( lo, hi, [&]( auto tile ) {
            auto &cell = counts( tile ).ref();
            atomic_add( cell, std::remove_reference_t<decltype( cell )>( 1 ) );
        } );
    }
};
```

Nothing is written but one integer per tile. Its size is known; there is no capacity to guess, and no
list yet. Several Gaussians reach the same tile from different work-items, hence `atomic_add`.

**2 · The storage.** `CsrTensor.from_counts( counts )` does the prefix sum **on the device**, reads
the **total** back to the host, and allocates `values` at exactly that size. This line is the whole
point of the example. Sizing `values` means reading a number a kernel has just written; under a bare
`jit` that number is a tracer, so XLA would need a bound chosen *before* tracing.

A `CsrTensor` is the standard compressed layout: `offsets` holds `nb_rows + 1` bounds (so a row's
length is a subtraction of neighbours, and the last bound *is* the total) and `values` is the flat
list. On the C++ side, `touching( row, k )` looks the row up in the offsets and returns the `k`-th
element of that row.

**3 · Fill.** The offsets are known now, so each Gaussian writes where it belongs. A per-tile cursor
hands out the slot, atomically:

```cpp
auto &cursor = cursors( tile ).ref();
const SI k = SI( atomic_fetch_add( cursor, TC( 1 ) ) );
args.outputs.touching( row_of<D>( at, nb_tiles ), k ) = i;
```

Two details in the Python above are worth noticing, beyond the point they make:

- **`FfiCode.inline`** takes the body of the handler. A kernel is written *where it is launched*, and
  the one line it contains says which C++ it runs. The heavy code stays in `splats.h`, which can be
  compiled and tested without loom at all.
- **A call hands back what it wrote**, so a pass is one statement. And when a kernel writes only
  *part* of an aggregate, the restriction is named from whichever end is shorter: `writes = ( "values", )`
  or `reads = ( … )`. Both lists are exclusive, and checked: a name that designates nothing is
  refused.

This pass is **not differentiable**: it returns integers, and its link to the centres is
discontinuous (a Gaussian either enters a tile or it does not).

→ [Sizes known at run time](/guide/dynamic-sizes) · [Roles of the arguments](/guide/arguments)

## Pass 2: the image

```python
def render( splats, touching, screen ):
    side = int( screen.tile_side )
    return loom.ffi_call(
        "splats_render",
        loom.FfiCode.inline( f"splats::render< { side } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        loom.FfiCode.inline( f"splats::render_bwd< { side } >( queue, batch_axes, args );",
                             includes = [ "splats.h" ] ),
        splats = splats,
        touching = touching,
        image = loom.out( loom.RealTensor[ screen.num_pixel, splats.num_channel ]() ),
    ).value
```

Two kernels, and the second is the adjoint: a kernel in its own right, running on other buffers with
its own launch domain. The call takes both and names the pair. `.value` gives the backend array, which
is what `jax.vjp` differentiates through and what `numpy.asarray` reads.

The forward kernel is one item per pixel, and a loop over its tile's list:

```cpp
SI x[ D ];
const SI row = pixel_and_row<D,SIDE>( coords, batch_axes, args.outputs.image, x );

TF acc[ C ] = {};
for ( SI k = 0; k < args.inputs.touching.row_size( row ); ++k )
    contribution<D>( args.inputs.splats, SI( args.inputs.touching( row, k ) ), x, C, acc );
```

and its launch leaves the channel axis *out* of the domain:

```cpp
queue.run_parallel( RenderPixel<D,SIDE>{},
                    batch_axes + args.outputs.image.domain( args.outputs.image.axes - num_channel ),
                    args, batch_axes );
```

One item per **pixel**, not per image scalar: otherwise each work-item would recompute the same
Gaussian once per colour channel.

### Nothing is flat, and nothing is named `x`

No flat rank, no division by an image side, no axis called `x` or `y`. Every kernel asks for the
domain it wants, and reads its coordinates by name. The three lines in `pixel_and_row` are the whole
bridge between axes, which exist only at compile time, and geometry written as
`for ( d = 0; d < D; ++d )`, which is how people actually write it:

```cpp
const auto main_axes = coords.axes - batch_axes;        // ours; a vmap's axes are not
SI x[ D ];
for_each_indexed( main_axes, [&]( auto axis, int d ) { x[ d ] = SI( coords[ axis ] ); } );
```

Not one line of `splats.h` names a dimension, save one, which says so: `half_box`, the half-extent of
the footprint, is the closed-form 2-D inverse of a 2×2 matrix. In 3-D it is a Cholesky, and it is the
single function to replace.

## The adjoint, in atomic accumulation

In [Diffusion](./diffusion) the adjoint was a pure gather: each cell read its neighbours and wrote only
its own value. Here it cannot be. A Gaussian is hit by **every pixel it covers**, so by different
work-items, and each adds its share to the same four gradients:

```cpp
// with  L the loss,  g the pixel's cotangent,  gc = <g, color>
//   dL/dcolor_i,c = w g_c                        w = opacity ( e - E )
//   dL/dopacity_i = ( e - E ) gc
//   dL/dcenter_ia = opacity e gc  Σ_b A_ab dx_b
//   dL/dA_i,ab    = opacity e gc  ( -½ dx_a dx_b )     doubled off the diagonal (packed storage)

if constexpr ( grad.opacities.is_valid )
    atomic_add( grad.opacities( i ).ref(), ( e - threshold<TF>() ) * gc );
```

Each is guarded by `is_valid`, so a gradient nobody asked for costs nothing. loom takes care of the
rest: the gradient buffers shared between work-items are **seeded with zero** before the kernel runs,
which is what an accumulation needs and what a plain output does not guarantee.

### A lesson about the model, not about loom

The first version cut the footprint with a hard threshold. Its adjoint was exact to eleven digits on
`colors` and `opacities`, and wrong by about 5 % on `centers` and `cov_inv`. That pattern *was* the
diagnosis. The exact ones are those that do not enter the exponential; the wrong ones are those that
**move the truncation boundary**, where the weight jumped from `opacity · 1.1·10⁻²` to zero. A
finite difference measures a jump divided by `2ε`, and a gradient descent takes noise at every step.

The fix is in the model: the truncation is made **continuous** by subtracting the value *at* the
threshold instead of cutting. The `E` in the formula above is that constant. It does not depend on
`q`, so it vanishes on differentiating, and the geometric gradient goes through `e` alone.

## Checking it

`check_grad` is run on each of the four families of parameters:

```text
d/dcolors     : adjoint +8065.115666425   finite diff. +8065.115666425
d/dopacities  : adjoint +7160.953800026   finite diff. +7160.953800024
d/dcenters    : adjoint  +103.227867923   finite diff.  +103.227709393
d/dcov_inv    : adjoint -1002485.748568417   finite diff. -1002752.904557065
```

`colors` and `opacities` agree to the last digits. `centers` agree to about five digits and
`cov_inv` to about four: that is the limit of a finite difference near a footprint's edge, not of the
adjoint. The check passes with the default tolerances.

The rendering itself is compared with **the sum over every Gaussian, with no lists at all**
(`the_rendering_matches_the_direct_sum`): the gap is `7·10⁻¹⁵`. If a single Gaussian had been lost by
the index the gap would show, which makes it a test of the structure and not of the stencil alone.

## Fitting an image

The animation at the top is ordinary Jax around **one loom call**. Each Gaussian starts at a random
position with the colour of the target under its centre (the usual initialisation); where they go, how
big they get and how opaque they are is for the descent to find.

The parameters are turned into the
packed inverse covariance by plain `jnp`, `render_scene` renders them, and `jax.grad` goes through
both:

```python
def cov_inv_of( log_sigma, angle ):
    """Packed upper triangle ( a, b, c ) of R diag( 1/s0², 1/s1² ) Rᵀ."""
    i0, i1 = jnp.exp( -2 * log_sigma[ :, 0 ] ), jnp.exp( -2 * log_sigma[ :, 1 ] )
    ca, sa = jnp.cos( angle ), jnp.sin( angle )
    return jnp.stack( [ ca * ca * i0 + sa * sa * i1,
                        ca * sa * ( i0 - i1 ),
                        sa * sa * i0 + ca * ca * i1 ], axis = 1 )

def image_of( p, shape ):
    s = Splats( nb_dims = 2, nb_channels = 3 )
    s.centers, s.colors, s.opacities = p[ "centers" ], p[ "colors" ], p[ "opacities" ]
    s.cov_inv = cov_inv_of( p[ "log_sigma" ], p[ "angle" ] )
    return render_scene( s, shape )

loss = lambda p: jnp.mean( ( image_of( p, shape ) - target ) ** 2 )
value, grads = jax.value_and_grad( loss )( params )     # then any optimiser: here Adam
```

Parametrising by log-scale and angle keeps the covariance positive definite, which a descent on
`cov_inv` directly would not. That is a modelling choice made in plain Jax, outside loom: the loom
call only sees the packed matrix.

`render_scene` builds the lists on centres whose gradient is **cut** (`loom.stop_gradient` on the
aggregate: which tile a Gaussian lands in is discontinuous, and that is not where the derivative
goes), then renders. Note that this loop runs *eagerly*: building the lists reads a total on the host
between two calls. Doing both in one handler, so that it also works under `jit`, is what
`tests/test_scratch_gpu.py` exercises.

## What a bound would have cost

The test `what_a_single_bound_costs` prices the alternative, honestly. XLA can build these lists if
the worst case is bounded, and the *right* bounded implementation is a fixed-size window per
Gaussian, scattered with an add. Its radius has to cover the **largest** Gaussian of the scene, and
everyone pays for it. In a scene whose sizes span a decade:

| scene | useful (Gaussian, pixel) pairs | fixed window | dense sum |
|---|---|---|---|
| 500 splats, 256×256 | 2 513 237 | ×6.5 | ×13.0 |
| 2000 splats, 256×256 | 9 749 623 | ×6.7 | ×13.4 |
| 2000 splats, 512×512 | 9 749 623 | ×6.7 | ×53.8 |

*The multipliers are work relative to the real footprints, which is what loom's lists amount to.*

And why a CSR rather than a padded rectangle `tiles × max_per_tile`, which is the obvious
alternative? The padded storage, **priced at its very best** (capacity exactly equal to the longest
row, which a real one could not even know ahead of time), costs more memory for the same images:

| scene | padded (ints) | CSR (ints) | ratio |
|---|---|---|---|
| 500 splats, 256×256 | 23 552 | 9 883 | ×2.38 |
| 2000 splats, 256×256 | 97 536 | 37 626 | ×2.59 |
| 2000 splats, 512×512 | 219 136 | 43 148 | ×5.08 |

That decision was made on a measurement, not a taste: the example used to carry both
representations, and the padded one was dropped. The ratio is still checked on every run
(`a_padded_storage_would_cost_more_memory`) and needs no second implementation to stay alive: what a
rectangle would cost is arithmetic on the row lengths.

### Where the time goes

`errand -k bench test_splats`, on 2000 Gaussians over a 512×512 image (42 123 (Gaussian, tile) pairs
over 1024 tiles):

| | |
|---|---|
| building the lists, 2 passes plus a host prefix | 3.3 ms |
| rendering, one pass over the pixels | 14.9 ms |

*One run on an Apple-silicon laptop, CPU; your numbers will differ and so does their split. On the
machine the example was written on, the two were about equal.* The CSR's own trade shows up in the
open: it buys exactness with a second pass and a host round trip, so building the lists is a
significant fraction of the whole. The padded storage was cheaper to build and no faster to render,
which is why this was a decision about memory and not about speed.

## What to remember

| | |
|---|---|
| **The host can read a count a kernel wrote.** | `CsrTensor.from_counts` is count, prefix sum on the device, one number read back, exact allocation. |
| **A call is a pipeline.** | Several `ffi_call`s sharing aggregates; each hands back what it wrote. |
| **Atomic adjoints exist.** | A gradient shared by several work-items is accumulated with `atomic_add`, and its buffer is zero-seeded by loom. |
| **Make the model continuous.** | A hard cut puts impulses in the derivative. If the adjoint is exact on some parameters and wrong on others, look at which ones move a boundary. |
| **Axes are declared once.** | Computed extents, `AxisList`, `domain( … )`: no flat rank, no `x`, no `y`. |
| **Ordinary Jax around one loom call.** | The parametrisation, the loss and the optimiser are yours. |

## Where next

- [Sizes known at run time](/guide/dynamic-sizes) and [The adjoint](/guide/adjoint), for the details
  this tutorial skipped.
- [Diffusion](./diffusion), if you came straight here and want the dimension-agnostic body first.
