# `splats` — an index discovered at run time, and what loom buys exactly

Gaussian splatting in additive blending: loom's second **deliberately foreign user**, after
[`diffusion`](../diffusion/). It imports only `loom`, its C++ knows only `<loom/support/...>`, and it
was written to put the ragged machinery to the test.

It did, and the conclusion is not the one we expected. This document says what loom buys *here*,
what it does not, and the numbers for both.

```
splats.py       the data, the calls, and what each pass does
splats.h        the C++ of the four kernels, in the same order
reference.py    what one would write without loom, and what the bound costs
test_splats.py  six tests and a bench
```

```bash
errand test_splats          # the tests
errand -k bench test_splats # the timings ( takes the machine for itself )
```

## The case

    image( p ) = Σ_i  opacity_i · max( exp( −q_i(p)/2 ) − exp( −k²/2 ), 0 ) · color_i
    q_i(p)     = Σ_ab A_ab dx_a dx_b,     dx = p − center_i

Anisotropic Gaussians (`A` is the inverse covariance, symmetric), summed over an image. A pixel must
only look at the splats that reach it, otherwise rendering is O(pixels × splats). Hence an index
`tile → splats`, whose length **depends on the data**: it is not known until one has looked at where
the splats are and how big they are.

That is the whole question. **Who decides the size of that index, and when?**

## What loom buys, and the boundary it does not cross

One single thing, but it exists nowhere else: **the host can read a count a kernel has just written,
and size the next allocation with it**. That is what makes the CSR path *exact* — no bound to guess:

```python
counts = loom.ffi_call(
    "splats_count",
    loom.FfiCode.inline( f"splats::count< { TILE_SIDE } >( queue, batch_axes, args );",
                         includes = [ "splats.h" ] ),
    splats = splats,
    counts = loom.out( loom.IntTensor[ *tile_axes ]() ),
)

touching = loom.CsrTensor.from_counts( counts )   # <- the total is READ on the host,
                                                  #    and `values` sized on it EXACTLY
```

Two things to notice in that snippet, beyond the point it makes:

- `FfiCode.inline` takes the body of the handler, so a kernel is written **where it is launched**
  rather than in a module-level constant far from its call — and the one line it contains says which
  C++ it runs;
- **a call hands back what it wrote**, so a pass is one statement. `loom.out` returns the very object
  it was given (`mutable` returns the new buffer, of the same kind as what it was handed). And when a
  kernel writes only *part* of an aggregate, the restriction is named from whichever end is shorter —
  `writes = ( "values", )` or `reads = ( … )`. Both lists are **exclusive**, and checked: a name that
  designates nothing is refused.

XLA cannot do that. It *can* build a CSR (a count, a prefix sum, a scatter) — it cannot **know its
total**: under `jit` that count is a tracer, and nothing can read it on the host. The list therefore
has to be sized by a bound chosen before tracing: too small, it loses splats silently; safe enough,
it makes everyone pay the worst case.

(loom also has the *other* answer to the same problem — guess a capacity, let the kernel report that
it did not hold, and run again with more room. This example used it until the measurement below
retired its padded storage; `tests/test_call.py::capacity_overflow` is where it lives now.)

**The boundary, as we believed it to be:** both capabilities are *eager-only*. `ShapeArray`
explicitly refuses a tracer, and `capacity_overflows()` returns `None` under `jit` — "its content
only exists at execution time, so no Python loop can look at it and try again". Under `jit` one would
have to prescribe the capacity, and the advantage would only hold for eager code.

> **Correction (2026-09-26).** That boundary is not a property of XLA: it is a property of doing the
> read-back **in Python**. Moved into the handler's C++, it disappears. A handler runs at execution
> time, so it can read a count its own kernel has just written on the card, and allocate exactly on
> it inside XLA's pool — **under `jit` as in eager**. Measured, on GPU, in
> `tests/test_scratch_gpu.py`: a function compiled **once**, allocated sizes of 1989 / 2023 / 2040 /
> 2074 depending on the draw.
>
> What that changes for this example: the index need be neither an output nor a guessed capacity,
> **provided its construction and its consumption fit in a single call**. What remains true of the
> limit above is what must cross the Python boundary: an **output** shape is still fixed at trace
> time.

The rest of what loom brings here is not specific to the ragged part: the adjoint is written by hand,
in atomic accumulation, and passes `check_grad` on all four families of parameters.

## Nothing is flat, and nothing is named `x`

No flat rank, no division by an image side, no `int( … .value )` — and no axis called `x` or `y`.
Every kernel asks for **the domain it wants** and reads its coordinates **by name**:

```cpp
queue.run_parallel( RenderPixel<D,SIDE>{},
                    batch_axes + args.outputs.image.domain( args.outputs.image.axes - num_channel ),
                    args, batch_axes );
...
const auto main_axes = coords.axes - batch_axes;        // ours; a vmap's axes are not
SI x[ D ];
for_each_indexed( main_axes, [&]( auto axis, int d ) { x[ d ] = SI( coords[ axis ] ); } );
```

`domain( … )` on a tensor was what was missing: `domain()` takes **every** axis, which here would
mean one item per image *scalar* — the same Gaussian recomputed once per channel. The workaround was
to build a flat rank and cut it back up with `/` and `%`, i.e. to lose the axes at the very moment
one iterates over them.

And the three lines above are the whole bridge between the axes — which exist only at compile time —
and a geometry written as `for ( d = 0; d < D; ++d )`, which is how one actually writes it. **Not one
line of `splats.h` names a dimension**, save one, which says so:

| | |
|---|---|
| `precision( splats, i, a, b, D )` | `A_ab` out of the packed upper triangle |
| `quadratic_form` | `Σ_ab A_ab dx_a dx_b` |
| `contribution` | the weight and the colour, over `nb_channels` |
| `touched_tiles` | the box of tiles, clipped to the grid |
| `for_each_in_box<D>( lo, hi, f )` | an **odometer**, because `D` is compile-time but the bounds are not |
| **`half_box`** | **2D only** — the half-extent along `k` is `σ·sqrt((A⁻¹)_kk)`, which in nD is a Cholesky. One function to replace. |

Consequences down the chain, every one of them *removing* something:

- **the tile grid stays a grid, with as many axes as the splats have dimensions** — `_tile_axes(
  shape )` is a comprehension, and the C++ reads its extents off the tensor it is about to write
  into (`to_array<D>( counts.shape(), nb_tiles )`). The one place anything is flattened is the CSR's
  `row_of`, and that is not a shortcut: a CSR's rows ARE a sequence, its offsets being a prefix sum
  along one order;
- **no screen object** — width, height and tile counts are read off the tensors
  (`counts.shape()`, `image.size( axis )`). They no longer cross, and above all they are **no longer
  compile-time**: one more resolution recompiles nothing, where a `Ecran` of `CtShapeVar`s
  recompiled every kernel for every image size. Measured on the test suite, which renders at
  256×256, 96×64, 37×53, 48×48 and 512×512: **one** library per kernel, not five each. The only number
  left at compile time is the **tile side**, which is a choice of algorithm — and it lives in
  Python, crossing as a template argument;
- **an axis extent can be computed** — `num_coeff : Axis[ "nb_dims * ( nb_dims + 1 ) / 2" ]`, because
  the inverse covariance is symmetric. 3 in 2D, 6 in 3D, nothing to prescribe and nothing to keep in
  agreement. It is a *formula*, not an affine expression: it is **computed** instead of being
  inverted (see `ShapeVar.set_formula`), and loom admits both side by side.

## The numbers

**What the bound costs XLA**, on clustered scenes whose sizes span a decade — the real regime, fine
detail and blurry background (2000 splats, 512×512):

| ( splat, pixel ) pairs | | |
|---|---|---|
| what is useful — the sum of the footprints | 9 749 623 | — |
| a fixed window per splat, R = 90 **set by the largest** | 65 522 000 | ×6.7 |
| the dense sum, every pixel sees every splat | 524 288 000 | ×53.8 |

**What a PADDED storage would have cost.** The example carried both for a while — a rectangle
`tiles × capacity` next to the CSR — and the comparison is what settled it. Priced *at best* for the
padded one, with its capacity exactly equal to the longest row (which a real padded version cannot
even know in advance):

| | padded, at best | CSR, exact | |
|---|---|---|---|
| 500 splats, 256×256 | 23 552 | 9 883 | ×2.38 |
| 2000 splats, 256×256 | 97 536 | 37 626 | ×2.59 |
| 2000 splats, 512×512 | 219 136 | 43 148 | ×5.08 |

So the padded storage went. The ratio is still **checked on every run**
(`a_padded_storage_would_cost_more_memory`) and needs no second implementation to stay alive: what a
rectangle would cost is arithmetic on the row lengths.

**Where the time goes**, with the machine to itself (2000 splats, 512×512, 42 123 (splat, tile) pairs
over 1024 tiles):

| | |
|---|---|
| building the lists — 2 passes over the splats, plus a host prefix | 7.2 ms |
| rendering — one pass over the pixels | 6.1 ms |

Near fifty-fifty, and that is the CSR's own trade showing up in the clear: it buys exactness with a
second pass and a host round trip, so building the lists costs about what rendering them does. The
padded storage was cheaper to build (one pass, no round trip) and no faster to render — which is why
this was a decision about **memory**, not about speed.

One last figure, counter-intuitive and worth keeping even though the code that produced it is gone:
the cost of the index pass grew linearly with the **allocated** capacity at constant useful work —
4.7 / 5.8 / 8.8 ms for ×1 / ×2 / ×4. It was **not** the zeroing of the buffer (measured by removing
it, the times did not move). It was the **allocation**, which XLA redoes on every call. An
over-generous capacity does not become free by not being written into.

## What the example demolished

Its first version had only one representation — the padded ragged one — and cited the waste of a
"fixed capacity" as its argument. That was shooting itself in the foot: that waste belongs to *our*
representation, not to a limit of XLA. **For this problem, the padded ragged index is the worse of
the two structures**, and it has been deleted; the measurement that says so is above, and
`git log -- examples/splats` is where the code that produced it lives.

And the finding that matters most to loom: the ragged machinery — a `ShapeVar` with `dep_axes`, one
count per cell, capacity-checked on write by `ShapeVarView::set` — was developed for Laguerre cells,
and **sdot does not use it anywhere**. This example was its first real use, and it concluded that a
CSR does better. So after this cleanup **nothing in the repository writes a ragged count**: the
`dep_axes` of a `ShapeVar` are still used to SIZE things (`Image`'s per-dimension `shape`), but the
per-cell counter, its capacity bound and its error slot have no user left. That is an argument for
simplifying loom, and it is information about loom, not about splatting.

## What the exercise returned to loom

- **The "accumulated output" criterion keyed on the wrong axis**: it was conditioned on
  `dtype.floating_point`, because the only known case was a gradient. An **integer** counter
  accumulated by every splat disproved it — without zeroing it returns an indeterminate total that
  then becomes an *allocation size*: `overflow in static extent product:
  dimensions=[2421069375325856419]`. Fixed: the criterion is "shared".
- **Automatic seeding pays for itself**, and the example proved it by crashing without it.
- **`ShapeVarView` has no `reserve()`**: booking a slot atomically in a ragged list is the very
  gesture of a ragged scatter, and it had to be written by hand, exposing four internal details
  (`.view.ref()`, `.max`, `.errors`, `.id`). Moot here now — the CSR books into a plain integer
  cursor — but it is what any future ragged writer will hit.
- **"Who am I?" kept coming back**: the example used to build a decoy aggregate carrying an `iota`,
  for a rank the launch loop already knew. Hence `flat_index`, handed to the body by the scaffold —
  then `domain( axes… )`, which removes the need for a rank altogether: a kernel names the axes it
  walks, and there is nothing left to flatten.
- **No scan**: the prefix sum used to be on the host, in numpy. Hence `Tensor.cumsum`, and
  `CsrTensor.from_counts` which does it on the device — only the **total** comes back to the host,
  because that is what sizes things.
- **The CSR became a base tool**: `loom.CsrTensor`, an aggregate like any other, with `csr( i, j )`
  on the C++ side — the `i` looks the row up in the offsets. `offsets` carries `nb_rows + 1`
  **bounds** rather than counts, so a row's length is a subtraction of neighbours and the last bound
  *is* the total.

## A lesson about the model, not about loom

The first adjoint was wrong, and the **structure** of the errors gave the diagnosis with no
debugging: exact to eleven digits on `colors` and `opacities`, wrong by 5.4 % on `centers` and 0.8 %
on `cov_inv`. The exact ones are those that do not enter the exponential; the wrong ones are those
that **move the truncation boundary**, where the weight jumped from `opacity · 1.1·10⁻²` to zero —
which a finite difference measures as a jump divided by 2ε.

The cure is not to widen the tolerance but to remove the jump: subtract the value at the threshold,
the truncation becomes continuous, and all four become exact again. It was not a test artefact — a
jumping boundary puts noise into every step of a descent.

Not to be missed in the adjoint: the **value** goes through `(e − E)`, the **geometric derivative**
through `e` alone, the subtracted constant not depending on `q`.

## What is next

**Ragged is not a property of the `ShapeVar`, it is a STORAGE.** While both lived here, the example
had to write two rendering loops — `touching.ids( tile, k )` against `touching( row, k )` — for an
identical computation. That is the sign that the representation leaks into the kernel body, when
`loom/tensor/storage.py` already exists for exactly that: "one object per way a value can be backed".
A `Padded` and a `Csr` would be two more variants there, and the body would write
`touching.row( tile )( k )` without knowing which.

The cleanup settled it by DELETION rather than by abstraction, which is the cheaper answer and the
honest one as long as a single storage wins. It also means the acceptance test — **the body must not
change when the storage changes** — now has nothing to run on. Whoever brings a second storage back
(alpha compositing wants an *ordered* list, and may well want a different one) should write it
through `storage.py` rather than beside it.

**And the geometry in nD.** Everything above is already dimension-free; `half_box` is the one
function left, and it needs the diagonal of an inverse precision matrix (a Cholesky in registers,
`nb_dims` being compile-time). After that the example renders a volume without a line changing.

And for the rendering itself, **alpha compositing**: that is the real algorithm, it makes the
*ordered* list necessary (sort by depth, an adjoint walking the list back along the transmittance)
and it would exercise the **cooperative** launch — one work-group per tile, the list in local memory
— which today has a single user in the whole repository.
