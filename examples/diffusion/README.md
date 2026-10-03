# `diffusion` — a deliberately foreign user of loom

A differentiable diffusion solver, written as if its author had never heard of optimal
transport. It only imports `loom`, its C++ only knows `<loom/support/...>`, and nothing
it does resembles a Laguerre cell: Cartesian grid, five-point stencil, no ragged,
no geometry, no scratch.

This is not a demonstration: it is a **generality test**. The question asked is "what
in loom is general, and what is merely sdot in disguise?". The assessment is at the bottom.

**A single file**, `diffusion.py`: the kernel's C++ is inside it, in plain sight. You read the tutorial without
navigating.

    du/dt = k Δu, explicit time step, temperature imposed at the boundary, CONSTANT k

    u'( a ) = u( a ) + c · Σ_{b neighbour} ( u( b ) − u( a ) ),   c = dt·k / h²

## What the axes buy

The body **never** counts dimensions:

```cpp
HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
    const auto main_axes = coords.axes - batch_axes;          // my axes, not the vmap's

    const TF uc = args.temperature( coords );
    if ( on_border( coords, main_axes, args ) ) { args.next( coords ) = uc; return; }

    TF total = 0;
    for_each( main_axes, [&]( auto axis ) {                   // unrolled at compile time
        total += args.temperature( coords + axis )
               + args.temperature( coords - axis ) - 2 * uc;
    } );
    args.next( coords ) = uc + args.coef * total;
}
```

Three primitives, and everything follows from them:

| | |
|---|---|
| `coords[ axis ]` | the coordinate **by name**, not by position. |
| `coords ± axis` | the neighbour along **that** axis; the other coordinates do not move — including those of the batch. This is what makes the stencil writable once. |
| `coords.axes - batch_axes` | my own axes. A **set subtraction**, done at compile time. |

Consequence, and it is tested (`le_meme_corps_se_batche_sans_le_savoir`): **a `vmap` adds an axis
without the body changing, and without it knowing that it exists** — a difference of exactly `0.0` against the
hand-written loop. The same body works in 2-D, in 3-D, batched or not.

Note what is **not** written: no `TF(...)` conversion. A rank-0 tensor converts
to a scalar by itself, so `args.temperature( coords + axis ) - uc` reads like the formula.

Loom calls `void kernel( auto &&queue, auto &&batch_axes, auto &&args )` and only writes what is
painful — the FFI wrapping, the binding of buffers, the adjoint on the Jax side:

| | |
|---|---|
| `queue` | the execution context. `queue.run_parallel` is **its** tool, not an obligation: a Kokkos or OpenMP user ignores it and takes `queue.stream` plus the pointers and shapes of `args`. |
| `batch_axes` | the axes that the call added. A **value**: you subtract it, or compose it (`+` does the **union**, not the concatenation — a batched tensor already carries the axis). |
| `args` | our arguments under their Python names, plus `machine` ([`Machine.h`](../../include/loom/support/kernels/Machine.h)), `errors`, and `scratch` if it was requested. `TF` is the call's real scalar. |

The `namespace { }` is written by the user: their `#include`s therefore go wherever they want, and it gives everything
it contains internal linkage — necessary, since several kernels end up linked into the same
library. And `include_roots` is not stated: by default it is the `.py`'s directory.

```bash
errand test_diffusion
```

## What it does

`du/dt = div( k grad u )`, explicit time step, temperature imposed at the boundary:

    u'( a ) = u( a ) + dt/h² · Σ_{b neighbour} K( a, b ) ( u( b ) − u( a ) ),   K = ( k_a + k_b ) / 2

The story is that of an outsider who wants to put their solver in an optimization loop. Here:
**going back in time** — recovering the initial state from the temperature observed after N steps, by
gradient descent through the whole chain. The loss drops by a factor of 1.8·10⁶.

The adjoint is written by hand — a solver's derivative is part of the solver — and it too is
dimension-generic. It is written as a **pure gather** (each cell reads its neighbours and writes only its
own value), hence without atomic accumulation: `u( a )` only enters the output of `a` and of its
neighbours. `check_grad` compares it against finite differences — agreement to 9 digits.

The descent step is not a setting to be found by trial and error: `evolve` **contracts** (diffusion only
smooths), so the loss's Hessian has eigenvalues ≤ 2 and any step beyond ~0.5
diverges. This is also why the inversion is slow, and why it is the **loss** that is asserted
and not `u`: going back in time is ill-posed, diffusion erases the high frequencies and nothing
can restore them.

The only gradient not written is that of `dt/h²`, which would require a global reduction; a
`static_assert` on `grad_for_coef.is_valid()` says so at compile time rather than returning a
silent zero.

## What worked without asking anyone

- **Names carry through.** A call argument arrives under its Python name (`args.temperature`),
  and an axis declared `"y"` is written `y` in the kernel. Nothing to declare twice.
- **The VJP.** The adjoint is plugged in by `driver.grad` without declaring anything, and the
  `NoneTensor` / `ZeroTensor` categories really do make the terms drop out at compile time.
- **The header root** is the `.py`'s directory by default: a third-party package has nothing to
  declare, and loom does not know its users by name.
- **The tensor factories.** `IntTensor[ cell ].iota()`, `RealTensor[ y, x ].ones()`,
  `.linspace( 0, 1, x )`, `.random( seed = 3 )`: the shape comes from the axes, the type comes from the
  class, and `driver` no longer appears anywhere except `grad` / `jit` / `call`.

## The frictions, in the order in which they are met

0. ~~**`compilation.register_include_root( ... )`**~~ **Fixed**: it was a module
   incantation, pronounced before any other import and with no visible relation to the kernel that needs it.
   A kernel knows where its headers are, and says so in the call: `FfiCode( include_roots = [ ... ] )`.

1. **Everything is called `sdot`.** loom's C++ lives in `namespace sdot`; the environment
   variables are `SDOT_BUILD_DIR`, `SDOT_CACHE_DIR`, `SDOT_KERNELS`, `SDOT_EXTERNALS`; the
   cache is `~/.cache/sdot`; the error messages say "sdot:"; the default functor of an unnamed
   `FfiCode` was called `sdot_fwd_kernel` (the name has been mandatory since). You install loom and you get sdot. A
   simple rename — but it is the first thing the outsider sees.

2. ~~**`driver.array( [ 0, 1, 2 ] )` returns floats.**~~ **Fixed**: the documented path is the
   class, which IS the type declaration (`IntTensor[ x ]( [ 0, 1, 2 ] )` cannot go wrong),
   and `driver` has gone back to being the low layer. The friction was going through it.

3. ~~**A call's batch only comes from aggregates.**~~ and ~~**"Who am I?" has no
   answer.**~~ **Dissolved**, both of them — and it is instructive, because neither was the
   problem. Both were symptoms of a single cause: **the launch was implicit.**

   The scaffold always wrote `run_parallel( queue, global_batch_indices, ... )`, and
   `global_batch_indices` is only filled with `vmap` axes. A kernel whose parallelism
   is not a `vmap` axis — a Cartesian grid, traversed in (j, i) — therefore had no
   way of saying so. Hence the cascade: building a flat batch axis of `ny · nx`, **materializing
   an int64 `iota` of n² elements** to carry the rank, wrapping it in a pretext aggregate
   (`Cellules`, whose docstring admitted it served no purpose), and re-splitting `j = p / n,
   i = p % n` in **both** kernels — to recover two coordinates that the data already had.
   On a 128² grid, that meant 131 kB allocated and traversed per call, 40 calls per gradient,
   to compute a Euclidean division.

   The remedy is that **the user launches themselves**, on the data's domain:

   ```cpp
   queue.run_parallel( OneStep(), args.next.axes(), args, batch_axes );
   ```

   `Cellules`, `new_batch_axis`, the `iota`, the rank, the decoding: all gone. And "who
   am I?" no longer arises, because the item IS the multi-index that was asked for — named, hence
   readable as `coords[ y ]` instead of a position to count.

   **Then a second time, further on.** The remedy above still had the functor and
   the launch written *in Python*, as strings. The right answer was simpler: this kernel's C++
   lives in **its own headers**, functors included, and Python says only one line about it. A
   functor there declares its own parameters instead of inheriting them from the kwargs order, and the
   file compiles and is tested **without loom**. The two mechanisms that loom had gained for
   the previous step (`functors = { ... }` and an injected `launch`) became useless the same
   day, and were removed.

   What it costs, and it must be said: a body that launches by itself ignores
   `global_batch_indices`, so **it no longer takes part in `vmap`** on its own. For this example it is
   free (it is not `vmap`ped); in general the two domains would have to be composed.

5. ~~**No `arange`, no `linspace`, no `ones`.**~~ **Fixed**, and on the tensors rather than
   on the driver: `zeros`, `ones`, `full`, `iota`, `linspace`, `random`, which read their shape
   from the AXES — so there is no shape to repeat.

6. **One compilation per grid size** (`CtShapeVar` engraves the extent into the source) — here
   it is intended, the stencil benefits from it, but nothing says so — **and one compilation per
   derivation pattern**. **Measured, then fixed**: `LOOM_JOURNAL=1` now says how many kernels were
   built and *why each one was new*. It immediately showed that the ten-step chain
   compiled **thirty**, of which twenty-eight differed only by the name of the batch axis
   (`cell_0` … `cell_19`) — the axes of a chain are alive at the same time, so each one
   borrowed a different index, and it grew linearly with the length of the chain.
   Batch axes are now named **at lowering** (`batch_0`, `batch_1`, …), so
   two identical calls yield the same source: **30 kernels → 3**, 135 s → 17 s. The two
   backward variants that remain are real variants (`temperature : out -> in`, depending on
   whether the step reads a perturbed input or a constant).

7. **An output buffer is not guaranteed to be zero.** You only learn it from a comment in
   `loom/tests/test_call.py`. An outsider who writes a backward guarded by
   `if constexpr ( surely_null )` and omits the `else` branch returns garbage, silently.

8. **The factories swallowed their keywords.** `Parametrized.__getattr__` poured *every* kwarg
   into the `template_kwargs`: `RealTensor[ x ].random( seed = 7 )` therefore drew a different
   value on each call, silently — and `full( v )` only worked because its argument
   is positional. Fixed: the factory's signature settles it.

9. **The Torch driver does not exist.** This is the heaviest finding, and it was first
   *missed here*: `--driver torch` selects an ENVIRONMENT, not a driver, and that
   environment has jax installed — the first two runs, "jax" and "torch", were
   therefore both `JaxDriver`. Forced by `SDOT_FRAMEWORK=torch`, nothing starts:
   `TorchDriver` has no `available_gpus`, no `array`, no `grad`, no `vjp`, no `vmap`, no `jit` —
   **no `call`**, which is nevertheless THE entry point. What it carries instead
   (`optimize_using_lbfgs`, `to_nanobind_compatible_objects`, `linalg_solve`) dates from another
   design. Today loom is a **Jax** library.

10. **Importing the example requires `sys.path` surgery**: `./run test` imports the test file
   by its path, and the neighbouring module cannot be found. Harmless here, but it is
   the first line of the file.

## What the exercise says about loom

The core — `tensor` + `drivers` + `compilation` — absorbed a frankly foreign use **without
a single line of loom changing** to make it work: the model holds, and that is the best
argument available for saying that loom is not a by-product of sdot.

But half the promise is not kept (9): there is only one driver. As long as the Torch path
does not exist, "a homogeneous interface for whoever has C++ to plug into Jax **or** Torch" cannot
be the tagline — and it is precisely the sentence that would justify an independent life. The other frictions are marginal: a rename (1), constructors on the
caller's side (2, 4, 5, 8 — fixed), a structural assumption (3), documentation (6, 7, 10).
