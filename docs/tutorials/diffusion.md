# 1 · Diffusion

A differentiable heat-equation solver, written as if its author had never heard of loom's other users:
a Cartesian grid, a five-point stencil, nothing else. By the end you will have

- a **C++ kernel** that works on a 2-D grid, a 3-D grid, and under `vmap`, without being edited;
- a **Python function** that Jax, Torch or numpy can call;
- a **hand-written adjoint**, checked against finite differences;
- an **inversion** that runs gradient descent *through* the solver.

The whole thing is one file, [`examples/diffusion/diffusion.py`](https://github.com/hleclerc/loom/blob/main/examples/diffusion/diffusion.py),
with the C++ inside it in plain sight. Every code block below is taken from it.

```bash
cd examples/diffusion
python diffusion.py          # compiles two small kernels, then prints three lines
```

→ If it does not run, [Installing](/guide/installing) says why.

## The problem

Heat diffuses: `du/dt = k Δu`. With an explicit time step, a constant `k`, and the temperature
imposed on the boundary, one step on a grid is

```text
u'( a ) = u( a ) + c · Σ_{b neighbour of a} ( u( b ) − u( a ) )          c = dt · k / h²
```

Each cell moves towards the mean of its neighbours. The boundary cells do not move at all. `c` must
stay under `1/(2D)` (0.25 in 2-D) or the scheme blows up.

In Jax or Torch you would write this with slices (`u[1:-1, 1:-1] + c * (u[2:, 1:-1] + …)`), and
it would be fine. It is a good first kernel precisely because the loop version is *also* easy, so
what is new is loom, not the numerics.

## The kernel

```cpp
namespace {
    struct OneStep {
        /// the boundary carries an imposed temperature: it is never updated.
        HD bool on_border( auto coords, auto main_axes, const auto &args ) const {
            return any_of( main_axes, [&]( auto axis ) {
                return coords[ axis ] == 0
                    || coords[ axis ] + 1 == args.inputs.temperature.size( axis );
            } );
        }

        HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
            const auto main_axes = coords.axes - batch_axes;

            const TF uc = args.inputs.temperature( coords );
            if ( on_border( coords, main_axes, args ) ) {
                args.outputs.temperature( coords ) = uc;
                return;
            }

            TF total = 0;
            for_each( main_axes, [&]( auto axis ) {
                total += args.inputs.temperature( coords + axis )
                       + args.inputs.temperature( coords - axis ) - 2 * uc;
            } );
            args.outputs.temperature( coords ) = uc + args.inputs.coef * total;
        }
    };

    void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
        queue.run_parallel( OneStep(), args.outputs.temperature.domain(), args, batch_axes );
    }
}
```

This is plain C++. `HD` marks a function that may run on host or device, so the same text compiles for
a CPU and for CUDA. Read it from the bottom.

### What loom calls

`kernel( queue, batch_axes, args )` is the one entry point loom looks for. It receives three things:

| | |
|---|---|
| `queue` | The execution context. `queue.run_parallel( functor, domain, args, batch_axes )` runs the functor once per point of the domain, in parallel. It is a convenience, not an obligation: if you prefer OpenMP or Kokkos, ignore it and take `queue.stream` and the pointers inside `args`. |
| `batch_axes` | The axes that the *call* added, for instance by a `vmap`. It is a value: you subtract it from something, or add it to something. |
| `args` | Your arguments, under their Python names, in groups: `args.inputs`, `args.outputs`, and (for the adjoint) `args.grad_of_inputs`, `args.grad_of_outputs`. `TF` is the call's real scalar type (`float` or `double`). |

The `namespace { }` is yours. It keeps your `#include`s wherever you want them, and gives the
functor internal linkage, which matters because several kernels can end up linked into one library.

### The launch

```cpp
queue.run_parallel( OneStep(), args.outputs.temperature.domain(), args, batch_axes );
```

You launch the loop yourself, *on the domain of the data*: one item per cell of the output. The item
the functor receives, `coords`, is a multi-index. There is no flat rank to decode with `/` and `%`.

### Three primitives, and everything follows

The body never says "two dimensions". It uses three things:

| | |
|---|---|
| `coords[ axis ]` | the coordinate **by name**, not by position. |
| `coords + axis` | the neighbour along **that** axis. The other coordinates do not move, batch coordinates included. |
| `coords.axes - batch_axes` | *my* axes: those of the grid, and not those a `vmap` added. A set subtraction, done at compile time. |

Then `for_each( main_axes, … )` walks those axes, unrolled at compile time, and the lambda is called
once per axis. In 2-D it is two neighbour pairs; in 3-D, three. The same text is a five-point
stencil in 2-D and a seven-point one in 3-D.

Note what is *not* written: no `TF( … )` conversion. A rank-0 tensor (`args.inputs.coef`)
converts to a scalar by itself, so the body reads like the formula.

## The call

```python
import loom

def step( temperature, coef ):
    return loom.ffi_call( "diffusion_step", _forward, _backward,
        temperature = loom.mutable( temperature ),
        coef = coef,
    )

def evolve( temperature, coef, nb_steps ):
    for _ in range( nb_steps ):
        temperature = step( temperature, coef )
    return temperature
```

`_forward` and `_backward` are the two [`loom.FfiCode`](/reference/python-api) objects that hold the
C++ of the kernel and of its adjoint (the second one is the next section). `"diffusion_step"` names
the call: it prefixes the compiled targets and groups the entries of the compilation journal.

**Nothing to wrap.** `temperature` is a Jax array, a Torch tensor or a numpy array, as it is. loom
reads its shape and its element type; the axes are deduced and named by position, so you can write
a stencil without declaring one.

**`loom.mutable( temperature )`** says the only thing left to say: this grid is read, then
rewritten. XLA does not allow a kernel to write what it reads, so loom makes two buffers and rebinds
the name on the Python side. The kernel sees both:

```text
args.inputs.temperature      what goes in
args.outputs.temperature     what comes out
```

and the adjoint sees `args.grad_of_outputs.temperature` (coming in) and
`args.grad_of_inputs.temperature` (going out). What the call returns is what it wrote, of the same
kind as what it was given: a Jax array went in, a Jax array comes out.

→ [Roles of the arguments](/guide/arguments)

## Running it

`step` is a function of arrays. So everything you can do with a function of arrays works:

::: code-group

```python [Jax]
import jax, jax.numpy as jnp
from diffusion import step, evolve

u = jnp.zeros( ( 9, 9 ) ).at[ 4, 4 ].set( 1.0 )

step( u, 0.2 )                                       # eager
jax.jit( lambda u: step( u, 0.2 ) )( u )             # compiled with the rest of the program
jax.vmap( lambda a: step( a, 0.1 ) )( jnp.stack( [ u, u ] ) )     # a batch of grids -> ( 2, 9, 9 )

u3 = jnp.zeros( ( 6, 6, 6 ) ).at[ 3, 3, 3 ].set( 1.0 )
step( u3, 0.1 )                                      # the same kernel, in 3-D
```

```python [Torch]
import torch
from diffusion import step, evolve

u = torch.zeros( 9, 9, dtype = torch.float64 )
u[ 4, 4 ] = 1.0

step( u, 0.2 )                                       # a torch.Tensor back
```

```python [numpy]
import numpy as np
from diffusion import step

u = np.zeros( ( 9, 9 ) ); u[ 4, 4 ] = 1.0
step( u, 0.2 )                                       # a numpy array back
```

:::

All three were run on CPU, and the 2-D, 3-D, `jit` and `vmap` lines on Jax. The kernel was compiled
the first time each signature was seen (a 2-D grid, a 3-D grid, a batched grid are three
signatures), and every later call, in this process or another one, finds it in the cache.

::: tip The batch adds nothing to the kernel
`vmap` gives the call an extra axis. The body still walks only `coords.axes - batch_axes`, so
the same code ran on a `( 2, 9, 9 )` batch without knowing the batch exists. The test
`the_same_body_batches_without_knowing_it` compares a vmapped call with one call per grid, by
hand: the difference is exactly `0.0`.
:::

## The adjoint

To differentiate through `step`, loom needs its adjoint: given the gradient of the loss with
respect to the output, produce the gradient with respect to the input. In loom it is **a kernel
like any other**, written in the same C++ and given to the same `ffi_call` as its second argument.

What does the adjoint of one step look like? Look at who `u( a )` contributes to:

- to the output at `a` itself: `1` in the identity term, minus `2cD` if `a` is interior (`D`
  being the number of axes);
- to the output at each **interior** neighbour `b`: `c`.

So the gradient at `a` is a sum over `a` and its neighbours of the output gradient at those
points: one read of the neighbours and a single write. It is a pure **gather**, which means no
atomic accumulation.

```cpp
struct OneStepAdjoint {
    HD bool on_border( auto coords, auto main_axes, const auto &args ) const { /* as above */ }

    /// true if `coords` shifted by `d` along `axis` is still inside the grid
    HD bool inside( auto coords, auto axis, SI d, const auto &args ) const {
        const SI c = coords[ axis ] + d;
        return c >= 0 && c < args.inputs.temperature.size( axis );
    }

    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto main_axes = coords.axes - batch_axes;

        static_assert( ! args.grad_of_inputs.coef.is_valid,
            "diffusion: the gradient with respect to dt k / h^2 is not implemented" );

        constexpr bool is_null = args.grad_of_outputs.temperature.surely_null;

        TF res = 0;
        if constexpr ( ! is_null ) {
            constexpr SI D = DECAYED_TYPE_OF( main_axes )::ct_size;
            const TF c = args.inputs.coef;
            const TF g = args.grad_of_outputs.temperature( coords );

            res = on_border( coords, main_axes, args ) ? g : g * ( 1 - c * ( 2 * D ) );

            for_each( main_axes, [&]( auto axis ) {
                for ( SI s = -1; s <= 1; s += 2 ) {
                    if ( ! inside( coords, axis, s, args ) )
                        continue;
                    const auto neighbor = s > 0 ? coords + axis : coords - axis;
                    if ( ! on_border( neighbor, main_axes, args ) )
                        res += c * args.grad_of_outputs.temperature( neighbor );
                }
            } );
        }

        if constexpr ( args.grad_of_inputs.temperature.is_valid )
            args.grad_of_inputs.temperature( coords ) = res;
    }
};
```

Three things in there are about *the contract with the framework*, and they are the ones a
hand-written binding gets wrong:

**`surely_null`.** The framework may hand the backward a *symbolic zero*: the cotangent of an
output nothing downstream used. It is never materialised, so there is nothing to read. And an
output buffer is **not** guaranteed to be zero, so if the cotangent is a symbolic zero the kernel
must still *write* a zero. That is why `res` starts at 0 and the write is outside the
`if constexpr`.

**`is_valid`.** The framework may want the gradient with respect to some inputs and not others. If
`grad_of_inputs.temperature` is not wanted, the write compiles away. Nothing to declare: loom tells
the kernel, at compile time.

**The `static_assert`.** The gradient with respect to `coef` would need a global reduction, so it
is not written. Asking for it is a *compile-time error that says so*, instead of a silent zero.

→ [The adjoint](/guide/adjoint)

## Checking it

A wrong adjoint does not crash. It makes training a little worse. So loom ships a checker:
`loom.testing.check_grad` compares the adjoint with a centred finite difference, on random
projections, without building the Jacobian. It is tested across **a chain** of steps, because the
gradient has to travel back up it:

```python
from loom.testing import check_grad

n, coef = 9, 0.18
u0 = bump( n )                                    # a Gaussian bump, zero on the border
ad, df = check_grad( lambda u: evolve( u, coef, 3 ), u0, seed = 11 )
# d/du  : adjoint +18.301381039   finite diff. +18.301381039
```

The two agree to nine digits. The test suite also checks that the eigenmode `sin( πx ) sin( πy )`,
an exact eigenvector of the stencil, decays by the factor computed by hand, and that the border does
not move.

```bash
errand test_diffusion          # five tests, all passing
```

## Using it: going back in time

What did we make the solver differentiable *for*? An inversion. We observe the temperature after
`nb_steps` steps and recover the *initial* state, by gradient descent through the whole chain. With
Jax, `jit` and `grad` apply to `evolve` like to any other function:

```python
n, nb_steps, coef = 14, 6, 0.2

# `true_state`: a Gaussian bump on an n×n grid, zero on the border
observed = evolve( true_state, coef, nb_steps )

def loss( u ):
    diff = evolve( u, coef, nb_steps ) - observed
    return ( diff * diff ).sum()

loss_jit = jax.jit( loss )
gradient = jax.jit( jax.grad( loss ) )

u = jnp.zeros( ( n, n ) )
for _ in range( 400 ):
    u = u - 0.4 * gradient( u )

# loss 6.969e+00 -> 3.823e-06      ( a factor 1.8 million )
```

![Gradient descent through the solver: the estimate of the initial state, starting from zero, converges towards the true initial state while the loss falls over 600 iterations](/anim/diffusion_inversion.gif)

*A 40×40 grid, 25 diffusion steps, 600 iterations of gradient descent from an all-zero guess. Left to right: the temperature observed after the steps (the only thing the optimiser sees), the current estimate of the initial state, the true initial state, the loss. The script is [`docs/scripts/diffusion_inversion.py`](https://github.com/hleclerc/loom/blob/main/docs/scripts/diffusion_inversion.py).*

Look at the thin vertical ridge on the right-hand panel, which is in the true state and **not** in the
estimate: that is the second remark below, seen with the eyes.

Two remarks, both about the problem and not about loom:

- **The step size is not a setting to tune.** `evolve` is contracting (diffusion only smooths), so
  the Hessian of the loss has eigenvalues at most 2, and any step beyond about 0.5 diverges. It is
  also why the inversion is slow.
- **The loss is what is asserted, not `u`.** Going back in time is ill-posed: diffusion erases
  the high frequencies and nothing can restore them. Without regularisation you recover a state that
  *explains the data*, not the true state. What the test checks is that the gradient does go through
  the six calls.

The same `step` and `evolve` give the same gradient under Torch: `loss.backward()` goes through
`torch.autograd.Function`, and the gradient at the centre of a 9×9 grid is the same number
(`0.128128`) as with Jax.

## What to remember

| | |
|---|---|
| **The kernel is one functor** and one `kernel( queue, batch_axes, args )`. | You launch the loop on the domain you want. |
| **Axes by name.** | `coords[ axis ]`, `coords ± axis`, `coords.axes - batch_axes`. Dimension and batch are not the kernel's business. |
| **`loom.mutable`** | The way to say "read then rewrite". The kernel sees an input and an output buffer. |
| **The adjoint is a kernel.** | Handle `surely_null` and `is_valid`; assert what you do not implement. |
| **`check_grad`** | Always. A wrong adjoint is silent. |
| **Plain arrays in, plain arrays out.** | Whatever the framework applies to a function of arrays applies to the call. |

## Where next

- [Gaussian splatting](./splats): a structure whose size depends on the data.
- [Axes and coordinates](/guide/axes) and [Roles of the arguments](/guide/arguments), for the
  details this tutorial skipped.
