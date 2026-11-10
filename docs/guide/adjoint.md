# The adjoint

A call is differentiable when it gets a second kernel, the **adjoint**: given the gradient of what
the forward wrote, it computes the gradient of what the forward read. loom does not derive it: you
write it, as a kernel like any other. In exchange, it runs at the speed of the forward, with the
parallelism you choose.

## On one example

`take( x, idx )` picks `y[ k ] = x[ idx[ k ] ]`. The forward reads `x`; the adjoint sends each
gradient back where its value came from:

```python
def take( x, idx ):
    num  = loom.Axis( loom.ShapeVar(), name = "num" )
    pick = loom.Axis( loom.ShapeVar(), name = "pick" )
    return loom.ffi_call( "take", _forward, _backward,       # the adjoint comes second
        x   = loom.RealTensor[ num ]( x ),
        idx = loom.IntTensor[ pick ]( idx ),
        y   = loom.out( loom.RealTensor[ pick ]() ),
    )
```

```cpp
// _forward: one item per pick. `x( coords, num = k )` is `x` at the point, with `num` set to `k`
args.outputs.y( coords ) = args.inputs.x( coords, num = args.inputs.idx( coords ) );
```

```cpp
// _backward: the same items, the other way
struct TakeBwd {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        // several picks may read the same `x`: their gradients add up
        atomic_add( args.grad_of_inputs.x( coords, num = args.inputs.idx( coords ) ).ref(),
                    TF( args.grad_of_outputs.y( coords ) ) );
    }
};

void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
    args.computed_grads.x_from_y = true;     // what this adjoint computes: d y / d x
    if constexpr ( ! DECAYED_TYPE_OF( args.grad_of_outputs.y )::surely_null )
        queue.run_parallel( TakeBwd(), batch_axes + args.inputs.idx.domain(), args, batch_axes );
}
```

Both kernels pass the point to `x`, and launch over `batch_axes + idx.domain()`: under a batch,
`x` has one more axis, which `idx` does not have (see [Axes and coordinates](./axes)).

Then `jax.grad`, `torch.autograd` and the rest work through `take`:

```python
jax.grad( lambda x: ( take( x, idx ) * w ).sum() )( x )

x.requires_grad_()
( take( x, idx ) * w ).sum().backward()      # x.grad
```

## In Python: giving the adjoint

`loom.ffi_call( name, forward, backward, ... )`: the adjoint is the second kernel, and it is
optional. Both are `FfiCode`s, compiled separately, and each chooses its own launch.

**Where it works.** Jax (`jax.grad`, `jax.vjp`, under `jax.jit` and `jax.vmap` too) and Torch
(`.backward()`, `torch.autograd.grad`, `torch.func`). NumPy and CuPy have no differentiation: there,
only the forward runs. `loom.grad` and `loom.vjp` call those of the framework in use, for code that
does not choose one.

**What is differentiated.** The real tensors among the inputs, and among the written arguments
(`loom.out`, `loom.mutable`). An integer tensor (`idx`) has no gradient, and an input the caller does
not differentiate with respect to has none either: the adjoint is told, at compile time.

**Checking it.** A wrong adjoint does not fail: it gives a plausible gradient, and the optimisation
that uses it goes on, more slowly or somewhere else. `loom.testing.check_grad` compares the adjoint
with a finite difference, on random directions:

```python
from loom.testing import check_grad

check_grad( lambda x: take( x, idx ), x, seed = 0 )     # raises AssertionError when they differ
```

It works under Jax and Torch alike. Pass a `seed`, so that the directions do not depend on what
ran before.

## In C++: writing the adjoint

### What the adjoint sees

The adjoint is called like the forward, `kernel( queue, batch_axes, args )`, with the arguments of
the forward call **under the same names**, plus their gradients:

| | |
|---|---|
| `args.inputs.<name>` | the inputs of the forward, as they were |
| `args.outputs.<name>` | what the forward wrote, to be read |
| `args.grad_of_outputs.<name>` | the gradient that comes in, one per written argument |
| `args.grad_of_inputs.<name>` | the gradient to write, one per input |

The group names keep the roles **of the forward**: the adjoint reads `grad_of_outputs` and writes
`grad_of_inputs`. A `loom.mutable( u )` is in both, as `args.grad_of_outputs.u` (the gradient of the
new value) and `args.grad_of_inputs.u` (the gradient of the old one).

### What is not asked for is known at compile time

- `DECAYED_TYPE_OF( args.grad_of_outputs.y )::surely_null`: the gradient coming in is a
  **symbolic zero** (the result does not depend on `y`). It is another type, with no buffer: nothing
  to read, and usually nothing to launch.
- `args.grad_of_inputs.x.is_valid`: `false` when the gradient of `x` is not asked for (an integer, or
  an input nobody differentiates with respect to). Writing it would not compile; skip it:

```cpp
if constexpr ( args.grad_of_inputs.x.is_valid )
    args.grad_of_inputs.x( coords ) = ...;
```

Each case compiles its own variant, so these tests cost nothing at run time.

### `computed_grads`: what the adjoint computes

An adjoint says which derivatives it computes, one flag per pair (written argument, input):
`args.computed_grads.x_from_y = true`, or `args.computed_grads.from_y.x` for a member of an
aggregate. After the adjoint, loom checks that each derivative the differentiation needs was set,
and fails otherwise:

```
loom.ffi_call( 'take' ): the backward does not say it computes these derivatives, which this
differentiation needs -- the gradient would miss their terms.
    d y / d x  ( computed_grads.from_y.x )
```

A derivative left out is the most common way of writing a wrong gradient; this turns it into an
error.

### Gather or accumulate

Each gradient value is the sum of the contributions of every item that read it. The adjoint gets
it in one of two ways:

- **Gather**: each item of the adjoint computes one gradient value, reading all the contributions
  to it. One write, no conflict. The adjoint of the [Diffusion](/tutorials/diffusion) step does
  that: a point of the grid collects from its neighbours.
- **Accumulate**: each item of the adjoint pushes its contributions where they go, with
  `atomic_add`, as `take` does. It follows the forward exactly, at the cost of atomic writes. The
  adjoint of [Gaussian splatting](/tutorials/splats) does that: every pixel adds to the splats that
  cover it.

Accumulating needs a gradient that starts at zero: loom fills the outputs with zeros before the
call (see `LOOM_ZERO_OUTPUTS` in the [environment variables](/reference/environment)).

## Next

[Sizes known at run time](./dynamic-sizes) · [Roles of the arguments](./arguments) · [Diffusion](/tutorials/diffusion)
