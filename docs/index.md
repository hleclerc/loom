---
layout: home

hero:
  name: "loom"
  text: "Your C++ kernel, as a Jax or Torch function"
  tagline: "Write the inner loop once, in plain C++ or CUDA. loom compiles it, hands it the framework's tensors, runs it inside jit, vmap and grad, and gives you the gradient you wrote for it."
  actions:
    - theme: brand
      text: "What loom is"
      link: /guide/what-is-loom
    - theme: alt
      text: "Why it is not trivial"
      link: /guide/why-not-trivial
    - theme: alt
      text: "Tutorials"
      link: /tutorials/
    - theme: alt
      text: "GitHub"
      link: https://github.com/hleclerc/loom

features:
  - icon: 🧵
    title: Two worlds, one call
    details: "Deep-learning tensors are immutable, traced and owned by XLA or Torch. C++ wants pointers, strides and templates. loom is the part in between: one call, and both sides see what they expect."
    link: /guide/why-not-trivial
    linkText: Why that is hard
  - icon: 🧭
    title: A kernel that never counts dimensions
    details: "Coordinates are read by name: coords[axis], coords + axis, coords.axes - batch_axes. The same body runs in 2-D and in 3-D, batched or not, and a vmap adds an axis without the kernel knowing."
    link: /guide/axes
    linkText: Axes
  - icon: 🎭
    title: The role is said on the value
    details: "loom.out( x ), loom.mutable( x ), loom.scratch( x ): what the kernel reads, writes or only borrows is written where the argument is passed, not in a list on the side that rots."
    link: /guide/arguments
    linkText: Roles
  - icon: ↩️
    title: The adjoint is a kernel like any other
    details: "You write the backward in the same C++, loom wires it into jax.custom_vjp or torch.autograd.Function, and loom.testing.check_grad checks it against finite differences."
    link: /guide/adjoint
    linkText: The adjoint
  - icon: 📏
    title: Sizes the host learns at run time
    details: "A kernel counts, the host reads the count, the next buffer is allocated exactly. Under jit, where XLA alone would need a bound chosen before tracing. This is what CsrTensor is made of."
    link: /guide/dynamic-sizes
    linkText: Dynamic sizes
  - icon: 📦
    title: Compiled on demand, or shipped precompiled
    details: "The first call compiles and caches. For a wheel, loom-kernels builds a catalogue ahead of time, so a user with no compiler still gets the kernels."
    link: /guide/compilation
    linkText: Compilation
---

## A kernel, and its gradient, in one screen

One explicit step of the heat equation on a grid. The C++ is the whole kernel; the Python is the whole
binding.

::: code-group

```cpp [the kernel]
struct OneStep {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto main_axes = coords.axes - batch_axes;     // my axes, not the vmap's

        const TF uc = args.inputs.temperature( coords );
        if ( on_border( coords, main_axes, args ) ) {
            args.outputs.temperature( coords ) = uc;
            return;
        }

        TF total = 0;
        for_each( main_axes, [&]( auto axis ) {              // unrolled at compile time
            total += args.inputs.temperature( coords + axis )
                   + args.inputs.temperature( coords - axis ) - 2 * uc;
        } );
        args.outputs.temperature( coords ) = uc + args.inputs.coef * total;
    }
};
```

```python [the call]
import loom

def step( temperature, coef ):
    return loom.ffi_call( "diffusion_step", _forward, _backward,
        temperature = loom.mutable( temperature ),   # read, then rewritten
        coef = coef,
    )

def evolve( temperature, coef, nb_steps ):
    for _ in range( nb_steps ):
        temperature = step( temperature, coef )
    return temperature
```

:::

`temperature` is a Jax array, a Torch tensor or a numpy array, as it is. There is nothing to wrap,
and nothing about 2-D in the kernel. [The full walkthrough](/tutorials/diffusion) writes it line by
line, adjoint included.

## What it is for

Everything you would write a custom operator for: a stencil, a geometric query, a rasteriser, a
particle interaction, a piece of numerical code you already have in C++ and that your model must now
differentiate through. If a few lines of `jnp` or `torch` express it, use those. loom starts where
the operation is *naturally* a loop over a structure that tensor primitives do not express, or where
the C++ already exists.

## Where to go next

- **New here** → [What loom is](/guide/what-is-loom), then [Why it is not trivial](/guide/why-not-trivial).
- **Want to see one** → [Diffusion](/tutorials/diffusion), the shortest complete example.
- **Data-dependent shapes** → [Gaussian splatting](/tutorials/splats).
- **Looking for a name** → [the Python API](/reference/python-api).

::: warning Status
loom is early and its API may still move. The two tutorials are programs that ship in `examples/`
and are run by their own tests. Jax is the most complete backend; Torch runs on CPU for now.
:::
