# loom

**Your C++ kernel, as a Jax or Torch function.**

Write the inner loop once, in plain C++ or CUDA. loom compiles it, hands it the framework's
tensors, runs it inside `jit`, `vmap` and `grad`, and gives you the gradient you wrote for it.

📖 **Documentation: <https://hleclerc.github.io/loom/>**

## In short

- **One call, two worlds.** Jax arrays, Torch tensors or numpy arrays go in as they are; the kernel
  sees pointers, strides and templates.
- **Kernels that never count dimensions.** Coordinates are read by axis name, so the same body runs
  in 2-D or 3-D, batched or not, and a `vmap` adds an axis without the kernel knowing.
- **Roles on the value.** `loom.out( x )`, `loom.mutable( x )`, `loom.scratch( x )` say what the
  kernel writes or borrows, right where the argument is passed.
- **The adjoint is a kernel like any other.** Written in the same C++, wired into
  `jax.custom_vjp` / `torch.autograd.Function`, checked by `loom.testing.check_grad`.
- **Sizes learnt at run time.** A kernel counts, the host allocates exactly, even under `jit`.
- **Compiled on demand, or shipped precompiled.** The first call compiles and caches; a wheel can
  carry a prebuilt catalogue for users without a compiler.

## A glimpse

```cpp
struct OneStep {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        const auto main_axes = coords.axes - batch_axes;     // my axes, not the vmap's
        const TF uc = args.inputs.temperature( coords );
        TF total = 0;
        for_each( main_axes, [&]( auto axis ) {
            total += args.inputs.temperature( coords + axis )
                   + args.inputs.temperature( coords - axis ) - 2 * uc;
        } );
        args.outputs.temperature( coords ) = uc + args.inputs.coef * total;
    }
};
```

```python
import loom

def step( temperature, coef ):
    return loom.ffi_call( "diffusion_step", _forward, _backward,
        temperature = loom.mutable( temperature ),   # read, then rewritten
        coef = coef,
    )
```

The full walkthrough, adjoint included, is the
[diffusion tutorial](https://hleclerc.github.io/loom/tutorials/diffusion).

## Installing

```bash
git clone git@github.com:hleclerc/loom.git
cd loom
pip install -e '.[jax]'       # or '.[torch]', '.[cuda]' for GPU kernels
```

loom compiles kernels on first use, so a recent C++ compiler is needed (the adjoints rely on C++23
rules Apple's clang lacks). See [Installing](https://hleclerc.github.io/loom/guide/installing) for
the details, then try `python examples/diffusion/diffusion.py`.

## Repository layout

```
src/loom/     the Python package and the C++ header library
examples/     diffusion and Gaussian splatting, the two tutorials' programs
tests/        Python and C++ tests
docs/         the website (VitePress): `make site`, `make site-deploy`
```

## Status

loom is early and its API may still move. Jax is the most complete backend; Torch runs on CPU for
now.
