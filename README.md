# loom

**Your C++ kernel, as a Jax/Torch/Numpy/Cupy/... function.**

Write the inner loops once, in plain C++ or CUDA. loom compiles it, packages it, hands it the framework's
tensors, runs it inside `jit`, `vmap` and `grad`, and gives you the gradient you wrote for it.

📖 **Documentation: <https://hleclerc.github.io/loom/>**

## In short

- **One call, several worlds.** Jax arrays, Torch tensors, cupy or numpy arrays go in as they are; the kernel
  sees pointers, strides and templates.
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

loom compiles kernels on first use, so a recent C++ compiler is needed. See [Installing](https://hleclerc.github.io/loom/guide/installing) for
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
