# Installing

loom is a Python package with a C++ header library. It compiles your kernels on first use, so the
machine needs a C++ compiler.

## From a checkout

```bash
git clone git@github.com:hleclerc/loom.git
cd loom
pip install -e .              # numpy only
pip install -e '.[jax]'       # with Jax
pip install -e '.[torch]'     # with Torch
pip install -e '.[cuda]'      # the pip CUDA compiler, for GPU kernels
```

Use a dedicated environment per project (a micromamba environment, a venv); loom does not care which.

## The compiler

loom writes C++ and calls the compiler through `ninja`, which is installed with it. Two things to
know:

- **The adjoint kernels need a recent compiler.** They rely on C++ rules (P2280, unknown references
  in constant expressions) that Apple's clang does not implement and that recent clang releases only
  enable from C++23 on. On macOS, install a conda `clangxx` and point loom at it.
- **For CUDA, the nvcc is pinned** by the `cuda` extra (`>=13.3,<13.4`): the 13.4 frontend fails on
  loom's kernels, 13.3 works.

```bash
export LOOM_CXX=$CONDA_PREFIX/bin/clang++
export LOOM_CXXFLAGS=-std=c++23
export LOOM_FRAMEWORK=jax          # which framework loom talks to
```

→ [Environment variables](/reference/environment)

## Check that it works

```bash
cd examples/diffusion
python diffusion.py
```

The first run compiles two small kernels (a few seconds each, with `ninja` output on the way) and
prints:

```text
40 diffusion steps on a 21x21 grid ( c = 0.2 )
  peak 1.0000 -> 0.1988
  sum  25.1327 -> 22.5915   ( it leaks through the boundary )
```

A second run prints only those three lines: the compiled kernels are cached
(`~/Library/Caches/loom` on macOS; see `LOOM_CACHE_DIR`).

::: tip Where things are cached
Compiled kernels are keyed by a hash of their source and of what it was rendered with. Changing the
C++ changes the key and recompiles; nothing else does.
:::

## Which backend

| framework | CPU | GPU |
|---|---|---|
| Jax | yes | CUDA, through the `cuda` extra |
| Torch | yes | not yet |
| numpy | yes | — |

::: warning To confirm
This table is written from the sources, not from a run on every combination. It is checked against
the test suite before release.
:::
