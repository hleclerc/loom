# Installing

loom is a Python package with a C++ header library. It compiles your kernels on first use, so the
machine needs a C++ compiler.

## Install

```bash
pip install loom
```

loom does not install a framework: it uses the ones you have, whichever they are. A call runs on
the framework of the arrays it is given.

For CUDA kernels, the `cuda` extra brings a pinned nvcc (see below):

```bash
pip install 'loom[cuda]'
```

From a checkout, `pip install -e .` in the clone.

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
```

`loom-toolchain` says what this machine can compile for: the compiler and flags of each device,
and what is missing.

When several frameworks are installed, the one used for values built *from nothing*
(`RealTensor[ n ].zeros()`) is the first found among Jax, Torch, CuPy and NumPy;
`LOOM_FRAMEWORK=torch` chooses another. A call given arrays always runs on theirs.

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

| framework | CPU | CUDA | gradient, `vmap` |
|---|---|---|---|
| Jax | yes | yes | yes |
| Torch | yes | yes | yes |
| CuPy | — | yes | no, forward only |
| NumPy | yes | — | no, forward only |

CUDA kernels are compiled by the nvcc of the `cuda` extra. The test suite runs on every line of
this table.
