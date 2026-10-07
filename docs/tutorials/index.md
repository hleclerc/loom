# Tutorials

Two walkthroughs. Each is a program in [`examples/`](https://github.com/hleclerc/loom/tree/main/examples),
written as an outside user of loom would write it, and run by its own tests.

| | what it shows | level |
|---|---|---|
| [1 · Diffusion](./diffusion) | a stencil kernel, `mutable`, the adjoint, a body that works in 2-D, 3-D and under `vmap` | start here |
| [2 · Gaussian splatting](./splats) | a structure whose size depends on the data, a pipeline of passes, an atomic adjoint, measurements | after the first |

Both need loom installed, and a compiler: see [Installing](/guide/installing).
