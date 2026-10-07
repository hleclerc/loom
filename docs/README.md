# The loom website

[VitePress](https://vitepress.dev), same tooling as errand's. Everything web lives under this
directory, so the project root stays a Python package root.

```bash
cd docs
npm install
npm run dev        # http://localhost:5173/loom/
npm run build      # -> .vitepress/dist
```

From the project root: `make site`, `make site-build`.

## Layout

```
index.md                 the home page: the pitch, one example, the five ideas
guide/                   the documentation; start with what-is-loom and why-not-trivial
tutorials/               two walkthroughs, one per example in examples/
reference/               Python API, C++ API, LOOM_* variables
scripts/                 programs that generate the tutorials' animations (`make anim`)
public/anim/             the generated GIFs, committed: building the site does not need jax
.vitepress/config.mts    nav, sidebar, base
```

## Keeping it true

Every code block of a tutorial is taken from `examples/`, which is run by its own tests
(`errand test_diffusion`, `errand test_splats`). When an example changes, its tutorial is part of
the change. Pages that state an option or a variable are written from the source (`src/loom/`),
not from memory.

Pages marked *to write* are stubs: they say what they will contain.
