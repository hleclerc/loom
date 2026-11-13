# Compilation and catalogue

::: warning To write
The cache, `LOOM_KERNELS=auto|catalogue|compile`, the `loom-kernels` catalogue builder, the `loom-toolchain` command, build directories, GPU architectures.
:::

## The catalogue: one directory, one tag per architecture

A catalogue is a directory, and the directory is the index:

    <root>/<tag>/<key>.so            one library per kernel
    <root>/<tag>/libloom_runtime_*   the runtime they share, found next to them
    <root>/<tag>/index.json          what is in it (for `catalogue.report`; never read to look up)

The *target* describes the architecture: the instruction-set level of a CPU (`x86-64-v4`, `x86-64-v3`, `x86-64-v2`,
`x86-64`, or `armv8-a`) or the compute capability of a GPU (`sm_90` ... `sm_70`). A *tag* is `<kind>-<target>`
(`cpu-x86-64-v3`, `cuda-sm_80`). The *key* of a kernel is a hash of its source: it does not depend on the architecture.

**Serving.** A machine looks in ONE tag: the architecture its device behaves as (`Device.current_catalogue_tag`) -- a
`stat`, no index to load, and only the libraries that are used get loaded. There is no fall back on a poorer architecture:
its build is another code, with other results and timings. Which architecture a kernel runs as is decided by the choices
table (`loom.kernel_parameters_for`, measured by `loom.benchmark`), and by nothing else. `LOOM_KERNELS=auto|catalogue|compile`
says what to do when the tag has no such file.

**Producing.** There is no tool to run: a Python script of the package that ships the catalogue runs its standard usage,
and the loops on the hardware belong to it.

```python
from loom import catalogue, hardware

with catalogue.collect( "my_package/_catalogue" ):
    for run in hardware.runs():                # each device present, as each architecture it can behave as
        with hardware.behave_as( run ):
            exercise_the_package()
print( catalogue.report( "my_package/_catalogue" ) )
```

Inside `collect` the catalogue is not consulted: every kernel the program triggers is compiled, then copied under the tag
of the architecture the device behaves as. Inside `behave_as( run )` the device really behaves as `run.target`
(`LOOM_TARGET`): a machine can pretend to be a *weaker* one -- an AVX-512 CPU as `x86-64-v3`, a card of capability 8.9 as an
8.0 -- never a stronger one, so the program, kernels included, really runs. That is what makes it work when the hardware
decides things in Python that loom does not manage (`device.nb_threads`, `group_size`): they are decided again for each
architecture, and a kernel whose source depends on them gets another key. `catalogue.report` lists those kernels.

- A kernel is published under every architecture the program is run as.
- The architectures that no hardware present can run cannot be produced (`hardware.uncovered` says which). A GPU build is
  made on a machine that has the card; the directories of several machines are merged.
- Producing a catalogue for a wheel is done in the manylinux image (Linux) or with a `MACOSX_DEPLOYMENT_TARGET` (macOS), so
  that the binaries load on the oldest system supported.

## What a kernel is made of: body, entry and units

A kernel is not one library. For a call on the CPU or on a card, the generated source is cut in two:

* the **body** -- the work of the call and everything it instantiates -- is compiled **once** as a unit of its own (an `FfiModule`), and exported as a function of plain buffers (`loom_raw::Raw`), `int64` attributes and, on a card, the stream and the memory pool (`loom_raw::Context`);
* the **entry** of each framework is a few lines that decode what the framework hands over and call the body: Jax's XLA frame (decoded by hand, `loom/support/ffi_raw.h`), or the pointer frame of Torch, Cupy and Numpy.

So Jax, Torch, Cupy and Numpy share one body, and a catalogue holds it once. `LOOM_RAW_FFI=0` goes back to the typed XLA API and one library per framework.

Code that is not specific to a call is a **unit**: a shared library built once per (source, defines, compiler) and found beside the kernels (`$ORIGIN`). Name it with `FfiCode( sources = [ ... ] )`, either a file (`"sdot/x.cpp"`, with defines) or an `loom.FfiModule`:

```python
engine = loom.FfiModule( "engine_3d", includes = [ "sdot/engine/Cells.h" ],
                         source = f"namespace sdot::engine {{ template LOOM_EXPORT int engine_run<{ local },{ seeds }>( ... ); }}" )
loom.ffi_call( name, loom.FfiCode.inline( "...", sources = [ engine ] ), ... )
```

The text is written to `loom_modules/<name>_<hash>.h` and `.cpp` (or `.cu`), compiled once, and the header is included by the kernels that name it. Entry points of a unit wear `LOOM_EXPORT`, since kernels are compiled with hidden visibility.

## External libraries

A header-only library that is not part of any package -- a sparse solver, a linear algebra library -- can be fetched by loom itself, with no system package and no manual step. A package **declares** it once, at import, with a pinned archive and its SHA-256:

```python
loom.compilation.register_external( "mylib", "1.2.0", "https://example.org/mylib-1.2.0.tar.gz",
                                    "<sha256 of the archive>", include = "" )   # the subdirectory to put on `-I`
```

Declaring fetches nothing. A kernel that includes the library **asks** for it by name:

```python
loom.ffi_call( name, loom.FfiCode.inline( "...", sources = [ "mypackage/solver.cpp" ], externals = [ "mylib" ] ), ... )
```

The archive is then downloaded once, into `<cache>/ext/<name>-<version>` (`LOOM_EXT_DIR` puts it elsewhere), and its directory goes on the `-I` of that kernel **and of the units it links** -- of nothing else. A kernel that does not ask never triggers the download and never sees the headers, and the same unit compiled for a kernel that asks and for one that does not is two units.

The request is written on the first line of the kernel's source (`// loom externals: mylib@1.2.0`), so it is part of the kernel's name and of its catalogue key: a new pin is a new kernel.

Whether the library can be had is checked **once per process**. Without network, or with `LOOM_EXTERNALS=0`, it is missing, loom says so once on stderr, and the source makes do without it -- which it must be written for, with `__has_include`:

```cpp
#if __has_include( <mylib/solver.h> )
#  include <mylib/solver.h>
#  define HAS_MYLIB 1
#endif
```

The flip side: a kernel that includes the library but forgets `externals` compiles without it, silently. A name nobody declared is an error.

On a card, units are the only way to share code: the device code of a library cannot be linked to that of another one (`nvlink` links objects and static archives, not shared libraries), and a function pointer taken in one library is not callable from another. `kernels/cuda_support.cu` is such a unit: the fills that every call would otherwise instantiate for its own view types. The device flags of a catalogue build are `--cudart=shared` (the CUDA runtime of the framework is used), `--compress-mode=size`, and no PTX (a catalogue never falls back to another architecture).

The extents and byte strides of a buffer whose layout is not the identity (a card aligns its batches) cross as attributes, not as literals of the source: the same kernel serves every size.
