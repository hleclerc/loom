# Why it is not trivial

"I will just write a C++ function and call it from Python" works on the first afternoon. It stops
working when the function has to sit inside a model: under `jit`, under `vmap`, on a GPU, with a
gradient. The difficulty is not the C++. It is that two sides with different assumptions have to
agree on every call.

## Two sides that do not share assumptions

| | The deep-learning world | The C++ world |
|---|---|---|
| **Values** | Immutable tensors; `x.at[i].set(v)` makes a new one | Mutable memory behind pointers |
| **Who owns the memory** | The framework: XLA's allocator, Torch's caching allocator | You, or an allocator you choose |
| **Shapes** | Part of the type, known when tracing | Runtime values, or template parameters |
| **Execution** | Traced once, run many times, scheduled asynchronously on streams | A function that runs when called |
| **Dimensions** | `vmap` adds or removes axes at will | `for ( i … ) for ( j … )`, fixed by the author |
| **Derivatives** | Built by the framework from the operations it knows | By hand, if at all |
| **Types** | A dtype tag | Templates, concepts, overloads |

A custom kernel has to be valid in both columns at once.

## What you meet when you do it yourself

### 1. Each framework has its own door

Jax calls foreign code through XLA's FFI: a handler with a specific signature, registered by name
for a platform, receiving buffers and returning results. Torch has custom operators and
`autograd.Function`. NumPy has `ctypes` or a C extension. CuPy hands over device pointers, and the
stream to launch on is yours to find. Four bindings, four ways to describe buffers, four ways to
fail when you get it wrong. The kernel, on the other hand, should be one.

### 2. The code around the kernel is written per framework, too

A kernel is rarely alone. Around it: allocate an output, reshape, sum, take a prefix, read a count,
hand the result back. Each of these is `jnp.zeros`, `torch.zeros`, `numpy.zeros` or `cupy.zeros`,
and they do not agree on devices, dtypes or in-place updates. Either the code around your kernels
is written four times, or it is full of `if`s, or it supports one framework. And your users, each
in their own, expect their own arrays back.

loom makes it one kind of value: inside, a loom tensor, with its axes and the same operations
whoever owns its buffer; at the boundary of your module, the user's arrays in and out
(`loom.export_namespace( globals() )`). → [Tensors](./tensors)

### 3. Inputs and outputs are disjoint

XLA never lets a kernel write into what it reads: an in-place update is two buffers and a rebinding
on the Python side. Written by hand, that is the classic source of aliasing bugs. In loom,
`temperature = loom.mutable( temperature )` says "read it, then rewrite it", and the kernel sees
`args.inputs.temperature` and `args.outputs.temperature`.

### 4. Layouts, padding, alignment

The buffer behind a tensor is not always the logical tensor: strides, padding for alignment (128
bytes on CUDA, so a batch of 3 in fp64 takes 16 slots), capacities larger than the used extent. A
kernel that indexes the raw pointer reads garbage or someone else's memory. loom hands the kernel a
view that knows its axes and extents, and crops what it returns.

### 5. The kernel does not know how many dimensions it has

`vmap` adds an axis. A hand-written 2-D stencil is silently wrong, or crashes, the first time it is
batched. A kernel that is *meant* to be vmapped has to separate its own axes from the ones the
caller added. loom makes that a set operation: `coords.axes - batch_axes`.

### 6. Derivatives are a contract

For `jax.grad` or `loss.backward()` to work, you provide a backward, and the framework decides what
to call it with: the cotangent of each output, possibly a *symbolic zero* the framework never
materialised, possibly no cotangent wanted for some input. The residuals saved between forward and
backward must be exactly what the backward needs, not what a scratch buffer happened to hold. And a
wrong adjoint does not crash: it trains a little worse. loom gives the backward the same call
machinery as the forward, tells it which cotangents are wanted (`args.grad_of_inputs.x.is_valid`),
and ships `check_grad`, a finite-difference check on random projections.

### 7. Shapes must be known to trace

An output shape is fixed when the function is traced. That is fine for a stencil. It is not for an
index whose length depends on the data (the splats that touch each tile, the neighbours of a point).
You either pick a bound before tracing, and lose elements or pay for the worst case, or you leave
the framework. In loom, a count a kernel has just written sizes what follows: under Torch, NumPy
and CuPy, in the call itself; under `loom.jit`, an output too small is detected in the compiled
program, and the function runs again with more room, which is then kept.
→ [Sizes known at run time](./dynamic-sizes)

### 8. Devices, streams and threads

The kernel must run where its data lives, on the stream the framework scheduled it on, without
synchronising behind its back. On CPU it needs a thread pool of its own. On CUDA it needs the right compiler (and the right
version: see [Installing](./installing)). loom passes a `queue` that carries all of that.

### 9. Compilation is part of the user experience

Types, ranks and the real scalar (`TF`) change per call, so a kernel is specialised per signature.
Compile on every call and nothing is usable; compile never and every user needs a toolchain. loom
caches by a hash of the source and what it was rendered with, and can precompile a catalogue for a
wheel. → [Compilation](./compilation)

## What loom changes

None of that is deep, and each piece is a day of work. Together, for each new kernel, they are a
week, and they come back with every change of framework version. loom does them once:

- one `ffi_call`, whatever the framework;
- one kind of value inside, and the user's own arrays at the boundary;
- views that know their axes, so kernels are dimension-agnostic;
- roles said on the value, so aliasing is a declaration and not a convention;
- the adjoint wired through the framework's own autodiff, and a checker for it;
- run-time sizes, also under `jit`;
- compilation cached, and shippable.

What remains is the part that only you can write: the loop.

## Next

[Installing](./installing) · [Your first kernel](./first-kernel) · [Diffusion](/tutorials/diffusion)
