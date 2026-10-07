# What loom is

> **Write a kernel once, in C++ or CUDA, and call it as a function of your framework — Jax, Torch or
> numpy — with the gradient you wrote for it.**

You have an operation that a few `jnp` or `torch` calls do not express well. Perhaps it is a loop
over a structure whose size depends on the data, perhaps it is C++ you already own, perhaps it is
simply faster and clearer as a loop. You want it to sit in your model like any other operation:
accept arrays, work under `jit` and `vmap`, and be differentiable.

loom is what turns the C++ function into that.

## The shape of it

You provide two things.

**The kernel.** A functor in plain C++ (or CUDA), working on views of tensors:

```cpp
HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
    args.outputs.next( coords ) = args.inputs.u( coords ) + args.inputs.coef * laplacian( coords, args );
}
```

**The call.** One Python function, which names the kernel, passes the arguments, and says what is
written:

```python
return loom.ffi_call( "step", _forward, _backward,
    u    = loom.mutable( u ),
    coef = coef )
```

loom provides everything between the two: it generates the glue, compiles it, caches it, registers
it with the framework, binds the framework's buffers to the views the kernel reads, allocates the
outputs, and plugs the adjoint into the framework's own autodiff.

## What you get

| | |
|---|---|
| **One kernel, three frameworks** | The same C++ is called from Jax (inside `jit`, `vmap`, `grad`), from Torch (through `torch.autograd.Function`) and from numpy. |
| **Dimension-agnostic bodies** | Coordinates are read by name. `coords + axis` is the neighbour along that axis. The same kernel serves 2-D and 3-D, batched or not. → [Axes](./axes) |
| **Roles on the values** | `loom.out`, `loom.mutable`, `loom.scratch`, `loom.unbound` say what each argument is, where it is passed. → [Roles](./arguments) |
| **Hand-written adjoints** | The backward is a kernel, wired into the framework's autodiff and checked by `loom.testing.check_grad`. → [The adjoint](./adjoint) |
| **Run-time sizes** | A kernel counts, the host reads the total, the next buffer is exactly that big, also under `jit`. → [Sizes known at run time](./dynamic-sizes) |
| **Structured tensors** | Named axes, computed extents, aggregates of tensors, `CsrTensor` for ragged data. |
| **Compilation handled** | Compiled on first call and cached; or precompiled into a catalogue for a wheel. → [Compilation](./compilation) |

## How it works

When Python reaches `loom.ffi_call`:

1. loom reads what it was given: shapes, element types, device, which framework owns the arrays.
2. It renders a C++ source: a handler for the framework's foreign-function interface, plus a
   launcher that calls your `kernel( queue, batch_axes, args )`. The `args` carry your arguments
   under their Python names, in groups: `args.inputs`, `args.outputs`, `args.scratch`, and for the
   adjoint `args.grad_of_outputs` and `args.grad_of_inputs`.
3. It compiles that source with the device's compiler and caches the library, keyed by a hash of the
   source and everything it was rendered with. A second call, or a second process, finds it.
4. It registers the handler with the framework (an XLA FFI target for Jax) and calls it. For Jax this
   works eagerly and under `jit`.
5. If there is an adjoint, it is registered as the backward of the call: `jax.custom_vjp` for Jax,
   `torch.autograd.Function` for Torch.

```
   Python                          loom                                   C++
 ┌──────────┐   ffi_call   ┌───────────────────────────┐   kernel( queue, batch_axes, args )
 │ jax array│ ───────────► │ read shapes / dtypes      │ ──────────────────────────────────►
 │ torch    │              │ render + compile + cache  │    your functor, on views of
 │ numpy    │ ◄─────────── │ bind buffers, allocate    │ ◄──────────────────────────────────
 └──────────┘   outputs    │ register forward + adjoint│
                           └───────────────────────────┘
```

## What it is not

**It is not a tensor library.** loom does not provide the operations of your model. It provides the
bridge for the one you write yourself.

**It does not auto-differentiate your C++.** The adjoint is a kernel you write, with the same tools.
loom makes it easy to wire and to check, not to avoid.

**It does not replace `jnp` or `torch` where they suffice.** A kernel is worth its cost when the loop
is the natural way to say the operation, or when the C++ already exists.

## Next

[Why it is not trivial](./why-not-trivial) · [Installing](./installing) · or go straight to
[Diffusion](/tutorials/diffusion).
