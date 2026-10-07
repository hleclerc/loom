# Frameworks and devices

A kernel is written once. What changes from one framework to the next is how a value reaches it and
how the call fits in the framework's own machinery.

| | Jax | Torch | numpy |
|---|---|---|---|
| How a value reaches the kernel | XLA buffer, inside the traced program | by address (`data_ptr`) | by address |
| Devices | CPU, CUDA | CPU, CUDA | CPU |
| `vmap` | batching rule: one launch of a kernel with one more axis | the same, through `torch.func.vmap` | — |
| `grad` / `vjp` | `custom_vjp` | `torch.autograd.Function` | — |
| `jit` | XLA: the call is inside the compiled program | `torch.compile`: the call runs between compiled regions | identity |

## Nothing is copied that does not have to be

Under Torch and numpy a tensor is handed to the kernel **where it is**: same address, whichever device it
lives on. A value held with its own strides -- a transposed view, a slice, an `expand` (stride 0) -- is read
with those strides: the kernel is generated to take them from the buffer at run time, so a new stride never
means a new compilation. The only copy is the one you ask for.

A tensor on another device than the kernel's is refused with a message, rather than moved behind your back:
a transfer is a cost you should see.

## Bringing a result back to the host: `host`

Reading a result on the host is sometimes needed (a test, a plot, a decision in Python). Frameworks do not
agree on how: `numpy.asarray( x )` works for Jax and numpy, but a Torch tensor on a card refuses it, and a
traced value has no content to read at all. `loom.testing.host` is the one spelling:

```python
from loom.testing import host

host( out.value ).tolist()      # whichever framework, wherever the value lives
```

It is `driver.to_numpy`, and it copies **only if the value is not already on the host**. Use it where the
data really has to come back; code that stays on the device -- which is most of it -- never needs it. In
particular, numpy has no place in code that must run on a GPU or under a trace.

## Torch on a card

On a CUDA device the kernel is launched on the stream Torch is working on and takes its scratch memory
(`FfiCode( allocator = True )`) from Torch's own allocator. What Torch launched before is finished when the
kernel starts, and what the kernel launched is finished when Torch reads the result: no synchronization to
write, on either side.

## `vmap`

A mapped call is not replayed item by item. The code gains one batch axis, the buffers that were mapped gain a
leading dimension, and the kernel that comes out runs the `N` items in a single launch -- under Jax and under
Torch alike. Its backward is the backward of one item, mapped: an input the mapping did not give an axis to is
shared by every item, and receives the sum of what they contribute.

## `jit`

Under Jax, `jit( f )` runs `f` once on *tracers* -- values with a shape and a type but no content -- and what
it records, kernel calls included, is compiled into one program. The call is a node of that program, and while
`f` runs nothing can be read: a count a kernel wrote has no value yet.

Under Torch, `jit( f )` is `torch.compile( f )`, and a kernel call is **not** put in the compiled graph: it is a
break in it. The tensor code before and after is compiled; the call itself runs eagerly, on real values, between
the two. Two consequences:

* a call can read what a kernel just wrote (a count, a capacity) and size what follows from it, which a traced
  call cannot -- see [Sizes known at run time](./dynamic-sizes);
* a kernel is not fused with the code around it, and the whole of `f` is not one launch.

Inductor needs a C++ compiler. On macOS, inside a conda environment that has `clangxx`, loom points it to
Apple's `/usr/bin/clang++` (conda's cannot build Inductor's precompiled headers); `CXX` overrides this. If the
machine cannot compile at all, `jit` warns once and runs `f` as it is: the values are the same.
