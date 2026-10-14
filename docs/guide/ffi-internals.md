# The FFI boundary

This page is about internals: how a Python tensor becomes something a C++ kernel can read.
Nothing here is needed to *use* loom — see [Tensors](./tensors.md) for that. It is here because the
choices it describes are the ones that decide performance, and because a new kind of value has to
slot into them.

---

## A tensor is three separable contracts

| contract | what it says | where it lives |
| --- | --- | --- |
| **axes** | what the dimensions mean, and how big they are | `axes` — `Axis` / `AxisList` |
| **element** | what it is made of, and whether a gradient flows through it | the class: `RealTensor`, `IntTensor`, `BoolTensor` |
| **storage** | how the value is actually backed | `storage` — see below |

Keeping them apart is what lets each one vary alone: an `IntTensor` is non-differentiable whatever
backs it; a symbolic zero is storageless whatever it is made of.

---

## How a value is backed

`storage` is one object per way a value can exist, and they map one-to-one onto the C++ types a
member lowers to:

| storage | C++ type | what it is |
| --- | --- | --- |
| `Unbound` | `NoneTensor` | declared, holds nothing — an optional member this kernel does not use |
| `Buffer` | `TensorView` | a real backend buffer, sized at capacity, optionally with a physical layout |
| `SymbolicZero` | `ZeroTensor` | holds a value (reads as 0) but backs no buffer |
| `Fill` | `FillTensor` | one scalar backing a whole logical shape |

An absent member is not a degenerate view the kernel has to test at run time: it is a distinct
**type**, so the kernel discriminates at compile time and a `static_assert` can forbid touching it.
A symbolic zero reads as 0 wherever it is indexed, so the compiler drops the arithmetic it feeds.

There are no kind flags on the tensor. `is_fill`, `is_symbolic_zero`, "is it bound", the physical
layout and the C++ spelling are all answered by the variant, so adding a way of being backed — a
range, a strided window, a broadcast — is a new variant with nothing to change in the lowering.

Two questions compose at the boundary, and they are genuinely independent:

- the **call** decides whether a value crosses at all. Holding a buffer does not make a member an
  argument (a call may exclude it), and holding nothing does not make it absent — an *output* is
  precisely a member the call is about to allocate a buffer for.
- the **storage** decides what form it takes when it does cross.

Hence the `bound_cpp_type` / `absent_cpp_type` pair each variant answers.

---

## Counts cross by value when they can

A count is one integer. When the host already knows it and the kernel only *reads* it, sending it
through a device buffer costs an allocation, a transfer and a dereference on every read, and buys
nothing — the value is uniform over the whole call. So it travels as an XLA FFI attribute and lands
in the kernel as a `ScalarValue<SI>`, in registers:

```cpp
static ffi::Error sdot_ffi_impl( ..., int64_t count_cnt_nb_wanted ) {
    auto nb_wanted = make_shape_var_view( ScalarValue<SI>{ SI( count_cnt_nb_wanted ) }, ... );
```

`ScalarValue` sits between the two things it is not:

- `TensorView` — a *pointer* into memory the kernel dereferences.
- `Ct<SI,n>` — the value in the *type*, compile-time known, part of the library hash. This is what a
  `CtShapeVar` is: it enables specialization, at the price of one compiled library per value.

A buffer is kept exactly where it is unavoidable:

- the kernel **writes** the count — that is where the result goes;
- the count is **ragged** — one per segment;
- the count is **per batch item**;
- the host does not know it — a count a previous call wrote, which under a `jit` Python cannot read.

That last case is decided on `static_count()`: a count *prescribed* in Python, or solved from the
shape of a tensor we were given, is a fact that holds whether or not we are tracing. A count a
kernel wrote is not — eagerly Python could read it back, but under a `jit` it is a tracer, and an
FFI attribute cannot be one. Keying on `static_count` keeps the generated signature identical in
both, instead of compiling a second library for the eager case.

---

## A fill crosses as one scalar

`Tensor.filled_with( v )` is a value whose every element reads the same scalar. It crosses as one
rank-0 buffer, not as an `[n]` array — and its logical extents are not baked into the generated
source either. The C++ view reads them off a *sibling* argument that carries the same axis:

```cpp
sdot_ffi_impl( ffi::BufferR1<F64> ffi_x, ffi::BufferR0<F64> ffi_f, ... )
auto f = FillTensor<double, Tuple<SI>, Tuple<_num>>{ ffi_f.typed_data(),
                                                     tuple( SI( ffi_x.dimensions()[ 0 ] ) ) };
```

So one compiled library serves every fill value and every size. Indexing a fill ignores the index
and yields the scalar; it is read-only, since there is nowhere to write a broadcast.

::: warning Not yet chosen automatically
`Tensor.full( v )` still materializes a real buffer. The symbolic form works end to end, but
nothing produces it on its own yet: it needs a fill to survive the symbolic algebra it flows
through (a `c * fill` must stay a fill).
:::

---

## The dtype is a contract the FFI spells

A tensor's dtype is lowered as the C++ element type of the buffer it binds. A buffer whose type
disagrees is therefore *reinterpreted* by the kernel, not converted — silent garbage rather than a
type error. That is why the declaration is enforced at every point where a buffer becomes a
tensor's (`set`, `set_raw`), checked once more at the boundary, and why a *derived* tensor — an op
result, which declares nothing — reads its dtype off the buffer instead of inheriting it.

---

## Capacity, counts, and the error buffer

A capacity is a guess: only the kernel knows how many items it will produce. So a kernel may ask for
more than it was given. `ShapeVarView::set` compares the count against the `max` it was handed,
records the overflow in the call's error buffer, and **clamps** the count — so whatever the body
writes next stays inside the buffers this call allocated. Python then reserves more and runs again.
Nothing of a failed run survives: an output is a fresh buffer every time.
