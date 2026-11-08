# Sizes known at run time

::: warning To write
`CsrTensor.from_counts` and the count / prefix-sum / fill pattern. Source: the splats example, and `tests/test_scratch*.py`.
:::

## Capacities and overflow

A kernel often decides how many items it produces: the vertices of a cut cell, the rows that go to a rank. The output
buffers have to exist before it runs, so the call allocates them at a **capacity** -- a guess, given by the caller
( `loom.out( cell, capacities = { "nb_vertices": 8 } )` ) or read back from a buffer that already exists.

The guess may be too small, and that is not an error. The contract every kernel honours, without writing anything for it:

* when a count is set past its capacity ( `c.nb_vertices.set( n )` ), the count is **clamped** to the capacity, so whatever
  the body writes next stays inside the buffers, and
* the count it wanted is **recorded** in the call's error buffer.

After the run, the error buffer says whether the contract held, and how much room was missing. The call then runs again with
`max( wanted, 2 * capacity )` -- a capacity exceeded once tends to be exceeded again, so it makes room rather than fit the
count. Nothing of a failed run survives: the outputs are fresh buffers. You get outputs that are large enough; WHERE the
call runs again depends on who allocates the outputs.

### NumPy, Torch, CuPy: in the call

Under these frameworks loom allocates the outputs itself. So the call grows them **itself**: after a run, the generated
entry reads the count of records of the error buffer in C++ ( on a card, on the call's stream ), and on an overflow it asks
for larger outputs and runs the body again -- in the same call. Python is only asked on an overflow, to analyse the call
again with the grown capacities and allocate the new outputs ( `CallArgsAnalysis.grown` ); a call that does not overflow
never goes back to it. A batched call ( `vmap` ) grows the same way.

### Jax: from the host

XLA allocates the outputs of a call **before** it runs, with shapes fixed when the program is traced. The call cannot grow
them, only its own scratch ( `loom.scratch` ). So an output overflow is handled from Python:

* **eager**: the call reads the error buffer and runs again, as above;
* **`loom.jit`**: the function is traced and compiled once. Each call adds a check on its error buffer ( `checkify` ), which
  leaves the compiled program as an output -- through `lax.while_loop`, `scan`, `cond`, `vmap` and `grad`, the user's own
  included. The host reads it once, after the run. On an overflow, the capacity of that call site -- the call's name and its
  rank among the calls of that name -- grows and is **kept** for the next calls, and the whole function is traced and run
  again. Running again is safe because a jitted Jax function is pure; `donate_argnums` is refused for that reason.
* **plain `jax.jit`**: the trace cannot carry the check out, so each call checks at run time with a `jax.debug.callback`,
  and raises if a capacity was too small -- it cannot run again. The callback is a host round trip in the middle of the
  program, with the card idle while Python runs, and XLA can no longer drop or merge the call.

What the check costs under a jit, measured on an RTX 2080 Ti with a chain of small dependent calls
( `sdot-ffi/bench/experiments/jit_watch.py` ):

| | per call |
|---|---|
| `jax.jit`, `jax.debug.callback` | ≈ 415 µs |
| `loom.jit`, `checkify` | ≈ 38 µs |
| no check at all | ≈ 28 µs |

A call whose outputs are all sized upstream ( no count is decided by the kernel ) can say so with
`has_dynamic_capacity = False`: no check is made, under `jax.jit` either.

Under an outer transform ( `jax.grad( loom.jit( f ) )` ), the error would be a tracer the host cannot read: `loom.jit` is then
a plain `jax.jit`, with the callback.

### `torch.compile`

A kernel call is a break in the compiled graph ( see [Frameworks](./frameworks#jit) ): it runs eagerly, on real values, so it
grows its outputs in the call as above.
