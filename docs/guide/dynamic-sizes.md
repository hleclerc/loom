# Sizes known at run time

Some kernels decide how many items they produce: the values kept by a filter, the vertices of a
cut cell, the splats that touch a tile. The buffers have to exist before the kernel runs, so the
size cannot be known when they are allocated. loom offers two ways out:

- **a capacity**: allocate room for a guess, let the kernel write how many items it used, and run
  the call again with more room when the guess was too small;
- **count, then fill**: a first kernel counts, the host reads the total and allocates exactly that,
  a second kernel fills.

## On one example

Keeping the positive values of an array. The output is declared with a count, `nb_kept`, that the
kernel writes:

```python
class Kept( loom.Aggregate ):
    values   : loom.RealTensor[ "num_kept" ]
    num_kept : loom.Axis[ "nb_kept" ]
    nb_kept  : loom.ShapeVar          # written by the kernel


def positives( x ):
    kept = loom.ffi_call( "positives", _forward,
        x = x,
        kept = loom.out( Kept(), capacities = { "nb_kept": 2 } ),   # room for 2, to begin with
    )
    return kept.values
```

The kernel is short and sequential, so it is a `loom.FfiCode.per_item` body: it runs once per item
of the batch, and sees `inputs`, `outputs` and `batch_index` directly.

```cpp
auto x = inputs.x( batch_index );
auto kept = outputs.kept( batch_index );

SI n = 0;
for ( SI i = 0; i < x.size( axis_0 ); ++i )
    n += x( axis_0 = i ) > 0;
kept.nb_kept.set( n );            // clamped to the capacity; recorded if it does not fit

for ( SI i = 0, k = 0; i < x.size( axis_0 ) && k < SI( kept.nb_kept ); ++i )
    if ( x( axis_0 = i ) > 0 )
        kept.values( num_kept = k++ ) = x( axis_0 = i );
```

`positives( numpy.array( [ 1., -2., 3., 4., -5., 6. ] ) )` is `[ 1., 3., 4., 6. ]`. The capacity
was 2 and the kernel wanted 4: the call ran again with room for 4, and returned the 4 values that
were written, not the whole buffer.

## In Python

### Capacities

A count that a kernel writes is a `ShapeVar` of an aggregate, and the axes over it size the
buffers. How much room to allocate is a decision about **one** call, so it is given to the call:

```python
loom.out( cell, capacities = { "nb_vertices": 8 } )
```

`capacities` is accepted by `loom.out`, `loom.mutable` and `loom.scratch`. When a buffer that uses
the count already exists (the output of a previous call), its room is read from it, and there is
nothing to restate.

A count can also have axes of its own: `nb_vertices : loom.ShapeVar[ "num_cell" ]` is one count per
cell. The buffer is then `cells × capacity`, and each cell uses the part its count says.

### What you get back

The count is part of the result. `kept.values` is read at its **count**, not at its capacity: in
Python, `kept.values.shape` is `[ 4 ]` while the buffer has room for more, and the array the caller
gets has 4 values.

Under a jit, the count is a value of the device, which the trace cannot read: it cannot shape an
array. A kernel-sized tensor is then given to **other kernels**, which read the count where it is:

```python
def sum_of_positives( x ):
    kept = loom.ffi_call( "positives", _forward, x = x,
        kept = loom.out( Kept(), capacities = { "nb_kept": 2 } ) )
    return loom.ffi_call( "kept_sum", _sum, kept = kept, s = loom.out( loom.RealTensor() ) )
```

```cpp
// _sum: the count, read on the device
for ( SI k = 0; k < SI( kept.nb_kept ); ++k )
    s += kept.values( num_kept = k );
```

`loom.jit( sum_of_positives )` works, and grows the capacity when it is too small (see
[below](#running-again)).

### Count, then fill: `loom.CsrTensor`

When the rows have very different lengths, a capacity per row wastes memory. A `CsrTensor` stores
the rows end to end, at exactly their size: `offsets` holds the bounds of the rows, `values` all of
them. It is built in three steps, from the [Gaussian splatting](/tutorials/splats) tutorial:

```python
# 1. count: one integer per tile
counts = loom.ffi_call( "splats_count", count_kernel,
    splats = splats,
    counts = loom.out( loom.IntTensor[ screen.num_tile ]() ),
)

# 2. the offsets, by a prefix sum on the device; `values` allocated at exactly the total
touching = loom.CsrTensor.from_counts( counts )

# 3. fill: each work-item takes its slot in a row with an atomic cursor
_, touching = loom.ffi_call( "splats_fill", fill_kernel,
    splats = splats,
    cursors = loom.out( loom.IntTensor[ screen.num_tile ]() ),
    touching = loom.out( touching, writes = ( "values", ) ),
)
```

The price is a host read: the total sizes `values`, and it is a count a kernel has just written. So
`from_counts` cannot run under a jit, where a capacity is the way.

## In C++

**Writing a count.** `count.set( n )`. When `n` exceeds the capacity, the count is clamped to it and
the overflow is recorded: the kernel does not need to check anything.

**Reading a count.** `SI( count )` is the count as it stands, never more than the capacity. A loop
bounded by it stays inside the buffers, whatever happened.

**Reading a CSR.** `csr( i, j )` is element `j` of row `i`: the `i` looks up the offsets. `csr.row_size( i )`
is the length of row `i`.

## Running again

When a count did not fit, the call has written what it could, and recorded how much room was
missing. It then runs again with `max( wanted, 2 * capacity )`: a capacity exceeded once tends to be
exceeded again, so it makes room rather than fit the count. Nothing of a failed run survives: the
outputs are fresh buffers. Where the call runs again depends on who allocates the outputs.

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

## Next

[Frameworks and devices](./frameworks) · [The adjoint](./adjoint) · [Gaussian splatting](/tutorials/splats)
