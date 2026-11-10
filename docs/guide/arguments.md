# Roles of the arguments

Every argument of `loom.ffi_call` has a role: read, written, read and rewritten, or used as work
space. **The role is said where the value is passed**, by wrapping it:

```python
def step( x, dt ):
    return loom.ffi_call( "step", _forward,
        x      = loom.mutable( x ),                              # read, then rewritten
        dt     = dt,                                             # read
        sq     = loom.scratch( loom.RealTensor.like( x ) ),      # work space
        energy = loom.out( loom.RealTensor.like( x ) ),          # written
    )
```

and the kernel finds each argument under its Python name, in the group of its role:

```cpp
HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
    const TF x = args.inputs.x( coords );
    args.scratch.sq( coords ) = x * x;
    args.outputs.x( coords ) = x + args.inputs.dt;
    args.outputs.energy( coords ) = args.scratch.sq( coords ) / 2;
}
```

The call returns what it wrote, `x` and then `energy`, in the order of the arguments:
`x, energy = step( x, 0.5 )`.

## In Python: the roles

| | the kernel | the call returns it |
|---|---|---|
| `value` | reads it | no |
| `loom.out( value )` | writes it | yes |
| `loom.mutable( value )` | reads it, then writes its new value | yes |
| `loom.scratch( value )` | writes and reads it during the call | no |
| `loom.unbound( value, "member", … )` | does not see these members | no |

### Inputs

An argument with no marker is an input. It can be an array of any framework, a loom tensor, a
Python number, a list, or an [aggregate](./axes#aggregates). loom reads the element type of an array
(real, integer, bool), and leaves the size of its scalars (32 or 64 bits) to the
[framework settings](./frameworks).

### `loom.out`: written

`loom.out( t )` takes a tensor with no content: `loom.RealTensor.like( x )`, or
`loom.RealTensor[ row ]()` with [named axes](./axes#named-axes). loom allocates its buffer from its
axes, and the kernel writes all of it.

An aggregate that the kernel writes only in part says which part, from either end:

```python
loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ) )   # these, and only these
loom.out( cell, reads  = ( "weights", ) )                          # everything but these
```

The two lists are exclusive, and checked: a name that designates nothing is refused. `reads` is
usually the list that lasts: it names what the kernel must not overwrite, so it does not change when
an output is added to the aggregate.

### `loom.mutable`: read, then rewritten

Inputs and outputs of a call are **disjoint**, as in XLA: a kernel never writes a buffer it reads.
An update in place is therefore two buffers, the old value and the new one, and a rebinding in
Python: `x = step( x, dt )`. `loom.mutable( x )` says all of that. The output buffer is built like
the input.

### `loom.scratch`: work space

A buffer the kernel writes and reads during the call. It is allocated like an output, but it is not
returned, and the [adjoint](./adjoint) does not receive it: what it held is transient.

### `loom.unbound`: not seen

`loom.unbound( cell, "cut_offsets" )` hides members of an aggregate from this kernel, even when they
hold values: nothing of them crosses, and the adjoint has no gradient to find for them.

### Capacities

`loom.out`, `loom.mutable` and `loom.scratch` take `capacities = { "nb_vertices": 8 }`: how much room
to allocate for a count only the kernel will know. When a buffer already has that room, there is
nothing to say. See [Sizes known at run time](./dynamic-sizes).

### What the call returns

Everything it wrote, `loom.out` and `loom.mutable`, in the order of the arguments: one value alone,
a tuple when there are several, `None` when there are none. Scratch buffers are not returned.

Inside a module that ends with `loom.export_namespace( globals() )`, these are loom tensors. The
caller gets arrays of its own framework, when they cross the boundary.

### Settings of the call

Four names are settings of the call, not arguments of the kernel:

| | |
|---|---|
| `nb_items = n` | launch `n` items, without a tensor carrying the axis (see `FfiCode.per_item`) |
| `failures = { code: message }` | what a failure the kernel records means: the call raises `KernelFailure` with the message |
| `batch_alignment` | the alignment of the batch dimension in memory |
| `has_dynamic_capacity = False` | no output has a count the kernel decides: under a trace, the check for an overflow is skipped |

## In C++: the groups

`args` holds one group per role, and each argument under its Python name:

| | |
|---|---|
| `args.inputs.<name>` | the inputs, scalars included |
| `args.outputs.<name>` | what the kernel writes |
| `args.scratch.<name>` | the work buffers |
| `args.grad_of_outputs.<name>` | (adjoint) the gradient that comes in, for each output |
| `args.grad_of_inputs.<name>` | (adjoint) the gradient to write, for each input |
| `args.machine`, `args.errors` | the call itself: what the kernel can know of the hardware (workers, warp width, shared memory), and where it reports failures |

Since the arguments live in groups, any name is free for them, `inputs` or `name` included.

**A `mutable` is in two groups.** `x = loom.mutable( x )` is `args.inputs.x` (the old value) and
`args.outputs.x` (the new one). In the adjoint, it is `args.grad_of_outputs.x` and
`args.grad_of_inputs.x`.

**A scalar is read as a value.** A Python number is `args.inputs.dt`, a `TF` or an `SI`, with no
coordinates. `TF` is the real type of the call, `SI` its integer type.

**An aggregate keeps its structure.** `args.inputs.splats.centers` is the `centers` tensor of the
`splats` aggregate. An aggregate written in part (`writes`, `reads`) is in `args.outputs` as a
whole; its members that are not written are read there like inputs.

**The adjoint mirrors the roles.** The group names always say the role **in the forward call**: the
adjoint reads `grad_of_outputs`, the gradient of what the forward wrote, and writes
`grad_of_inputs`, the gradient of what it read. See [The adjoint](./adjoint).

## Next

[The adjoint](./adjoint) · [Sizes known at run time](./dynamic-sizes) · [Axes and coordinates](./axes)
