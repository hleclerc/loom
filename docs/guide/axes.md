# Axes and coordinates

A loom kernel does not index by position: it reads a value **at a point**, moves **along an axis**,
and asks **which axes it has**. It never counts dimensions, so the same body works in 1-D, 2-D or
3-D, batched or not.

Axes live on both sides of a call: **Python** declares them, and solves their extents from the
arrays; **C++** reads, writes and moves along them. This page goes through one example, then each
side in turn.

## On one example

The sum of each row of a matrix. In Python, the two axes are named, and the output has only one of
them:

```python
def row_sums( m ):
    row = loom.Axis( loom.ShapeVar(), name = "row" )
    col = loom.Axis( loom.ShapeVar(), name = "col" )
    return loom.ffi_call( "row_sums", _forward,
        m = loom.RealTensor[ row, col ]( m ),
        s = loom.out( loom.RealTensor[ row ]() ),
    )
```

Given a `( 2, 3 )` array, loom solves the two `ShapeVar`s (2 rows, 3 columns), and allocates `s`
with 2 items, because `s` and `m` share the **same** `row` object.

In C++, the kernel is launched once per row, and reads `m` at that row:

```cpp
struct RowSum {
    HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
        auto line = args.inputs.m( coords );      // `coords` has a `row`, no `col`: a line
        TF s = 0;
        for ( SI k = 0; k < line.size( col ); ++k )
            s += line( col = k );
        args.outputs.s( coords ) = s;
    }
};

void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
    queue.run_parallel( RowSum(), batch_axes + args.inputs.m.domain( row ), args, batch_axes );
}
```

`row` and `col` are C++ objects: the names given in Python. `m.domain( row )` is the set of points
of the launch, one per row; `batch_axes + …` adds the axes of a batch, if there is one.

**Batches.** The same function, unchanged, takes a stack of matrices:

```python
jax.vmap( row_sums )( ms )           # Jax
torch.func.vmap( row_sums )( ms )    # Torch
```

Both add an axis to the arrays, and loom passes it to the kernel in `batch_axes`: one launch, over
one more axis. The kernel never sees it, except in the line that chooses the domain. A batch can also
be made [by hand](#batches-by-hand), with any framework.

## In Python: declaring axes

### Nothing to declare

A kernel that only uses anonymous axes needs no declaration at all. An array given as it is has
anonymous axes, known by their position (`axis_0`, `axis_1`, …), and an output built with
`loom.RealTensor.like( x )` takes the axes of `x`. That is the case of
[your first kernel](./first-kernel) and of the [Diffusion](/tutorials/diffusion) step, which takes
a `( ny, nx )` grid or an `( nz, ny, nx )` one.

### Named axes

Naming axes is what lets them play different roles: a row and a column, a point and a coordinate.

- `loom.Axis( extent, name = "row" )` declares an axis. Its name is also its name in C++.
- `loom.RealTensor[ row, col ]( m )` gives these axes to the array `m`, in order.
- `loom.RealTensor[ row ]()` declares a tensor with no content: as an output, loom sizes it.

**An axis is an identity.** Two tensors that share an `Axis` object share its extent: there is
nothing to keep in agreement. Two axes created apart are two axes, even with the same name.

### Extents

| | |
|---|---|
| `loom.ShapeVar()` | a count known at run time, solved from the arrays or prescribed |
| `loom.CtShapeVar` | a count known at compile time: it must be prescribed, and each value compiles its own kernel |
| `"nb_dims + 1"` | an affine expression, which loom can **invert**: a size of 4 gives `nb_dims = 3` |
| `"nb_dims * ( nb_dims + 1 ) / 2"` | a formula, evaluated but not inverted |

The expressions are written as strings in an `Aggregate` (below), where their names are resolved.
Two families need more than one extent:

- **Ragged axes.** A `ShapeVar` with axes of its own (one count per row) gives each row its own
  length. See [Sizes known at run time](./dynamic-sizes).
- **A rank that follows the data.** `loom.AxisList[ "num_dim", "nb_pixels" ]` is one axis per item
  of `num_dim` (one side per dimension of an image). A tensor declared over it has as many axes as
  the image has dimensions.

### Aggregates

When several tensors share axes and counts, they are declared together in an `Aggregate`: a class
whose fields are the tensors, the axes and the counts of one object. From the
[Gaussian splatting](/tutorials/splats) tutorial:

```python
class Splats( loom.Aggregate ):
    centers     : loom.RealTensor[ "num_splat", "num_dim" ]
    cov_inv     : loom.RealTensor[ "num_splat", "num_coeff" ]
    opacities   : loom.RealTensor[ "num_splat" ]

    num_splat   : loom.Axis[ "nb_splats" ]
    num_dim     : loom.Axis[ "nb_dims" ]
    num_coeff   : loom.Axis[ "nb_dims * ( nb_dims + 1 ) / 2" ]

    nb_splats   : loom.ShapeVar
    nb_dims     : loom.CtShapeVar
```

The name of a field is the name of the axis. An aggregate passed to `loom.ffi_call` crosses as
one argument, whose tensors the kernel reads as `args.inputs.splats.centers`.

### Batches by hand

`jax.vmap` and `torch.func.vmap` make a batch from the outside. Without them (with NumPy, CuPy, or
to choose the batch yourself), a **batch axis** does the same thing:

```python
b = loom.new_batch_axis( len( ms ) )
s = loom.ffi_call( "row_sums", _forward,
    m = loom.RealTensor[ b, row, col ]( ms ),       # a stack of matrices
    s = loom.out( loom.RealTensor[ b, row ]() ),    # a stack of sums
)
```

A tensor declared over a batch axis makes the call batched over it, and the kernel receives it in
`batch_axes`, exactly as from a `vmap`. The same kernel runs unchanged. An aggregate is batched with
the `batch_axes = [ b ]` argument of its constructor, which puts `b` in front of all its tensors.

## In C++: using axes

### Axes are objects

Each axis of the call is a C++ object with the name it has in Python: `row`, `col`, `num_splat`, or
`axis_0`, `axis_1` for anonymous ones. Each one is a distinct type, so everything that takes an
axis is resolved at compile time.

### The three primitives

Inside `operator()( coords, args, batch_axes )`:

| | |
|---|---|
| `coords[ axis ]` | the index along `axis`, **by name** and not by position |
| `coords + axis`, `coords - axis` | the neighbour along `axis`; the other coordinates do not move, the batch ones included |
| `coords.axes - batch_axes` | the axes of the point that are yours, by set difference |

With `for_each` and `any_of`, which walk a list of axes and are unrolled at compile time, this is a
5-point Laplacian in any number of dimensions:

```cpp
HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
    const auto main_axes = coords.axes - batch_axes;     // my axes: not the ones of a batch

    const TF uc = args.inputs.u( coords );
    TF total = 0;
    for_each( main_axes, [&]( auto axis ) {
        total += args.inputs.u( coords + axis ) + args.inputs.u( coords - axis ) - 2 * uc;
    } );
    args.outputs.out( coords ) = uc + args.inputs.coef * total;
}
```

`t.size( axis )` is the extent of `t` along `axis`. A test for the boundary, still without a
dimension:

```cpp
any_of( main_axes, [&]( auto axis ) {
    return coords[ axis ] == 0 || coords[ axis ] + 1 == args.inputs.u.size( axis );
} );
```

When the extent comes from a `CtShapeVar`, `size( axis )` is a compile-time constant:
`DECAYED_TYPE_OF( t.size( axis ) )::value`.

### Reading at a point, and views

`t( coords )` reads `t` at a point. When the point covers all the axes of `t`, it is a scalar.
Otherwise it is a view on the axes left, indexed by name with `view( axis = k )`: the `line` of
`RowSum` above.

`t( coords, axis = k )` reads `t` at the point, with `axis` set to `k`. Axes of the point that `t`
does not have are ignored. That is how a kernel launched over the items of one tensor reads another
one at a computed index, while keeping the batch coordinates of the point:

```cpp
args.outputs.y( coords ) = args.inputs.x( coords, num = args.inputs.idx( coords ) );
```

### The domain of a launch

`queue.run_parallel( functor, domain, args, batch_axes )` calls the functor once per point of
`domain`. The domain is the parallelism of the kernel, and you choose it: one point per element, per
row, per particle, per pixel. It is read off a tensor:

| | |
|---|---|
| `t.domain()` | every axis of `t`, batch axes included: one point per element |
| `t.domain( row )` | only the axes named |
| `t.domain( t.axes - num_channel )` | all but some: one point per pixel, not per pixel *and* channel |
| `batch_axes + d` | `d`, plus the axes of a batch |

`t.domain()` already carries the batch axes, since they are axes of `t`. A domain picked by name
does not, so it is completed with `batch_axes + …`. Forgetting it is not a silent mistake: in a
batched call, the output read at a point without its batch coordinate is no longer a scalar, and the
kernel does not compile.

### Why a batch changes nothing in the body

A batch adds axes to the tensors of the call, and loom passes them in `batch_axes`. The body:

- reads values **at a point**, and the point carries the batch coordinates along;
- moves **along its own axes**, which `coords + axis` does without touching the others;
- iterates over `coords.axes - batch_axes`, which excludes the batch axes by construction.

So the kernel never sees the batch, except in the one place where it chooses the domain. Nested
`vmap`s, or several batch axes, add several axes, and `batch_axes` holds all of them.

## Next

[Roles of the arguments](./arguments) · [The adjoint](./adjoint) · [Sizes known at run time](./dynamic-sizes)
