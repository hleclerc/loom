# Tensors

A loom tensor is a JAX or Torch array plus the two things you would otherwise carry in your head:
**what each dimension means**, and **how many items are really in it**.

That turns out to buy quite a lot. Operations line up by meaning instead of by position, so you stop
writing transposes and `None`-indexing. Jagged data — meshes, per-angle point sets, variable-length
segments — is a first-class shape rather than something you pad by hand and then remember to mask.
And when a kernel produces fewer items than the buffer it was given, the tensor knows it.

```python
from loom import RealTensor, Axis

i = Axis( "i" )
x = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )
y = RealTensor[ i ]( [ 4.0, 5.0, 6.0 ] )

x + y                  # elementwise, along i
x.dot( y, "i" )        # 32.0 — contracted over i, no axis order assumed
```

---

## Axes that carry meaning

Declare a tensor with an extent and you get an anonymous axis — enough for plain array work:

```python
t = RealTensor( [ 1.0, 2.0, 3.0 ] )                              # no declared axis
m = RealTensor[ 2, 3 ]( [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ] )
```

Name an axis and pass **the same axis object** to several tensors, and operations start matching by
meaning. A name is all it takes — the extent is then solved from whatever tensor uses the axis:

```python
i = Axis( "i" )
j = Axis( "j" )

a = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )
b = RealTensor[ j ]( [ 1.0, 10.0 ] )

a.shape                # [ 3 ] — read from the value
( a * b ).shape        # [ 3, 2 ] — axes ( i, j ), in a canonical order
```

An axis one side does not have is broadcast, so an outer product needs no reshaping. Pin an extent
up front instead when you want it fixed: `Axis( 3, name = "i" )`.

Because the *object* is what matches, `a * b` and `b * a` give the same layout, and two axes that
merely happen to share a name stay distinct — a name is for you to read, never what the machinery
compares. Reductions and slicing take names too:

```python
m = RealTensor[ i, j ]( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ] )

m.sum( "i" )
m[ "i", 0 ]            # select along a named axis
```

### Slicing: a window is not a new dimension

`vertices[ 10:20 ]` is not a fresh dimension, and it is not `num_vertex` either — it is
**`num_vertex + 10`**: the same dimension, read from another origin. Loom keeps that offset, and it
is what makes slicing safe:

```python
v = RealTensor[ num_vertex ]( [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ] )
w = RealTensor[ num_vertex ]( [ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0 ] )

v[ 2:4 ] * w[ 2:4 ]     # [ 40.0, 90.0 ] — same window, so elementwise
v[ 2:4 ]                # RealTensor( shape=[2], axes=['num_vertex+2'], … )

v[ 2:4 ] * w[ 0:2 ]     # ValueError — two different windows of one dimension
```

That last one is refused rather than broadcast. Two windows of the same dimension are not
independent axes to outer-product (they index the same thing), and they are not aligned either
(position 2 is not position 0) — so pairing them silently is the one outcome that must not happen.
Slice both sides the same way, or give the result an axis of its own.

An axis therefore answers two different questions, and only one of them survives a slice unchanged:

| question | answered by | across a slice |
| --- | --- | --- |
| *which* dimension is this? | the axis | unchanged — `v[ 2:4 ].sum( "num_vertex" )` works |
| do our positions *correspond*? | the axis **and its window** | changed — a different window is a different coordinate |

The size goes with the window, so the shared dimension is never taught a narrowed count:
`v[ 2:4 ].shape` is `[ 2 ]` while `v.shape` stays `[ 6 ]`. Offsets compose through nested slices,
and an index *array* (rather than a slice) is not an affine window at all, so it yields a dimension
of its own.

A window keeps its bounds as **expressions**, not as frozen numbers — so an open-ended slice really
does end wherever the dimension ends:

```python
v[ 2:4 ]      # lo = 2,  hi = 4                 — both literal
v[ 2:  ]      # lo = 2,  hi = nb_vertex         — ends where the dimension ends
v[ :-1 ]      # lo = 0,  hi = nb_vertex - 1     — one short of it
```

Which means inference flows back *through* a window. Four items starting at 2 tells the dimension it
holds six, with nothing said twice:

```python
tail = RealTensor[ num_vertex.windowed( slice( 2, None ) ) ]( [ 9.0, 9.0, 9.0, 9.0 ] )

nb_vertex.value       # 6  — because the window's extent is `nb_vertex - 2`
```

(A dimension reads its count off the tensors that use it, and it holds them weakly — so the witness
has to be alive to be asked. Keep it in a variable, as real code does anyway.)

A window whose bounds are literal (`v[ 2:4 ]`) constrains nothing and says nothing; neither does a
stepped one, since a sampled view does not determine what it was sampled from.

::: tip Naming convention
An axis that indexes over *things* is `num_<thing>` — `num_cell`, `num_vertex`, `num_angle` — which
reads the way it is used (`positions( num_vertex = k )`). Short single letters (`i`, `dim`, `xy`) are
for index-algebra style, where the axis is a coordinate rather than a collection. Both are just
names; pick whichever makes the kernel body read best.
:::

::: tip Prefer `dot` over `@`
`@` contracts positionally, so it assumes an axis order. `x.dot( y, "i" )` is
`( x * y ).sum( "i" )` — it lines the shared axis up by identity and needs no order at all.
:::

A tensor with **no** declared axes falls back to positional numpy broadcasting, which is convenient
but is where surprises live: `RealTensor([1.,2.,3.]) + RealTensor([[1.],[2.],[3.]])` is `[3, 3]`, as
numpy would have it.

---

## Operations

Every operation exists **both as a method and as a free function** — `loom.dot( a, b, "i" )` *is*
`a.dot( b, "i" )`. Neither is the real one: some expressions read better chained
(`t.sum( "i" ).sqrt()`), others called (`dot( normals, points, "xy" )`), and a pipeline of free
functions composes where a method chain does not.

| what | method | function |
| --- | --- | --- |
| arithmetic | `a + b`, `-`, `*`, `/`, `//`, `%`, `**` | — |
| comparison | `a > b`, `==`, `!=`, `<`, `<=`, `>=` → a `BoolTensor` | — |
| contraction | `a.dot( b, over )` | `dot( a, b, over )` |
| positional matmul | `a @ b` | — |
| select on a mask | `cond.where( a, b )` | `where( cond, a, b )` |
| reductions | `t.sum( axis )`, `.prod`, `.min`, `.max`, `.mean`, `.all`, `.any` | `sum( t, axis )`, `prod`, `min`, `max`, `mean`, `all`, `any` |
| elementwise maps | `t.sqrt()`, `t.arcsin()`, `abs( t )`, `t.clip( lo, hi )`, `t.stop_gradient()` | `sqrt( t )`, `arcsin( t )`, `abs( t )`, `clip( t, lo, hi )`, `stop_gradient( t )` |
| permutation | `t.transpose( *axes )`, `t.T` | `transpose( t, *axes )` |
| indexing | `t[ 0 ]`, `t[ :, 1 ]`, `t[ "i", k ]` | — |
| conversion | `numpy.asarray( t )`, `float( t )`, `int( t )`, `len( t )`, `for v in t` | — |

`axis` is `None` (reduce everything), an axis name, a position, or a tuple of those. The free
reductions shadow python builtins, exactly as numpy's do — reach them through the module
(`loom.sum( t, "i" )`) rather than importing them bare.

A comparison gives a `BoolTensor`, and `where` is what you do with it. The three operands align by
axis identity like any elementwise op, so a per-row condition selects across a whole matrix with no
reshaping:

```python
import loom

mask = x > 0                                   # BoolTensor over i
loom.where( mask, x, 0.0 )                     # a scalar branch broadcasts
loom.where( mask, m, 0.0 )                     # m is [ i, j ] — the mask spreads over j
```

::: info Not there yet
No `exp` / `log` / `sin` / `cos`, no `stack` / `concat`, no `argmin` / `argmax` / `cumsum`. They are
missing rather than deliberately excluded — say which you need.
:::

---

## What it is made of

There is no `dtype=` keyword to remember — the class says it:

```python
from loom import RealTensor, IntTensor, BoolTensor

positions = RealTensor[ 3, 2 ]()       # differentiable
indices   = IntTensor [ 3 ]()          # not: no gradient flows through an index
mask      = positions > 0              # BoolTensor, produced by the comparison
```

The **size** (FP32 vs FP64, int32 vs int64) is deliberately not part of the declaration: it is a
driver-wide policy (`driver.ftype`, `driver.itype`, or the `SDOT_FTYPE` / `SDOT_ITYPE` environment
variables), resolved when you actually run. So the same model runs in single or double precision
without editing a line. Pin one only where it genuinely matters:

```python
RealTensor[ 3, dict( size = 32 ) ]()
```

The declared type is enforced, not decorative. Widening is silent; a conversion that would lose
meaning is refused rather than done behind your back:

```python
idx = IntTensor[ 3 ]()
idx.set( [ True, False, True ] )      # fine — nothing is lost
idx.set( [ 1.5, 2.5, 3.5 ] )          # TypeError: the conversion loses a fractional part
```

Results follow the operation rather than the operand, so `int_tensor / 2` is real and a comparison
is boolean — no silently mislabelled arrays downstream.

---

## Ragged tensors

Rows of different lengths are a shape, not a workaround. Declare an axis whose extent varies along
another one, and assign jagged data directly:

```python
from loom import Aggregate, Axis, ShapeVar, RealTensor

class Mesh( Aggregate ):
    cell_vertices    : RealTensor[ "num_cell", "num_vertex" ]

    num_cell         : Axis[ "nb_cells" ]
    num_vertex       : Axis[ "nb_vertices_per_cell" ]   # ragged: varies along `num_cell`

    nb_cells             : ShapeVar
    nb_vertices_per_cell : ShapeVar[ "num_cell" ]       # one count PER cell

m = Mesh()
m.cell_vertices = [ [ 1.0, 2.0, 3.0 ], [ 4.0 ], [ 5.0, 6.0 ] ]
```

The structure is read from the nesting alone — without touching the data:

```python
m.nb_cells.value               # 3
m.nb_vertices_per_cell.value   # ShapeArray( [ 3, 1, 2 ], axes = [ 'num_cell' ] )
```

Underneath, the values are assembled into one padded buffer (that is what a GPU wants), but the
padding is bookkeeping, not something you have to think about. **Reductions know about the holes:**

```python
t = m.cell_vertices

t.sum ( "num_vertex" )     # [  6.0,  4.0, 11.0 ]
t.max ( "num_vertex" )     # [  3.0,  4.0,  6.0 ]   — not 0 from a padded cell
t.mean( "num_vertex" )     # [  2.0,  4.0,  5.5 ]   — divided by the REAL count, not the box
t.prod( "num_vertex" )     # [  6.0,  4.0, 30.0 ]   — not 0 either
```

Each reduction fills the holes with its own identity first, so a `max` never returns a padding zero
and a `mean` never divides by the bounding box. Printing shows the real shape:

```
RealTensor( shape=[3, 3], axes=['num_cell', 'num_vertex'], dtype=TF, device=Cpu )
1 2 3
4
5 6
```

Assign a differently-shaped nesting and the per-segment counts are simply re-read.

---

## Sizes: what is used, and what is allocated

Two different numbers hide behind the word "size", and keeping them apart is what makes the ragged
and kernel-output cases work.

- a **count** — how many items are actually used. That is what a `ShapeVar` holds.
- a **capacity** — how big the buffer is. A decision made by whoever allocates, which may be
  larger (padding for a ragged row, room for a kernel that does not know its output size in
  advance).

```python
n = ShapeVar()
t = IntTensor[ Axis( n ) ]( [ 5, 6, 7 ] )

n.value          # ShapeArray( 3 ) — the count, read back on the host
t.shape          # [ 3 ] — the logical extents
t.capacity       # ( 3, ) — what was allocated
```

A count comes back as a `ShapeArray`: a plain host value you can size things with.

```python
[ 0.0 ] * n.value            # usable as a size — which is the point
range( int( n.value ) )
```

::: warning Counts live on the host, on purpose
Sizing an allocation, a Python loop or a `numpy.arange` needs a value Python actually holds. Under
`jax.jit`, a count a *kernel* produced is a tracer, and a tracer cannot size anything. `ShapeArray`
refuses to be built from one and says so, at the point where you asked for a host value — instead
of failing much later inside whatever tried to use it. When you genuinely want to compute with a
count on the device, ask for it explicitly: `n.as_tensor()` gives you an `IntTensor`.
:::

Its arithmetic is deliberately small: a count stays a count under `+`, `*`, `//` (`n + 1` is still a
size); a true division is no longer one and hands back plain numpy.

---

## Grouping tensors: aggregates

An `Aggregate` is where axes get names you can reference from several tensors, so a shared dimension
is shared by construction:

```python
class Cell( Aggregate ):
    vertex_positions : RealTensor[ "num_vertex", "dim" ]
    vertex_indices   : IntTensor [ "num_vertex", "dim" ]

    num_vertex       : Axis[ "nb_vertices" ]
    dim              : Axis[ "nb_dims" ]

    nb_vertices      : ShapeVar
    nb_dims          : ShapeVar
```

Assigning any one tensor teaches the aggregate the counts it implies — nothing is declared twice:

```python
c = Cell()
c.vertex_positions = [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]

c.nb_vertices.value    # 3
c.nb_dims.value        # 2
```

Axis extents can be affine expressions of counts (`Axis[ "nb_dims + 1" ]`), and one count can drive
several axes — which is exactly when the `ShapeVar` / `Axis` split earns its keep.

### Sharing a dimension between aggregates

Two `Cell`s mint their own `num_vertex`, so by default they do **not** line up — which is right:
two unrelated meshes both have vertices, and vertex 3 of one is not vertex 3 of the other. When they
really are the same dimension, say so by passing its identity:

```python
from loom.tensor import AxisId

vertex = AxisId( "num_vertex" )

a = Cell( num_vertex = vertex )
b = Cell( num_vertex = vertex )

a.positions * b.positions      # elementwise — the same dimension
```

Each instance still keeps a **count of its own**: an `AxisId` says *which* dimension, and carries no
size to impose. That is the difference with passing an `Axis`, which is a window and so shares the
extent with it. Two instances of different sizes therefore align and then disagree on the shape —
an error, rather than a silent broadcast.

String axis names are resolved *by the aggregate*, so they only work inside one. Standalone, pass
the `Axis` object itself, or use `Tensor.wrap( raw, [ "i", "j" ] )` to attach names to an existing
backend array.

---

## Reading a tensor

| what | you get |
| --- | --- |
| `t.tensor` | the **logical values**, padding cropped — what you usually want |
| `t.shape` | the logical extents |
| `t.capacity` | the allocated extents |
| `t.raw` | the underlying backend buffer, padding included |
| `numpy.asarray( t )`, `float( t )`, `int( t )`, `list( t )` | the usual conversions |

```python
numpy.asarray( c.vertex_positions.tensor )   # the 3x2 you put in
c.vertex_positions.raw                       # the buffer, which may be larger
```

---

## Current limitations

Worth knowing before you hit them:

- **The `ShapeVar` / `Axis` split still shows through.** `Axis( "i" )` mints its own count, so the
  standalone case no longer pays for it up front — but the two concepts remain visible as soon as
  you open an aggregate, where they earn their keep only when one count drives several axes. We are
  still looking for a spelling that scales down better.
- **A literal extent does not adapt.** `RealTensor[ 3 ]` declares 3; assigning two values does not
  currently complain.

---

## Under the hood

How a tensor is backed (a real buffer, an absent value, a symbolic zero, a constant fill) and how it
crosses into a C++ kernel are described in [The FFI boundary](./ffi-internals.md).
