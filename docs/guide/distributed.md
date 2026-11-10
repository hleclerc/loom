# Distributed programs

::: warning Early
Synchronous rounds, the count of boxes a multiple of the count of ranks, out of core on one rank and forward only. The API is meant to stay as it
is while the runtimes behind it grow ( see [What comes next](#what-comes-next) ).
:::

A kernel is written once for every framework. A distributed algorithm should be written once too, for every way of
running it: one process taking the pieces in turn, the devices of a Jax mesh, Torch processes, MPI ranks -- and, later,
pieces that do not all fit in memory. loom gives you **types of programs**: each one says what you declare, and
leaves the rest to a runtime chosen from the operands.

| Type | You write | What travels | For |
|---|---|---|---|
| `loom.spmd` | the program of **one rank**, with collectives | what you exchange yourself | the low level: everything else is built on it |
| `loom.BoxProgram` | the program of **one box**, in rounds | rows from box to box, adds to outputs | algorithms where who talks to whom depends on the data |

More types can come ( blocks with no messages, à la `map_blocks`; pipelines ), on the same ranks and the same
primitives.

## Why boxes, and why rounds

There are three usual ways of declaring a distributed computation, and the more general the declaration, the less a
runtime can do with it:

| Model | What is declared | What can run it |
|---|---|---|
| Split arrays ( `NamedSharding`, `DTensor`, `dask.array` ) | "this axis is split", operations by blocks | every runtime, `jit` included -- but nothing whose messages depend on the data |
| A static task graph ( `dask.delayed` ) | the whole graph, before the run | a scheduler, out of core; a `jit` only when the graph is regular |
| A dynamic graph ( futures, actors ) | nothing in advance | asynchronous runtimes only: no `jit`, no prefetching, no generic adjoint |

A `BoxProgram` sits between the last two. This is the **Pregel** model ( "think like a vertex" ): the data is
cut into **boxes**; in each **round**, every box reads what it received, sends rows to other boxes and adds values to
outputs; the program stops when no row travels. How many rounds, and to whom each row goes, depend on the data. But
every round has the same shape: a map over the boxes, then an exchange. That is what lets one program run on very
different runtimes.

The unit is the box, not the process: a rank holds several boxes, and which rank holds which box is the runtime's
business. That is what will let a runtime balance the load ( give boxes to other ranks ) or keep some of them out of
memory, without changing the program.

## An example: travellers

Box `b` starts `L` travellers. Traveller `( b, i )` makes `hops[ b, i ]` hops of stride `strides[ b, i ]`, adding to its
load the weight of each box it arrives at. Once it has made all its hops, its load is added to `res[ b, i ]`, held by its
home box, wherever the traveller ends. A row is a traveller: `[ home box, home index, hops left, stride, load ]`.

```python
import loom
from loom.tensor.Dtype import Dtype

i32 = Dtype.si( 32 )

def start( box, values, hops, strides, weights ):
    """round 0, on each box: its blocks ( `values[ b ]`... ) and the whole `weights`"""
    o = loom.ops( values )
    fp = o.dtype_of( values )
    idx = o.arange( values.shape[ 0 ], i32 )
    here, none = idx * 0 + box.index, idx * 0 - 1
    rows = o.stack( [ o.astype( x, fp ) for x in ( here, idx, hops, strides, values ) ], axis = 1 )
    going = hops > 0
    return loom.Outbox(
        send = ( o.where( going, ( here + o.astype( strides, i32 ) ) % box.nb_boxes, none ), rows ),
        add = { "res": ( o.where( going, none, here ), idx, values ) },    # ( no hop: home at once )
    )

def step( box, inbox, values, hops, strides, weights ):
    """the next rounds: `inbox.rows` the rows received, `inbox.valid` those that were really sent"""
    o = loom.ops( values )
    r, ok = inbox.rows, inbox.valid
    here = o.astype( r[ :, 0 ], i32 ) * 0 + box.index
    none = here * 0 - 1
    load = r[ :, 4 ] + o.astype( weights, o.dtype_of( values ) )[ box.index ]
    left = r[ :, 2 ] - 1
    go, home = ok & ( left > 0 ), ok & ( left <= 0 )
    return loom.Outbox(
        send = ( o.where( go, ( here + o.astype( r[ :, 3 ], i32 ) ) % box.nb_boxes, none ),
                 o.stack( [ r[ :, 0 ], r[ :, 1 ], left, r[ :, 3 ], load ], axis = 1 ) ),
        add = { "res": ( o.where( home, o.astype( r[ :, 0 ], i32 ), none ), o.astype( r[ :, 1 ], i32 ), load ) },
    )

prog = loom.BoxProgram( start, step, row_width = 5, outputs = { "res": L } )
out = prog( values, hops, strides, shared = ( weights, ) )     # values, hops, strides: [ K, L ]
out[ "res" ]                                                  # [ K, L ]
```

The arguments before `shared` have one block per box along their first axis ( `K` boxes ); `start` and `step` get the
block of their box. The `shared` ones are passed whole to every box. The functions are written with `loom.ops` and loom
kernels, so they are the same code under numpy, Torch and Jax, traced or not.

What a box gives back is an `Outbox`:

* `send = ( dest, rows )`: row `i` goes to box `dest[ i ]` for the next round; a negative box drops it;
* `add = { name: ( box, index, values ) }`: `values[ i ]` is added to `outputs[ name ][ box[ i ], index[ i ] ]`. The adds
  are summed where the output is held: the box can be any box, and this is how a result goes back to its owner.

An inbox has a fixed number of slots ( see [capacities](#capacities) ): `inbox.valid` says which ones were filled; the
others are zeros. `box.index` is the box running ( an int32 array `[ 1 ]` of the framework, a traced value under Jax ),
`box.nb_boxes` their count.

The full example, checked against a numpy loop on every runtime, is `tests/box_program_cases.py`.

### Boxes of different sizes

An argument split along the boxes can also be a **list**, one value per box, of any shape ( or any kind: a tuple, a dict
of arrays ). `start` and `step` get the value of their box, so boxes of different sizes need nothing more. Every rank is
given the whole list and keeps its boxes. Jax devices refuse lists: one trace serves every device, so their boxes must
have one shape -- pad them, or give arrays `[ K, ... ]`.

### A row width that follows a capacity

`row_width` can be a function, read at every run. When the width of a row depends on something the program learns
( the number of items a row carries ), make it a `loom.Capacity` reported by `step`, and `row_width = lambda: 3 + 2 *
cap.value`: an overflow runs the program again with wider rows.

## Where it runs

The program does not say where it runs. The operands say it, as they say which framework runs a kernel:

1. `ranks =` given to the call;
2. else the ranks of a spread argument: `loom.spread( x, ranks )`, or a Jax array sharded along its first axis;
3. else `loom.default_ranks`;
4. else what the process was started in ( `loom.detect_ranks()` ): the world of `torch.distributed` once initialised,
   or an MPI launcher -- `mpiexec`, `srun` -- seen in the environment, with `mpi4py` installed;
5. else one rank.

A Jax process with several devices is never guessed to be distributed: it runs on the devices of the mesh its arrays
are sharded on, or on one device. An MPI run, on the other hand, shows nothing in the data, hence the look at the
environment. When nothing is spread but there are several ranks, every rank is taken to hold the whole arrays, and keeps
its boxes.

| Ranks | One round is | The loop |
|---|---|---|
| one rank ( `LocalRanks` ) | the `K` boxes in turn; the exchange is the identity | a Python loop, or a `lax.while_loop` under Jax |
| `JaxRanks( mesh )` | the boxes of each device, in a `shard_map`; `all_to_all` | `lax.while_loop`, everything under one `jit` |
| `TorchRanks()` | the boxes of the process; `all_to_all_single` | a Python loop in each process |
| `MpiRanks()` | the boxes of the process; `Alltoall` ( CUDA-aware for CuPy ) | a Python loop in each process |
| out of core ( one rank ) | the boxes **with mail**, loaded one by one; the rows wait on the host | a Python loop, outside any `jit` |

With `P` ranks, rank `p` holds boxes `p K / P` to `( p + 1 ) K / P - 1`. The outputs come back spread the same way: a rank
of Torch or MPI holds its boxes' blocks, a Jax output is one sharded array.

Under the hood a round is a `loom.spmd` program: the boxes of the rank one after the other, then `loom.route` packs the
rows by destination box, `Ranks.exchange` sends them to the ranks that hold those boxes, and `loom.scatter_add` adds what
came back to the outputs. These are loom kernels: the same code for every framework, inside a `jit`.

## Out of core

When the boxes do not all fit in memory, give them as a **store**: `loom.BoxStore( load, nb_boxes )`, where `load( b )`
returns the value of box `b` on the host ( an array, or tuples / dicts of them ). A zarr or a dask array, or anything
indexable along its first axis wrapped by `loom.BoxStore.of( x )`, does as well. Or keep plain arrays and pass
`max_resident = M`.

```python
store = loom.BoxStore( lambda b: read_block( b ), K )       # e.g. a file per box
out = prog( store, hops, strides, shared = ( weights, ), max_resident = 2 )
```

The rounds are then driven from Python. In each round, only the boxes that received rows run; a box is read from its
store when it is needed, moved to the default framework, and kept among the `max_resident` most recent ones ( 2 by
default: the one running and the next one, which a thread reads ahead ). Rows and adds wait on the host, at their exact
size: the send and add capacities play no part, only those the program reports itself. The outputs are made on the host
and come back as tensors of the default framework.

`prog.last_loads` counts the reads of the last call, `prog.last_rounds` its rounds.

## Capacities

A round cannot know in advance how many rows a rank will send to a box. The buffers are sized by `loom.Capacity`s:

* `send_capacity`: the rows one rank sends to one box in one round;
* `add_capacity`: the adds of one output one rank sends to one box in one round;
* any capacity your own `start` / `step` reports ( `capacity.report( overflow )` ), for what they size themselves.

As for the capacities of a kernel, an overflow is not an error: the whole program runs again with that capacity doubled,
and the `BoxProgram` keeps it for its next calls. Start them small in a test, so that the rerun is tested too.

## The gradient

Under Jax and Torch, a `BoxProgram` is differentiable with respect to the real arguments, split along the boxes or shared, as long
as `start` and `step` are ( loom kernels with their adjoint, framework operations ):

```python
def loss( values, weights ):
    out = prog( values, hops, strides, shared = ( weights, ) )
    return jnp.sum( out[ "res" ].value * C )

g_values, g_weights = jax.grad( loss, argnums = ( 0, 1 ) )( values, weights )
```

Nothing is written for it. When differentiated, the forward keeps the inbox of every box in every round; the backward replays the
rounds from the last one, each box run again under the framework's `vjp`:

* the cotangent of a row comes back from the box that received it, by the exchange in reverse ( the same `route` and `exchange` );
* the cotangent of an add is asked for at the box that holds the output, like a row, and answered the same way;
* what a box gets for its inbox goes back to the boxes that sent it, one round earlier.

Rows that say where things go ( a destination, an index carried as a real ) get a cotangent like any other column; the program
only uses them through integers, so nothing flows through them.

The rounds the tape holds are a capacity, `prog.tape_rounds`: a longer run goes again with room for twice as many. Under Jax devices
the gradient is that of the whole ( a shared value's gradient summed over the devices ). Over Torch processes, each rank gets what its
boxes contribute, as for any computation split between processes: its blocks, and for what is whole on every rank its own copy's
share -- sum them with `all_reduce` if you need the total. Under `jax.grad`, a sharded array is traced: mark it with `loom.spread`
or pass `ranks =`. Numpy has no autodiff, and out of core is forward only for now.

## What comes next

The model is chosen so that these fit without changing the programs:

* **The adjoint out of core.** The tape kept with the boxes, rounds replayed by loading the boxes again.
* **A tape that is not kept.** Recompute the rounds instead of keeping their inboxes, when memory is short.
* **More boxes than ranks, moved.** The placement of the boxes is already the runtime's: balancing the load is giving
  some boxes to other ranks between calls.
* **Out of core on several ranks.** Each rank driving its own boxes from a store, the rows exchanged between rounds.
* **Asynchronous rounds.** A program whose result does not depend on the order in which a box handles its rows can say
  so; a runtime may then drop the barrier between rounds ( point-to-point sends under Torch and MPI ) and steal boxes
  from slow ranks.
