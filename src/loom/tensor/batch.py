"""Batching an `@aggregate` over extra leading axes.

Batching is a construction-time option of EVERY aggregate, not a distinct type: `Cell` stays
`Cell`. Passing `batch_axes = [ ax, ... ]` to a constructor (e.g. `Cell.make_hypercube`) adds those
axes on the LEFT of every tensor the aggregate declares. The annotations are the "scalar" schema
and stay untouched; `__base_init__` builds each per-instance `Tensor` with the batch axes prepended
(see `util/aggregate.py`).

On the C++ side a batch is nothing but a leading, NAMED tensor dimension, which the existing kernel
machinery already threads transparently (`global_batch_indices`, `cell( batch_index )`, each member
a template parameter carrying its axis-name types). So batching lives entirely on the Python side.

A batch axis is an ordinary `Axis` over its own `ShapeVar`, whose size is PRESCRIBED (a batch extent
is known in Python, so the outputs can be allocated and displayed without a kernel writing a count).
`new_batch_axis( size )` mints a fresh, unshared one; passing the SAME axis object to several
aggregates is how they get JOINED (co-iterated) instead -- which is opt-in, never the default.

= This name no longer leaves the object: C++ receives it from the CALL

WARNING: this section describes an outdated state, kept because it explains the mechanism below.
The name an axis carries here NO LONGER REACHES THE C++ SOURCE: `CallArgsAnalysis` renames the
batch axes of a call to `batch_0`, `batch_1`, ... in the order they show up in it (see
`CallArgsAnalysis.cpp_axis_name`). This is the definitive remedy that was announced at the bottom
of this docstring, and it is in place.

What follows remains true on a single point, and it still matters: two axes ALIVE AT THE SAME TIME
must carry DISTINCT names, otherwise the by-name collection of `CallArgsAnalysis` would merge them
into one. That is what the pool guarantees.

--- the previous state, and why it was not enough ---

The name went all the way to the C++ source (`DEFINE_AXIS( thread_0 )`, the type `_thread_0` that
each batched tensor carries), and the compilation cache key is the HASH OF THAT SOURCE. A fresh
name on every call therefore meant: two identical calls, two different sources, two compilations
of ~8 s -- and a disk cache that grows without ever being reused. This was not a minor
inefficiency: it made any timing of a loop of calls impossible to read.

The remedy back then was to BORROW the names: taken at the smallest free index of its prefix,
given back when the axis dies. A repeating sequence of calls would then find the same names again,
hence the same source, hence the cache -- but ONLY if the axes died between two calls. As soon as
a CHAIN keeps them all alive at once (ten steps of `examples/diffusion`, which the adjoint has to
walk back up), each one takes a different index and everything is paid for again: THIRTY kernels
measured (`LOOM_JOURNAL=1`) of which twenty-eight differed only by `cell_0` ... `cell_19`,
and it grew linearly with the length of the chain. Renaming at lowering brought them back to 3.

The PREFIX separates the families (`thread_0` for work-items, `cell_0` for cells): it stays
readable in a Python display and isolates the numberings, but it no longer shows in the generated
source.

= The `gc.collect()` is a SAFETY NET, and it should never be needed anymore

Borrowing assumes giving back, and giving back assumes the axis DIES. It was not dying: two rings
of references held it, and a ring is only undone by the cyclic garbage collector.

  * `CallArgsAnalysis <-> CallArg_*` -- the lowering tree, whose every tensor node held its
    analysis back (`CallArg_Tensor._caa`). This was THE culprit: it also held the aggregates of
    the call, hence their device buffers, long after the call ended. The back reference is now
    weak.
  * `Axis -> coeffs -> ShapeVar -> usages -> resolver -> Axis` -- the resolvers of
    `AbstractAxis._register_dense` captured their axis and their per-argument count through a
    default argument. Weak too now (the reference to the TENSOR already was).

Verified by disabling the cyclic garbage collector: axes are given back by plain reference
counting. The `gc.collect()` below only fires if the reserve is empty -- that is, at the very
moment a compilation would be paid for -- and should therefore never have anything left to give
back. It remains as a safety net: a ring reintroduced elsewhere would cost 9 ms instead of 9 s,
and `in_use( prefix )` would tell.

= The definitive remedy: IN PLACE

Name the axes at LOWERING time -- `CallArgsAnalysis` knows the exact set of axes of THIS call and
numbers them in a deterministic order -- so that there is no lifetime left to track to get the
cache.

The blocker announced here ("a batch axis can reach a call through a BARE tensor, so a pre-pass
would be needed") was not one: the analysis only collects `batch_axes` from AGGREGATE arguments,
and a bare tensor carrying a batch axis is simply ignored -- the kernel then receives an empty
`batch_index` and fails with a C++ `static_assert` (see `loom/examples/diffusion/README.md`,
friction 3). The set is therefore known before the first `CallArg` is built, and the renaming
happens there.

What remains for the pool is its other role, which has not moved: distinct names between axes
alive at the same time.
"""
import gc
import threading
import weakref

from .ShapeVar import ShapeVar
from .Axis import Axis


class _NamePool:
    """The BORROWABLE indices of a prefix: the smallest free one is reused before creating a new one.

    Taking the smallest, and not the last given back, is what makes the sequence REPRODUCIBLE:
    two runs of the same program borrow in the same order, however the borrows overlapped.
    """

    # collect before creating a fresh name. Downside: a caller making MANY very short calls
    # that would rather pay for the compilations. See the module docstring.
    collect_when_empty = True

    def __init__( self ):
        self._lock = threading.Lock()
        self._free = {}                 # prefix -> set of given-back indices
        self._next = {}                 # prefix -> the next index never handed out

    def take( self, prefix ):
        index = self._take( prefix )
        if index is not None:
            return index

        # the reserve is empty: this is where, and only where, we are about to pay for a
        # compilation. A cyclic collection gives back the axes of previous calls (the shapes graph
        # is cyclic, they do not go away by themselves) -- and if it gives back none, they are
        # really alive, and we create the name.
        if self.collect_when_empty:
            gc.collect()
            index = self._take( prefix )
            if index is not None:
                return index

        with self._lock:
            index = self._next.get( prefix, 0 )
            self._next[ prefix ] = index + 1
            return index

    def _take( self, prefix ):
        """The smallest free index, or `None` if there is none."""
        with self._lock:
            free = self._free.get( prefix )
            if not free:
                return None
            index = min( free )
            free.discard( index )
            return index

    def give_back( self, prefix, index ):
        with self._lock:
            self._free.setdefault( prefix, set() ).add( index )

    def in_use( self, prefix ):
        """How many indices of this prefix are out -- for tests, and to diagnose a leak (an axis
        held somewhere) without having to guess."""
        with self._lock:
            return self._next.get( prefix, 0 ) - len( self._free.get( prefix, () ) )


_pool = _NamePool()


def new_batch_axis( size, prefix = "batch" ):
    """A fresh batch `Axis` of extent `size`: a private `ShapeVar` prescribed to `size`, wrapped in
    an `Axis` whose name is BORROWED from `prefix`'s pool (see the module docstring: the name is a
    resource, and reusing it is what lets two identical calls hit the compilation cache). Pass a
    list of these as `batch_axes = [ ... ]` to any aggregate constructor; reuse one object across
    aggregates to co-iterate them.

    `prefix` should say what the axis IS (`"thread"`, `"cell"`, `"angle"`), which is what the
    generated C++ will then read like. It also isolates the numbering of one family from another.
    """
    index = _pool.take( prefix )

    axis = Axis( ShapeVar( size ) )
    axis.name = f"{ prefix }_{ index }"
    axis.is_batch = True    # sorts first in the logical layout of an elementwise result

    # given back when the axis dies -- a `finalize` and not a `__del__`: it does not hold the axis (it
    # only captures the prefix and the index), so it does not postpone its death by a GC cycle.
    weakref.finalize( axis, _pool.give_back, prefix, index )

    return axis
