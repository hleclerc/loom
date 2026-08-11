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
"""
import itertools

from .ShapeVar import ShapeVar
from .Axis import Axis


# fresh, process-unique names, so two independent batch axes never collide in `DEFINE_AXIS` /
# `global_batch_indices`. Sharing an axis is done by REUSING its object, not by matching a name.
_batch_counter = itertools.count()


def new_batch_axis( size ):
    """A fresh batch `Axis` of extent `size`: a private `ShapeVar` prescribed to `size`, wrapped in
    an `Axis` with a process-unique name. Pass a list of these as `batch_axes = [ ... ]` to any
    aggregate constructor; reuse one object across aggregates to co-iterate them."""
    axis = Axis( ShapeVar( size ) )
    axis.name = f"batch_{ next( _batch_counter ) }"
    axis.is_batch = True    # sorts first in the logical layout of an elementwise result
    return axis
