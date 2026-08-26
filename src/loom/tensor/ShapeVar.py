# from typing_extensions import Sequence, overload
from ..util.Attribute import Attribute, resolve_attribute
from typing import TYPE_CHECKING, Iterator
from ..drivers.driver import driver
from numpy._typing import ArrayLike
import weakref
import numpy


class ShapeVar( Attribute ):
    """An integer (possibly rank > 0) variable used to build shapes.

    Rank 0 is a plain scalar count; rank > 0 holds the per-row/per-segment counts
    of a RAGGED structure (`dep_axes` records which axes it varies along).

    Axis extents are affine *expressions* over `ShapeVar`s (e.g. `nb_dims + 1`).
    A `ShapeVar` is either prescribed (`c.nb_dims = 1222`) or solved from the
    shapes observed on the declared tensors that use it; a prescribed value wins.
    `c.nb_dims` reads back the value.

    Sharing: pass a `ShapeVar` to several constructors (`Cell( nb_dims = n )`);
    they then reference the same object, so the value is solved from the union of
    their tensors.

    `value` reads the count back on the HOST (a `ShapeArray`), because sizing is what a count
    is for; `as_tensor()` is the way to ask for it as a device value instead.

    The value a `ShapeVar` holds is a COUNT: how many items are used. A kernel
    reads it and/or writes it, so after a call it lives in a DEVICE buffer -- under
    `jit`, Python does not know it, and it can therefore never size anything.

    What sizes a buffer is a CAPACITY, and a capacity is deliberately NOT state
    kept here: it is a decision about ONE allocation, so it belongs to the call
    that allocates (`driver.call( ..., capacities = { "cell.nb_vertices": 8 } )`).
    Storing it on the object would open a window where the object contradicts
    itself -- a capacity of 64 next to a buffer of 8. What we can offer is what we
    KNOW: `allocated_capacity` (read off the buffers our tensors already have) and
    `static_count` (a count Python actually holds). See `CallArgsAnalysis`.
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: int ) -> None: ...
        def __iter__( self ) -> Iterator: ...

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_ShapeVar import CallArg_ShapeVar
        return CallArg_ShapeVar( caa, path, name, inst )

    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        from .AbstractAxis import AbstractAxis

        # (weakref(tensor), logical, capacity): two resolvers per using tensor. `logical(t)` inverts
        # our affine on the tensor's LOGICAL sizes (`t.reference_shape`, the unpadded truth); `capacity(t)`
        # on its ALLOCATED buffer (padded). Either returns None when it cannot yet. Weak
        # refs so a shared ShapeVar does not keep dropped instances' tensors alive.
        self.usages = []
        self._compact_usages_at = 8   # see `add_usage`
        self.dep_axes = [ resolve_attribute( d, scope, AbstractAxis ) for d in template_args ]

        self.prescribed_value = None
        self._count = None     # count produced by a kernel: a driver tensor, possibly traced

        if value is not None:
            self.set( value )

    def add_usage( self, tensor, logical, capacity ):
        # DERIVED tensors register too (see `Tensor._wrap_axes`), and a loop makes a great many of
        # them -- so dead entries are dropped as they accumulate instead of piling up for every
        # `_pull` to walk. Amortized: compact only once the list has doubled since the last time.
        if len( self.usages ) >= self._compact_usages_at:
            self.usages = [ u for u in self.usages if u[ 0 ]() is not None ]
            self._compact_usages_at = max( 8, 2 * len( self.usages ) )
        # appended LAST, so a declared witness is still the first `_pull` consults.
        self.usages.append( ( weakref.ref( tensor ), logical, capacity ) )

    def set_count( self, value ):
        """Rebind the count to what a kernel produced (a driver tensor). Nothing else moves:
        the buffers keep the size they were allocated with."""
        self._count = value

    def accept_capacity( self, capacity ):
        """Called when a call is given a capacity for us: our chance to refuse one that would
        contradict what we are (see `CtShapeVar`). Nothing is stored -- a capacity belongs to
        the call that allocates."""
        pass

    def set( self, value ):
        if isinstance( value, ShapeVar ):
            value = value.value
        self.prescribed_value = numpy.array( value, dtype = int )

    def _pull( self, kind ):
        """Solve our count from the tensors that use us: the FIRST usage able to invert one of its
        sizes, `None` if none can. `kind` picks which sizes -- the `"logical"` ones (`t.reference_shape`,
        the count) or the `"capacity"` ones (the allocation a chained call must reuse).
        First-that-answers, so a witness carrying the RIGHT rank (a ragged tensor holding per-segment
        counts) is reached the same way as a scalar one -- both invert this ShapeVar's own affine."""
        for tensor_ref, logical, capacity in self.usages:
            tensor = tensor_ref()
            if tensor is None:
                continue
            solved = ( logical if kind == "logical" else capacity )( tensor )
            if solved is not None:
                return numpy.asarray( solved )
        return None

    @property
    def raw( self ) -> ArrayLike:
        """The raw COUNT as a backend / numpy array, or `None` while UNRESOLVED -- the same role
        `Tensor.raw` plays: the backend buffer behind the nice object. A kernel-written count wins,
        being the freshest truth (and it is then a DEVICE value, never size a buffer with it); then
        a user prescription; then what the LOGICAL sizes of a using tensor solve to (`_pull`).

        Users read `value` (a host `ShapeArray`); `raw` is the escape hatch to the backend array
        (what the shape math and the FFI read, without wrapping) -- and it is where a count that
        only lives on the device is still reachable, `as_tensor()` being its tidy form."""
        if self._count is not None:
            return self._count

        if self.prescribed_value is not None:
            return self.prescribed_value

        return self._pull( "logical" )

    @property
    def value( self ):
        """The count on the HOST, as a `ShapeArray` -- rank 0 for a plain scalar count, rank > 0
        for a ragged one (its `dep_axes` name the dimensions). `None` while unresolved.

        Host, not device, and that is the point: a count is what SIZES things (an allocation, a
        `range`, an XLA shape), and only a value Python actually holds can do that. Reading one
        that lives on the device is refused HERE, with a message saying so, rather than handed
        back as a tracer that fails much later in whatever tried to size something with it.

        When the device value IS what you want -- to compute with it in a kernel or in backend
        algebra -- ask for it: `as_tensor()`."""
        raw = self.raw
        if raw is None:
            return None
        from .ShapeArray import ShapeArray
        return ShapeArray( raw, names = [ ax.name for ax in self.dep_axes ] )

    @value.setter
    def value( self, value ):
        self.set( value )

    def as_tensor( self ):
        """The count as a DEVICE value (an `IntTensor`), for the cases that genuinely compute with
        it on the backend rather than size something with it. `None` while unresolved.

        No dtype is claimed: a count buffer knows what it is, and it is NOT uniformly the driver's
        itype -- a kernel-written one is `int32` by design (see `CallArg_ShapeVar`), a prescribed
        one is numpy's int. `wrap` reads it off the buffer instead of labelling it."""
        raw = self.raw
        if raw is None:
            return None
        from .Tensor import Tensor
        return Tensor.wrap( raw, names = [ ax.name for ax in self.dep_axes ] )

    @property
    def max( self ) -> int:
        return driver.max( self.raw )

    def allocated_capacity( self ):
        """The capacity our tensors were ALLOCATED with, read off their buffers -- a fact, not
        a decision, so a chained call needs no restating of a capacity already materialized.
        `None` when we have no allocated tensor to read it from."""
        solved = self._pull( "capacity" )
        return int( numpy.max( solved ) ) if solved is not None else None

    def static_count( self ):
        """Our count when PYTHON holds it (prescribed, or solved from the LOGICAL sizes of a tensor
        we were given). `None` when it only lives on the device -- where it cannot size anything."""
        if self.prescribed_value is not None:
            return int( numpy.max( self.prescribed_value ) )
        solved = self._pull( "logical" )
        return int( numpy.max( solved ) ) if solved is not None else None
