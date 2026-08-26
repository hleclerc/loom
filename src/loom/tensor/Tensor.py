from typing import TYPE_CHECKING

import numpy
from numpy.typing import ArrayLike

from ..devices.Device import Device
from ..drivers.driver import driver
from ..util.Attribute import Attribute, resolve_attribute
from .AbstractAxis import AbstractAxis
from .Axis import Axis
from .AxisList import AxisList
from .Dtype import Dtype
from .ReferenceShape import ReferenceShape
from .ShapeVar import ShapeVar
from .storage import Fill, Storage, Unbound


class Tensor( Attribute ):
    """
    Tensor declaration: a thin wrapper around the backend tensor of the chosen
    library (Jax, Torch, ...).

        t = RealTensor( 17 )                    # rank 0, no declared axis
        t = IntTensor( [ 1, 2 ] )
        t = RealTensor[ x, y ]( [ [ 1, 2 ] ] )  # x, y being `Axis` objects

    `Tensor` itself is the ABSTRACT base: what it declares (the axes, the storage, the ops) is
    common to every element type. What the element type decides -- whether a gradient flows
    through it, which C++ scalar it lowers to, what an op promotes to -- belongs to the concrete
    subclasses `RealTensor` / `IntTensor` / `BoolTensor`, one per `Dtype` kind.

    Building a `Tensor` directly still works and yields the class its declared dtype calls for
    (`Tensor[ { "dtype": int } ]` is an `IntTensor`), so the two spellings coexist during the
    migration -- but `Tensor` is meant to hold no instances of its own.

    The logical contract is the axis list; axis extents may depend on other axes
    (RAGGED axes), in which case the varying sizes live in the `ShapeVar`s of
    rank > 0. With no declared axis at all, there is nothing to solve and the
    buffer IS the contract (see `shape`). The physical contract (padding / order
    / alignment, per device) is to come, as template kwargs, kept separate from
    the axis list.

    Attributes
    * axes    : list[ AbstractAxis ] -- the LOGICAL contract
    * dtype   : Dtype -- the ELEMENT contract (its kind is the class, its size a driver policy)
    * storage : how the value is actually BACKED (`storage.py`) -- nothing, a real buffer, a
        symbolic zero, a fill. It answers every physical question (`raw`, `tensor`, `capacity`,
        `buffer_layout`, `allocated_sizes`), so this class holds no kind flags and no separate
        `_raw` / `_layout` fields to keep in agreement.
    * device  : Device

    The storage also carries the `reference_shape`: the LOGICAL sizes read from the value ALONE,
    one entry per array dimension -- unpadded, and independent of the declared axes. The buffer may
    be padded, so it only serves the CAPACITY; the reference shape serves the COUNT, and a
    `ShapeVar` PULLS from it via the axis layout (see `AbstractAxis.register_in`).
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: ArrayLike | None ) -> None: ...

    # ---- the ELEMENT contract, carried by the CLASS -------------------------------------------
    # `dtype_kinds` is what a subclass accepts (`None` on the abstract base); `is_differentiable`
    # is whether a gradient can flow through it -- the predicate the FFI asks to know what is a
    # primal. Both are answered by the type, not tested for on the dtype at each site.
    dtype_kinds: tuple | None = None
    is_differentiable = True

    # kind -> concrete class, filled by `__init_subclass__`. A dtype is all it takes to know which
    # tensor class a buffer calls for, so nothing anywhere has to enumerate them.
    _by_kind: dict = {}

    def __init_subclass__( cls, **kwargs ):
        super().__init_subclass__( **kwargs )
        for kind in cls.dtype_kinds or ():
            Tensor._by_kind[ kind ] = cls

    @staticmethod
    def class_for( dtype ) -> type:
        """The concrete tensor class a `Dtype` calls for. Used wherever a tensor is built around a
        buffer that already exists (`wrap`, an op result): the buffer's type picks the class, so a
        comparison on a `RealTensor` comes back a `BoolTensor` without anyone saying so."""
        kind = Dtype.factory( dtype ).kind
        if kind not in Tensor._by_kind:
            from . import BoolTensor, IntTensor, RealTensor   # registers them (see __init_subclass__)
        return Tensor._by_kind[ kind ]

    @classmethod
    def default_dtype( cls, size = None ) -> Dtype:
        """The dtype this class means when a declaration names none. The abstract base has no
        element type of its own, so it defers to the real one -- the historical default."""
        from .RealTensor import RealTensor
        return RealTensor.default_dtype( size )

    def __new__( cls, *args, **kwargs ):
        # `Tensor( ... )` builds the concrete class its declared dtype calls for -- so the old
        # spelling keeps working while declarations migrate to `RealTensor` / `IntTensor`. A
        # concrete class builds itself (this is not a dispatcher its subclasses go through).
        if cls is not Tensor:
            return object.__new__( cls )
        return object.__new__( Tensor.class_for( _declared_dtype( cls, kwargs.get( "template_kwargs", {} ) ) ) )

    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        self.device = Device.factory( template_kwargs.get( "device", None ) )
        self.dtype = _declared_dtype( type( self ), template_kwargs )
        self.axes = self._read_axes( template_args, scope )

        # HOW our value is held (see `storage.py`): one object per way a value can be backed --
        # nothing, a real buffer (possibly with an explicit physical layout), a symbolic zero, a
        # fill. It answers every physical question below, so this class carries no kind flags.
        self.storage = Unbound()

        if value is not None:
            self.set( value )

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_Tensor import CallArg_Tensor
        return CallArg_Tensor( caa, path, name, inst )

    @classmethod
    def like( cls, other ) -> "Tensor":
        """An empty tensor sharing `other`'s axes, dtype and device -- what the backward pass
        builds to carry a gradient (same logical shape as the value it is the gradient of). What
        goes INTO it then decides its kind: a real cotangent buffer (`set_raw`) makes it a
        `TensorView`, a symbolic-zero cotangent a `ZeroTensor`, nothing at all a `NoneTensor`."""
        return Tensor.class_for( other.dtype )( template_args = other.axes,
                                                 template_kwargs = { "dtype": other.dtype, "device": other.device } )

    @classmethod
    def full( cls, value, *, template_args = (), template_kwargs = {}, scope = None ) -> "Tensor":
        """A tensor filled with `value` everywhere, built straight from its axes
        (`Tensor[ axis ].full( v )`) -- no shape to pass: `shape` already reads it off
        the axes, the same way it would for any other tensor built on them."""
        res = cls( template_args = template_args, template_kwargs = template_kwargs, scope = scope )
        fill = value.tensor if isinstance( value, Tensor ) else value
        shape = res.shape
        # NB: the SYMBOLIC-fill path (a storageless C++ `FillTensor`, i.e. a `Fill` storage over a
        # scalar) exists (see FillTensor.h and `storage.Fill`) but is NOT yet wired in: it needs the
        # fill to survive the symbolic algebra it flows through -- the `target/mass * weights` of a
        # second `normalized_version` (a `c * fill` must stay a fill), and the backward RESIDUAL rank.
        # Until then `full` MATERIALIZES: `[ n ]` weights are 80MB, already small (the [nb_angles, n]
        # blow-up is fixed by the shared `[ num_dirac ]` shape upstream).
        res.storage = Storage.of( driver.full( shape, fill, dtype = res.dtype ),
                                  ReferenceShape.from_dense_shape( shape ) )
        return res

    @classmethod
    def filled_with( cls, scalar, reference_shape = None, *, template_args = (), template_kwargs = {}, scope = None ) -> "Tensor":
        """The SYMBOLIC form of `full`: one scalar backing the whole logical shape, lowered as a
        storageless `FillTensor`. Being a fill is STATED here, never inferred -- one scalar looks
        like any other rank-0 buffer.

        `reference_shape` defaults to the extents the axes already give (as in `full`); it is
        passed explicitly where those are not resolved -- a backward residual, whose logical extents
        the C++ view reads from a sibling buffer at run time anyway."""
        res = cls( template_args = template_args, template_kwargs = template_kwargs, scope = scope )
        if reference_shape is None:
            reference_shape = ReferenceShape.from_dense_shape( res.shape )
        # the scalar is materialized as a rank-0 BUFFER: that is what the FFI binds (a `FillTensor`
        # is storageless in its extents, not in its value), and it is what carries our dtype.
        res.storage = Fill( driver.array( scalar, dtype = res.dtype, device = res.device ), reference_shape )
        return res

    def append_axis( self, axis ):
        """A new tensor sharing our buffer, with `axis` appended as one extra TRAILING axis --
        e.g. `SumOfDiracs1d.positions` (rank 1) reused as `SumOfDiracs`'s ( `num_dirac`, `dim` )
        positions once `dim`'s extent is known (see `SumOfDiracs1d.normalized_version`). Unset
        (unbound) stays unset: with no buffer yet, there is nothing to reshape."""
        res = type( self )( template_args = self.axes + [ axis ], template_kwargs = { "dtype": self.dtype, "device": self.device } )
        if self.is_defined:
            res.storage = Storage.of( self.storage.raw[ ..., None ],
                                      self.reference_shape.appended_dense( 1 ) )
        return res

    # ---- what our value IS: asked of the storage, never tested for here -------------------------
    @property
    def is_symbolic_zero( self ) -> bool:
        return self.storage.is_symbolic_zero

    @property
    def is_fill( self ) -> bool:
        """A symbolic constant: one scalar backs the whole logical shape, which lives in the axes,
        and it lowers to a storageless `FillTensor` (see `storage.Fill`)."""
        return self.storage.is_fill

    @property
    def reference_shape( self ):
        """The LOGICAL (unpadded) sizes read off the value we were built from -- one entry per array
        dimension, independent of our declared axes. `None` when there is none to read (a kernel
        output, whose counts live in its ShapeVars instead). This is what a `ShapeVar` PULLS its
        count from, via the axis layout (see `AbstractAxis.register_in`)."""
        return self.storage.reference_shape

    @property
    def buffer_rank( self ):
        """How many array dimensions our BUFFER has, or `None` while unbound -- the capacity-side
        analogue of `len( reference_shape )`, used to resolve an unrolled `AxisList`'s width."""
        buffer = self.storage.buffer
        return None if buffer is None else buffer.ndim

    @property
    def is_undefined( self ) -> bool:
        """Nothing to read here yet -- the spelling `Distribution.mass` and `Image.cell_cum_mass`
        use to decide whether to (re)compute a value they cache."""
        return not self.is_defined

    @property
    def is_defined( self ) -> bool:
        return self.storage.holds_value

    def set( self, value ):
        if isinstance( value, Tensor ):
            # take the source's STORAGE whole: its kind rides along (a fill stays a fill, a symbolic
            # zero stays one) and so does its reference shape -- it describes the very buffer we just
            # took, so it is our logical shape too, whatever our own axes are. What does NOT ride
            # along is the element type: our declared dtype is a contract the FFI reads
            # (`CallArg_Tensor._cpp_scalar`), so a buffer that disagrees would be reinterpreted, not
            # converted, in C++ -- hence `retyped`, which enforces it on whatever backs the value.
            self.storage = value.storage.retyped( self._as_declared )
            return

        # The LOGICAL shape is a pure fact about `value` (see `ReferenceShape`), read WITHOUT touching
        # its data. From it every ShapeVar PULLS its count (`register_in`); nothing is pushed. The
        # buffer is then a plain dense array, or -- when the value is jagged -- an ASSEMBLED padded
        # one whose per-dim capacity is (for now) the max size.
        reference_shape = ReferenceShape.from_value( value )
        self._check_convertible( _natural_dtype( value ) )
        if reference_shape.is_ragged():
            raw = _assemble( value, reference_shape.capacities(), self.dtype, self.device )
        else:
            raw = driver.array( value, dtype = self.dtype, device = self.device )
        self.storage = Storage.of( raw, reference_shape )

    def set_raw( self, raw, layout = None ):
        """Bind the buffer a kernel produced (a driver tensor). Sizes stay unobserved: an
        output's extents are the ones we ASKED for (`shape`), and its counts live in the
        ShapeVars the kernel wrote -- there is nothing to solve from the data, so the reference
        shape we already had (if any) is kept rather than re-read.

        `raw` may also be a symbolic-zero cotangent handed back by the framework, or `None` to
        unbind: `Storage.of` picks the variant, so there is no kind to test for here. `layout` is
        the physical arrangement the allocation was given, when it is not the plain contiguous one.
        """
        self.storage = Storage.of( self._as_declared( raw ), self.reference_shape, layout )

    # ---- the dtype INVARIANT: what we declare is what our buffer holds ---------------------------
    # A `Tensor`'s dtype is a DECLARATION, and `CallArg_Tensor` lowers it as the C++ scalar type of
    # the buffer it binds. So a buffer whose element type disagrees is not merely mislabelled: the
    # kernel reinterprets its bytes. The declaration is therefore enforced at every point where a
    # buffer becomes ours (`set`, `set_raw`), and a DERIVED tensor -- which declares nothing -- reads
    # its dtype off the buffer instead (`wrap` / `_wrap_axes`).
    def _as_declared( self, raw ):
        """`raw` retyped to our declared dtype. A widening conversion (bool/int -> real, a size
        change) is silent; a LOSING one is refused rather than performed behind the user's back."""
        if raw is None or driver.is_symbolic_zero( raw ):
            return raw                        # no storage to retype (a symbolic zero carries its own)
        have = Dtype.of( raw )
        if self.dtype.same_as( have ):
            return raw
        self._check_convertible( have )
        return driver.astype( raw, self.dtype )

    def _check_convertible( self, have ):
        """Raise if a value of dtype `have` cannot become our declared dtype without losing what it
        means. `have` is `None` when the value has no dtype of its own to read (a python list, a
        ragged nesting) -- there is then nothing to contradict."""
        if have is None or self.dtype.same_as( have ):
            return
        lost = ( "a fractional part" if have.floating_point and not self.dtype.floating_point else
                 "every value but 0 and 1" if self.dtype.boolean and not have.boolean else None )
        if lost is not None:
            raise TypeError(
                f"cannot bind a { have.cpp_name } value to a { self.dtype.cpp_name } tensor"
                f"{ '' if self.name is None else f' ({ self.name })' }: the conversion loses { lost }. "
                f"Convert explicitly if that is what you mean." )

    # ---- physical questions, all delegated to the storage (see `storage.py`) --------------------
    # Each variant answers them its own way -- a fill's capacity is its reference shape, an unbound
    # tensor has no allocation to report -- so none of it is a branch here.
    @property
    def buffer_layout( self ):
        """How our buffer is laid out physically vs the logical axes: the explicit layout if the
        allocation was given one, else the plain CONTIGUOUS one derived from the allocated extents.
        The single place `capacity`, the ShapeVar capacity inversion and the FFI lowering read the
        buffer's shape/strides from -- unifying padding (ragged / batch) and physical order."""
        return self.storage.layout( self.rank )

    @property
    def capacity( self ):
        """What our buffer IS: the allocated extents per LOGICAL dimension. An input is bound at THIS
        size -- an output that wants to grow must not force us to inflate the input."""
        return self.storage.capacity( self.rank )

    @property
    def allocated_sizes( self ):
        """The allocated capacity per LOGICAL dimension -- what a `ShapeVar` inverts to learn the
        capacity it was allocated with. Read through the layout, so a non-contiguous / flattened /
        padded buffer answers with the per-axis capacity rather than the raw physical extents.
        `None` when there is no allocation to invert (unbound, or a fill -- whose extents come FROM
        real sibling buffers, never the reverse)."""
        return self.storage.allocated_sizes( self.rank )


    def _read_axes( self, axes, scope ):
        res = []
        for entry in axes:
            if isinstance( entry, str ):
                if entry.endswith( "..." ):
                    entry = entry[ :-3 ]
                entry = resolve_attribute( entry, scope )
            elif isinstance( entry, ( ShapeVar, int ) ):
                # a `ShapeVar` (or a literal extent, `RealTensor[ 3, 4 ]`) names no axis of its
                # own: wrap it in an anonymous one, which is all a standalone tensor needs.
                entry = Axis( entry )

            assert isinstance( entry, AbstractAxis )
            entry.register_in( self )
            res.append( entry )

        return res

    def _dim_index( self, axis ):
        """The position of `axis` among our declared axes, by IDENTITY -- so an axis registered in us
        can find itself without the caller passing an index (see `AbstractAxis.register_in`)."""
        for i, a in enumerate( self.axes ):
            if a is axis:
                return i
        raise ValueError( "axis not declared in this tensor" )

    # ---- per-dimension geometry, DERIVED from the axes + `_raw` (no cached field) ----
    def _axis_array_dims( self, axis ):
        """How many ARRAY dimensions `axis` occupies here -- delegated to the axis itself (1 for a
        plain `Axis`, the unroll width for an `AxisList`). The tensor holds no unroll arithmetic:
        an `AxisList` gets its width from its loop axis, or as a last resort from this buffer's rank
        (see `AbstractAxis.array_dims` / `AxisList.array_dims`)."""
        return axis.array_dims( self )

    def _spec_dims( self ):
        """The first array dimension of each declared axis (accounting for an unrolled AxisList
        sibling spanning several). Derived on the fly from the axes' dimension counts."""
        dims, d = [], 0
        for axis in self.axes:
            dims.append( d )
            d += self._axis_array_dims( axis )
        return dims

    def _unroll_span( self, index ):
        """`( start, count )` of the unrolled `AxisList` at `index`, or `None` while its count is
        not yet knowable. Both come from the axis (`array_dims`) and the derived layout."""
        count = self._axis_array_dims( self.axes[ index ] )
        return None if count is None else ( self._spec_dims()[ index ], count )

    def _has_unroll( self ):
        return any( _is_unrolled( a ) for a in self.axes )

    @property
    def shape( self ):
        # with no declared axis there is no expression to evaluate: the buffer is the whole contract
        # (a standalone `Tensor( [ 1, 2 ] )`, or a derived tensor with no named axis), and it has
        # none while unvalued. `_raw.shape` also serves a symbolic zero (it carries its shape).
        if not self.axes:
            return list( self.storage.raw.shape ) if self.is_defined else []

        # each member contributes a LIST of extents (one for an `Axis`, `nb_dims`
        # for an unrolled `AxisList`); concatenation gives the tensor's extents.
        res = []
        for axis in self.axes:
            res += axis.max_list()
        return res

    @property
    def rank( self ):
        if not self.axes:
            return self.storage.raw.ndim if self.is_defined else 0
        return len( self.axes )

    @property
    def raw( self ):
        """The MATERIALIZED buffer, or `None` when there is nothing to bind -- a symbolic zero has
        no storage, so it reads as `None` here (that is how `is_bound` stays false for it), while
        the storage still holds the framework's zero object for `is_symbolic_zero` to recognize."""
        return self.storage.buffer

    @property
    def tensor( self ):
        """The dense VIEW of `raw`: its logical region, with the capacity padding cropped off.

        `raw` is a homogeneous buffer sized at CAPACITY -- padding included -- because that is what
        a kernel writes into; `tensor` slices it back to the logical `shape`, which is what one
        usually wants to read (`c.vertex_positions.tensor` instead of `c.vertex_positions.raw[ :n ]`).

        Meaningful for a DENSE (non-ragged) tensor: a ragged one has no single box to extract, so
        this returns its bounding box (inner padding kept). Needs a statically known `shape`, so it
        holds eagerly -- a kernel-written count is a device value under a trace, where Python cannot
        slice by it (`shape` raises there). A symbolic zero has no buffer to view -> `None`.

        How the crop is actually done depends on how the value is BACKED (a plain slice, a gather
        through a non-contiguous layout, ...), so it is the storage that answers -- this property is
        just the name everything else reads."""
        return self.storage.view( self )

    # @property
    # def value( self ):
    #     """The LOGICAL data, backend array: `tensor` (padding cropped), which is what one wants
    #     when reading a tensor as a value -- the same role `c.nb_dims.value` plays for a `ShapeVar`.
    #     `raw` stays the padded buffer the FFI needs."""
    #     return self.tensor

    # @value.setter
    # def value( self, value ):
    #     self.set( value )

    # ------------------------------------------------------------------ derived tensors
    # Every op below (operators, reductions, slicing) reads the LOGICAL values (`self.tensor`,
    # padding cropped) and returns a fresh DERIVED tensor built by `_wrap`. A derived tensor is no
    # special case: it carries a real list of `AbstractAxis` like any other. Each surviving dimension
    # gets a fresh DEFAULT axis -- a plain `Axis` over a new `ShapeVar`, observed straight from the
    # (dense) buffer, carrying the inherited name if any. So `shape` / `_dim_names` derive uniformly
    # from the axes, named reductions / slices keep working down the chain, and a partial slice is
    # correct (the fresh axis holds the sliced size, not the original's stale one).

    def _wrap( self, raw, names ):
        # no dtype passed on purpose: `raw` is the result of an OP, and the op decides the type
        # (a comparison yields booleans, an integer division reals). `wrap` reads it off the buffer.
        return type( self ).wrap( raw, names, device = self.device )

    @classmethod
    def wrap( cls, raw, names = None, dtype = None, device = None ):
        """A DETACHED tensor around an existing backend buffer: no aggregate, fresh DEFAULT axes
        (one per array dimension, named from `names`, observed straight from the dense buffer). This
        is how a `ShapeVar` hands its count back as a `Tensor`, and the FRESH-axis path for an op
        result that keeps no axis identity (`matmul`) -- the buffer is the whole contract, the axes
        just carry the names. An op that DOES preserve identity uses `_wrap_axes` instead."""
        # a wrapped buffer DECLARES nothing -- it already exists -- so its dtype is READ off it.
        # An explicit `dtype` is then a claim about that buffer, checked rather than believed.
        dt  = _wrapped_dtype( raw, dtype ) or cls.default_dtype()
        res = Tensor.class_for( dt )( template_kwargs = { "dtype": dt, "device": device } )
        res.storage = Storage.of( raw )
        if names is not None:
            axes = []
            for index, name in enumerate( names ):
                axis = Axis( ShapeVar() )
                axis.name = name
                axes.append( axis )
            res.axes = axes
            for axis in axes:
                axis.register_in( res )
        # a wrapped buffer is dense: its LOGICAL sizes ARE its shape (no padding). Record them so the
        # fresh ShapeVars pull their counts, AND so an axis-less result (no names, e.g. a `matmul`)
        # still has a reference shape to reshape from (`append_axis`). A symbolic zero carries none.
        if res.storage.buffer is not None:
            res.storage = res.storage.with_reference_shape( ReferenceShape.from_dense_shape( raw.shape ) )
        return res

    def _dim_names( self ):
        """One axis name (or `None`) per ARRAY dimension, read uniformly off the axes (an unrolled
        AxisList spreads its name over its spanned dimensions). The plain NAME, so a window into a
        dimension is still found by it -- `_dim_labels` is the form that shows the offset."""
        return self._dim_attr( lambda axis: axis.name )

    def _dim_labels( self ):
        """One display label per array dimension: the name, plus the offset when this is a WINDOW
        into a dimension rather than the whole of it (`num_vertex+10`)."""
        return self._dim_attr( lambda axis: axis.display_name )

    def _dim_batch( self ):
        """One `is_batch` flag per ARRAY dimension (same layout as `_dim_names`)."""
        return self._dim_attr( lambda axis: getattr( axis, "is_batch", False ) )

    def _dim_axes( self ):
        """One axis OBJECT per array dimension (an unrolled AxisList repeated over the dims it
        spans). The per-reference analogue of `_dim_names` -- what an elementwise op aligns on."""
        return self._dim_attr( lambda axis: axis )

    def _dim_attr( self, of ):
        res = []
        for axis in self.axes:
            if _is_unrolled( axis ):
                count = self._axis_array_dims( axis )
                count = count if count is not None else len( axis.max_list() )
                res += [ of( axis ) ] * count
            else:
                res.append( of( axis ) )
        return res

    def _axis_pos( self, key ):
        """A dimension index, from an axis OBJECT (matched by IDENTITY -- so a sliced dimension is
        still found by the axis it means), an int (returned as is), or an axis NAME."""
        if isinstance( key, AbstractAxis ):
            for i, a in enumerate( self._dim_axes() ):
                if a.identity is key.identity:
                    return i
            raise ValueError( "axis object not among this tensor's dimensions" )
        if isinstance( key, str ):
            names = self._dim_names()
            if key not in names:
                raise ValueError( f"no axis named '{ key }' in { names }" )
            return names.index( key )
        return int( key )

    # ---- array protocol: makes `numpy.asarray(t)`, `int(t)`, `list(t)`, `assert t == x` work ----
    def __array__( self, dtype = None ):
        arr = numpy.asarray( self.tensor )
        return arr.astype( dtype ) if dtype is not None else arr

    def __int__( self ):
        return int( numpy.asarray( self.tensor ) )

    def __float__( self ):
        return float( numpy.asarray( self.tensor ) )

    def __bool__( self ):
        return bool( numpy.asarray( self.tensor ) )

    def __len__( self ):
        if self.rank == 0:
            raise TypeError( "len() of a rank-0 tensor" )   # numpy's convention for a 0-d value
        return self.shape[ 0 ]

    def __iter__( self ):
        for i in range( len( self ) ):
            yield self[ i ]

    # ---- elementwise operators -------------------------------------------------------------------
    # An op is a MAP over axes matched BY REFERENCE -- the axis OBJECT itself, not its name (names are
    # optional in Python and fragile; two homonymous but independent axes are genuinely distinct). A
    # SHARED axis (the same object threaded through several tensors, e.g. a batch axis) lines up; an
    # axis only one side has is broadcast (size 1 on the other). The result's layout is canonical --
    # batch axes first (`Axis.is_batch`), then first-seen -- so `a * b` and `b * a` agree and a
    # per-batch value spreads over a full batched tensor with no reshape. Needs each ARRAY dim to map
    # to a DISTINCT axis object; a bare `Tensor( array )` (no axes) or a multi-dim `AxisList` (one
    # object over several dims) falls back to positional broadcasting. `@` is not here: it contracts.
    def _binary( self, other, op ):
        # a raw (non-`Tensor`) operand carries its OWN dtype (e.g. a numpy float64 constant computed
        # host-side), which would otherwise promote the op's result away from `self.dtype` (jax's
        # numpy-style promotion, notably FP32 + F64 -> F64 once x64 is enabled) -- coerced upfront so
        # a REAL tensor stays in its own precision whatever a python constant is spelled as.
        #
        # Only a real one: coercing on an INTEGER (or boolean) tensor changes what the op MEANS
        # rather than merely its precision -- `idx == 0.5` would compare against 0, and `idx / 2`
        # would be asked to answer in integers. There the backend's own promotion is the right
        # answer, and the result's dtype is read off the buffer it produces (see `_result_dtype`).
        if self.dtype.floating_point and not isinstance( other, Tensor ) and not driver.is_symbolic_zero( other ):
            other = driver.array( other, dtype = self.dtype, device = self.device )
        la = self._ref_layout()
        if la is not None:
            if isinstance( other, Tensor ):
                lb = other._ref_layout()
                if lb is not None:
                    return self._ref_binary( other, la, lb, op )
            else:
                raw = op( self.tensor, other )
                if getattr( raw, "shape", () ) == getattr( self.tensor, "shape", () ):
                    return self._wrap_axes( raw, la )   # scalar/array broadcast: our axes survive
        b = other.tensor if isinstance( other, Tensor ) else other
        names = self._dim_names()
        return self._wrap( op( self.tensor, b ), names if len( names ) == self.rank else None )

    def _ref_layout( self ):
        """One axis OBJECT per array dimension when each dim means a DISTINCT axis (so a map by
        reference is unambiguous), else `None`. A rank-0 tensor qualifies with an empty list."""
        axes = self._dim_axes()
        if len( axes ) != self.rank:
            return None
        for i, a in enumerate( axes ):
            if any( a.coordinate == b.coordinate for b in axes[ :i ] ):
                return None
        return axes

    def _ref_binary( self, other, la, lb, op ):
        return self._ref_apply( [ other ], op, layouts = [ la, lb ] )

    def _ref_apply( self, others, op, layouts = None ):
        """Apply `op` to us and `others`, all aligned onto ONE canonical axis order -- the n-ary
        form of a reference-mapped op (`_binary` is the binary case, `where` the ternary one).

        The order is canonical BY IDENTITY: batch axes first, then first-seen (us, then each other
        in turn), so the result does not depend on the order the operands were written in. A
        non-`Tensor` operand becomes a rank-0 tensor, which carries no axis and therefore broadcasts
        over all of them. `None` when an operand cannot be mapped by reference (a bare array with
        several dimensions and no distinct axis per dimension) -- the caller then falls back to
        positional broadcasting."""
        operands = [ self ] + [ o if isinstance( o, Tensor ) else Tensor.wrap( driver.array( o ) )
                                for o in others ]
        if layouts is None:
            layouts = [ t._ref_layout() for t in operands ]
            if any( l is None for l in layouts ):
                return None

        order = []
        for want_batch in ( True, False ):
            for dims in layouts:
                for ax in dims:
                    if bool( getattr( ax, "is_batch", False ) ) != want_batch:
                        continue
                    if any( ax.coordinate == o.coordinate for o in order ):
                        continue
                    _refuse_mismatched_window( ax, order )
                    order.append( ax )
        raw = op( *[ _aligned_to( t.tensor, l, order ) for t, l in zip( operands, layouts ) ] )
        return self._wrap_axes( raw, order )

    def where( self, a, b ):
        """`a` where we are true, `b` elsewhere -- what a mask (`t > 0`, a `BoolTensor`) is FOR.

        The three operands are aligned by axis IDENTITY like any elementwise op, so a per-row
        condition selects across a full matrix with no reshaping. Either branch may be a plain
        scalar, which broadcasts."""
        res = self._ref_apply( [ a, b ], driver.where )
        if res is not None:
            return res
        unwrap = lambda v: v.tensor if isinstance( v, Tensor ) else v
        return self._wrap( driver.where( self.tensor, unwrap( a ), unwrap( b ) ), self._dim_names() )

    def _wrap_axes( self, raw, dim_axes ):
        """A detached tensor around `raw` whose dimensions ARE `dim_axes` -- the very axis OBJECTS, so
        identity (hence names and `is_batch`) rides along into the next op. Consecutive repeats (an
        `AxisList` over several dims) collapse to one entry.

        The result REGISTERS as a usage of those axes. Without it an axis reads its extent only off
        the tensors that DECLARED it, and those are held weakly -- so chaining off a temporary
        (`( a * b ).shape`) could find the extent gone once the temporary was collected. Registering
        is only sound because every op reaching here preserves its axes' extents; one that narrows a
        dimension (a partial slice) hands over a fresh axis instead (see `__getitem__`)."""
        axes = []
        for ax in dim_axes:
            if not axes or axes[ -1 ] is not ax:
                axes.append( ax )
        # dtype read off `raw` for the same reason as in `wrap`: this is a result, not a declaration.
        dt  = _result_dtype( raw, self.dtype )
        res = Tensor.class_for( dt )( template_kwargs = { "dtype": dt, "device": self.device } )
        res.axes = axes
        res.storage = Storage.of( raw )
        if res.storage.buffer is not None:
            res.storage = res.storage.with_reference_shape( ReferenceShape.from_dense_shape( raw.shape ) )
            for axis in axes:
                axis.register_in( res )
        return res

    def __add__     ( self, o ): return self._binary( o, lambda a, b: a +  b )
    def __radd__    ( self, o ): return self._binary( o, lambda a, b: b +  a )
    def __sub__     ( self, o ): return self._binary( o, lambda a, b: a -  b )
    def __rsub__    ( self, o ): return self._binary( o, lambda a, b: b -  a )
    def __mul__     ( self, o ): return self._binary( o, lambda a, b: a *  b )
    def __rmul__    ( self, o ): return self._binary( o, lambda a, b: b *  a )
    def __truediv__ ( self, o ): return self._binary( o, lambda a, b: a /  b )
    def __rtruediv__( self, o ): return self._binary( o, lambda a, b: b /  a )
    def __floordiv__( self, o ): return self._binary( o, lambda a, b: a // b )
    def __mod__     ( self, o ): return self._binary( o, lambda a, b: a %  b )
    def __pow__     ( self, o ): return self._binary( o, lambda a, b: a ** b )

    def __matmul__  ( self, o ):
        # matmul CONTRACTS dimensions -- it is not a per-axis map, so it must not go through the
        # ref-aligned path. Positional, and the contracted layout has no meaningful surviving axis
        # identity, so none is carried. Prefer `dot` (contraction BY REFERENCE) over `@`.
        b = o.tensor if isinstance( o, Tensor ) else o
        return self._wrap( self.tensor @ b, None )

    def dot( self, other, over ):
        """Contract with `other` over the SHARED axis `over` -- the reference-based analogue of a
        matmul, and unlike `@` it assumes NO axis order: it is `( self * other ).sum( over )`. The
        elementwise product lines `over` up by IDENTITY (so it must be the SAME axis object on both
        sides) and outer-products the free axes; the sum then contracts `over`. `over` is an axis
        object (or a name / position `sum` accepts). E.g. detector projection = normals `.dot`
        points over the shared coordinate axis, giving `[ angles, points ]` free."""
        return ( self * other ).sum( over )

    def __neg__( self ): return self._wrap_axes( -self.tensor, self._dim_axes() )
    def __abs__( self ): return self._wrap_axes( abs( self.tensor ), self._dim_axes() )

    # ---- elementwise maps: shape (hence every axis OBJECT) is preserved, like `__neg__` ----
    def _map( self, op ):
        return self._wrap_axes( op( self.tensor ), self._dim_axes() )

    def sqrt  ( self ): return self._map( driver.sqrt )
    def arcsin( self ): return self._map( driver.arcsin )

    def clip( self, lo = None, hi = None ):
        """Values clamped to `[ lo, hi ]` (either bound may be `None` = unbounded)."""
        return self._map( lambda a: driver.clip( a, lo, hi ) )

    def stop_gradient( self ):
        """Detached from the gradient tape (`driver.stop_gradient`) -- same values, no derivative
        flows back through them. Used where a quantity is needed for its VALUE only, its derivative
        being supplied by another (better conditioned) route."""
        return self._map( driver.stop_gradient )

    def __eq__( self, o ): return self._binary( o, lambda a, b: a == b )
    def __ne__( self, o ): return self._binary( o, lambda a, b: a != b )
    def __lt__( self, o ): return self._binary( o, lambda a, b: a <  b )
    def __le__( self, o ): return self._binary( o, lambda a, b: a <= b )
    def __gt__( self, o ): return self._binary( o, lambda a, b: a >  b )
    def __ge__( self, o ): return self._binary( o, lambda a, b: a >= b )

    # `__eq__` returns a Tensor (elementwise), so instances are no longer value-comparable as keys;
    # keep IDENTITY hashing (a `Tensor` is never used as a by-value dict key -- `ShapeVar` is).
    __hash__ = object.__hash__

    # ---- axis permutation (numpy-like: positions or axis names; the names follow the move) ----
    def transpose( self, *axes ):
        """A view with the dimensions PERMUTED. No argument reverses the order (`t.T`); otherwise
        each entry is a dimension position or an axis NAME. Reads the LOGICAL values (`self.tensor`)
        and returns a fresh derived tensor, so the surviving names ride along with the permutation."""
        if not axes:
            perm = tuple( reversed( range( self.rank ) ) )
        else:
            if len( axes ) == 1 and isinstance( axes[ 0 ], ( tuple, list ) ):
                axes = tuple( axes[ 0 ] )
            perm = tuple( self._axis_pos( a ) for a in axes )
        dims = self._dim_axes()
        return self._wrap_axes( driver.transpose( self.tensor, perm ),
                                [ dims[ p ] for p in perm if p < len( dims ) ] )

    @property
    def T( self ):
        return self.transpose()

    # ---- reductions (`axis` = None / int / axis name / a tuple of those) ----
    def _reduce( self, op, axis, identity ):
        """Reduce over the LOGICAL values. A ragged tensor's bounding box (`tensor`) has HOLES
        (padding) that would corrupt the result -- a 0 surviving a `max`, a 1 lost in a `prod` -- so
        they are first filled with the operation's IDENTITY. A dense (or unrolled) tensor has no
        holes: `_hole_mask` returns `None` and the fast path is untouched."""
        data  = self.tensor
        holes = self._hole_mask()
        if holes is not None:
            data = driver.where( holes, identity, data )
        if axis is None:
            return self._wrap_axes( op( data ), [] )
        keys = axis if isinstance( axis, ( tuple, list ) ) else ( axis, )
        pos  = tuple( self._axis_pos( k ) for k in keys )
        # the survivors keep their axis OBJECTS, so two reductions of the same tensor (e.g. `sum` and
        # `_valid_counts` in `mean`) share them and line up by reference.
        survivors = [ a for d, a in enumerate( self._dim_axes() ) if d not in pos ]
        return self._wrap_axes( op( data, axis = pos ), survivors )

    def sum ( self, axis = None ): return self._reduce( driver.sum,  axis, 0 )
    def prod( self, axis = None ): return self._reduce( driver.prod, axis, 1 )
    def max ( self, axis = None ): return self._reduce( driver.max,  axis, -numpy.inf )
    def min ( self, axis = None ): return self._reduce( driver.min,  axis,  numpy.inf )
    def all ( self, axis = None ): return self._reduce( driver.all,  axis, True )
    def any ( self, axis = None ): return self._reduce( driver.any,  axis, False )

    def mean( self, axis = None ):
        # holes filled with 0 make the SUM correct; divide by the count of REAL cells, not the box.
        return self.sum( axis ) / self._valid_counts( axis )

    def _valid_counts( self, axis ):
        """A `Tensor` of how many non-hole cells fall along the reduced dims (the whole box when
        dense) -- the denominator `mean` divides by."""
        holes = self._hole_mask()
        valid = numpy.ones( tuple( self.shape ), dtype = int ) if holes is None else ( ~holes ).astype( int )
        if axis is None:
            return self._wrap_axes( driver.array( int( valid.sum() ), dtype = int ), [] )
        keys = axis if isinstance( axis, ( tuple, list ) ) else ( axis, )
        pos  = tuple( self._axis_pos( k ) for k in keys )
        survivors = [ a for d, a in enumerate( self._dim_axes() ) if d not in pos ]
        return self._wrap_axes( driver.array( valid.sum( axis = pos ), dtype = int ), survivors )

    def _hole_mask( self ):
        """A boolean array over the bounding box (`shape`), True at each PADDING position -- the
        holes a ragged tensor leaves inside its box. `None` when there is nothing to mask (a dense
        or unrolled tensor: its box IS its data), so the common path stays allocation-free.

        Built eagerly, cell by cell, from the axes' live extents (like the `__repr__` display): a
        reduction already needs a static `shape`, so an eager mask fits here."""
        if self._has_unroll() or not any( _axis_rank( a ) > 0 for a in self.axes ):
            return None
        shape = tuple( self.shape )
        mask  = numpy.zeros( shape, dtype = bool )
        for idx in numpy.ndindex( *shape ):
            if not _cell_valid( self.axes, idx ):
                mask[ idx ] = True
        return mask

    # ---- indexing: numpy-positional, or ( "axis_name", index ) to select by name ----
    def __getitem__( self, key ):
        dims = self._dim_axes()
        if isinstance( key, tuple ) and len( key ) and isinstance( key[ 0 ], str ):
            name, idx = key
            pos = self._axis_pos( name )
            key = tuple( idx if d == pos else slice( None ) for d in range( self.rank ) )
        elif not isinstance( key, tuple ):
            key = ( key, )
        # pad the trailing dimensions with full slices, then keep the axis OBJECTS of the surviving
        # dimensions (an int index drops its dimension; a slice / array keeps it).
        key = key + ( slice( None ), ) * ( self.rank - len( key ) )
        result = self.tensor[ key ]

        # An axis OBJECT survives only where the slice left its dimension INTACT. An axis is SHARED,
        # and its extent is solved from the tensors that use it (see `_wrap_axes`, which registers
        # this result as one), so handing it a NARROWED dimension would teach it a wrong count --
        # and through it, every other tensor built on the same axis. A narrowed dimension therefore
        # gets a fresh axis, carrying the name so the result still reads the same.
        extents   = self.shape
        survivors = []
        for d, k in enumerate( key ):
            if isinstance( k, int ):
                continue                                    # an int index drops its dimension
            axis = dims[ d ] if d < len( dims ) else None
            window = axis.windowed( k, extents[ d ] ) if isinstance( k, slice ) and axis is not None else None
            if window is None:
                # not an affine window (a boolean mask, an index array): the result's positions
                # bear no fixed relation to the original's, so it is a dimension of its own.
                survivors.append( Axis( ShapeVar() ) )
            elif window.coordinate == axis.coordinate and window.hi == axis.hi:
                survivors.append( axis )                     # the whole of it: the axis itself
            else:
                survivors.append( window )

        if len( survivors ) != len( result.shape ):
            return self._wrap( result, None )               # fancy indexing changed the rank
        return self._wrap_axes( result, survivors )

    def __repr__( self ):
        names  = self._dim_labels()
        axes   = "" if all( n is None for n in names ) else f", axes={ names }"
        kind   = ", symbolic_zero" if self.is_symbolic_zero else ""
        header = f"{ type( self ).__name__ }( shape={ self.shape }{ axes }, dtype={ self.dtype.name }, device={ self.device }{ kind } )"
        if self.raw is None:
            return header

        raw = numpy.asarray( self.raw )
        # an unrolled AxisList is always fully dense (no reservation, no padding);
        # otherwise mask out padding cell by cell, from the axes' LIVE extents.
        tree = raw.tolist() if self._has_unroll() else _display_tree( raw, self.axes )
        width = max( ( len( _fmt_scalar( v ) ) for v in _leaves( tree ) if v is not _BLANK ), default = 0 )
        return header + "\n" + _render_tree( tree, width, raw.ndim )


def _declared_dtype( cls, template_kwargs ):
    """The dtype a DECLARATION means, given the class it is written on: the explicit `dtype` kwarg
    if there is one, else the class's own default at the declared `size` (`RealTensor` -> the
    driver's ftype, `IntTensor` -> its itype). A dtype that contradicts the class is refused --
    `IntTensor[ { "dtype": float } ]` is not a narrower declaration, it is two of them."""
    declared = template_kwargs.get( "dtype", None )
    if declared is None:
        return cls.default_dtype( template_kwargs.get( "size", None ) )
    dtype = Dtype.factory( declared )
    if cls.dtype_kinds is not None and dtype.kind not in cls.dtype_kinds:
        raise TypeError( f"{ cls.__name__ } cannot be declared { dtype.cpp_name }: "
                         f"use { Tensor.class_for( dtype ).__name__ } instead" )
    return dtype


def _refuse_mismatched_window( axis, order ):
    """Refuse two DIFFERENT windows of the SAME dimension in one elementwise op.

    They are not independent axes to outer-product (they index the same thing), and they are not
    aligned either (position 0 of `num_vertex+2` is not position 0 of `num_vertex`). Broadcasting
    them would silently pair items that do not correspond, so the op stops instead. Rebuild the
    axis (or take the same window on both sides) to say what was meant."""
    for other in order:
        if other.identity is axis.identity and other.coordinate != axis.coordinate:
            raise ValueError(
                f"cannot combine '{ other.display_name }' with '{ axis.display_name }': two "
                f"different windows of the same dimension. They index the same thing, so they are "
                f"not independent axes to broadcast -- and they start at different places, so they "
                f"do not line up either. Slice both sides the same way, or give the result an axis "
                f"of its own." )


# ---- reading a dtype OFF a value, rather than believing a declaration --------------------------
def _natural_dtype( value ):
    """The dtype `value` ALREADY has, or `None` when it has none to read -- a python list of ints
    (whose type is a conversion decision, not a fact) or a ragged nesting (which is not an array at
    all). Never touches a driver buffer's data, and never forces one to the host."""
    if hasattr( value, "dtype" ):
        return Dtype.of( value )
    if isinstance( value, _containers + ( int, float, bool, numpy.generic ) ):
        try:
            return Dtype.from_numpy( numpy.asarray( value ).dtype )
        except ( ValueError, TypeError ):
            return None       # ragged nesting: numpy refuses to make one array of it
    return None


def _wrapped_dtype( raw, claimed = None ):
    """The dtype to give a tensor built AROUND an existing buffer: the buffer's own. `claimed` (a
    dtype the caller passed anyway) is CHECKED against it, not believed -- wrapping is where a
    mislabelling used to enter silently. With no buffer there is nothing to read, so `claimed` (or
    the default) stands."""
    if raw is None:
        return claimed
    have = Dtype.of( raw )
    if claimed is not None and not Dtype.factory( claimed ).same_as( have ):
        raise TypeError( f"Tensor.wrap: buffer holds { have.cpp_name }, "
                         f"but { Dtype.factory( claimed ).cpp_name } was claimed" )
    return have


def _result_dtype( raw, fallback ):
    """The dtype of an OP's result: the one its buffer came out with. An op decides its own type (a
    comparison yields booleans, an integer division reals), so nothing is inherited from the operand
    -- inheriting is precisely what used to label a bool buffer `TF`. `fallback` covers the case
    with no buffer to read (an unbound result)."""
    return fallback if raw is None else Dtype.of( raw )


# an axis is unrolled (spans several array dimensions) iff it is an `AxisList` -- the fact lives in
# the type, so a tensor stores only the axis, never a separate unroll flag (see `Tensor.__init__`).
def _is_unrolled( axis ):
    return isinstance( axis, AxisList )


def _aligned_to( arr, dims, order ):
    """`arr`, whose dimensions are the axis objects `dims`, viewed with its dimensions permuted into
    `order` and a size-1 axis inserted wherever `order` holds an axis `arr` does not have. After this
    both operands of an elementwise op share the SAME axis order, so a plain broadcast is a map by
    reference (a missing -- hence size-1 -- axis broadcasts). Matched by IDENTITY (`dims` distinct)."""
    pos = { ax.coordinate: i for i, ax in enumerate( dims ) }
    present = [ ax for ax in order if ax.coordinate in pos ]
    arr = driver.transpose( arr, [ pos[ ax.coordinate ] for ax in present ] )
    return arr[ tuple( slice( None ) if ax.coordinate in pos else None for ax in order ) ]


# containers recursed into by `_assemble` (a whitelist: anything else is a leaf)
_containers = ( list, tuple )


def _axis_rank( axis ):
    """Number of distinct axes the extent varies along (0 = dense, >0 = ragged).

    The coeffs' ShapeVars may share `dep_axes`, so this is the size of their
    union, not a sum or a max.
    """
    dep_axes = set()
    for shape_var in axis.coeffs:
        dep_axes.update( shape_var.dep_axes )
    return len( dep_axes )


def _leaves( tree ):
    if isinstance( tree, list ):
        for v in tree:
            yield from _leaves( v )
    else:
        yield tree


def _fmt_scalar( v ):
    if v is _BLANK:
        return ""
    return f"{ v:g}" if isinstance( v, float ) else str( v )


def _render_tree( tree, width, rank ):
    """Text form of a (possibly ragged) nested list of numbers: no brackets, each
    number right-justified to `width`, one row per line, a blank line between
    higher-rank blocks. A row/block that is ENTIRELY padding (e.g. the unwritten
    tail of a reservation) is dropped rather than printed as blank -- a row that
    is only partly padding (ragged in some other direction) still prints, with
    `_BLANK` cells shown empty in place, so column alignment is preserved."""
    if rank == 0:
        return _fmt_scalar( tree ).rjust( width )
    if rank == 1:
        return " ".join( _fmt_scalar( v ).rjust( width ) for v in tree )
    sep = "\n" if rank == 2 else "\n\n"
    kept = [ sub for sub in tree if not _is_blank( sub ) ]
    return sep.join( _render_tree( sub, width, rank - 1 ) for sub in kept )


# display sentinel for a raw cell that is padding, not a real value (see `_display_tree`)
_BLANK = object()


def _is_blank( tree ):
    return all( v is _BLANK for v in _leaves( tree ) )


def _shape_var_at( shape_var, axes, idx ):
    """Current value of `shape_var` (its LIVE `.value`, e.g. solved from a kernel
    write -- not a reservation) at raw position `idx`: dense (no `dep_axes`) is a
    single value (same convention as `Axis.max`); ragged is indexed by where its
    `dep_axes` sit among `axes` (a dependency this tensor does not itself carry
    as a dimension falls back to the max over it). `None` while unresolved (e.g. a
    kernel output not yet written) -- there is then nothing to tell padding from a
    real value with."""
    v = shape_var.raw
    if v is None:
        return None
    if not shape_var.dep_axes:
        return int( v.max() )
    key = tuple( idx[ axes.index( dep ) ] if dep in axes else slice( None ) for dep in shape_var.dep_axes )
    v = v[ key ]
    return int( v.max() ) if v.ndim else int( v )


def _cell_valid( axes, idx ):
    """Whether raw position `idx` holds a real value: every axis's OWN extent,
    evaluated at `idx` from its ShapeVars' current values, must cover it. Checked
    independently per axis, so ragged padding is caught in any direction, not
    only a trailing/horizontal one. An axis whose extent cannot be evaluated (an
    unresolved `ShapeVar`) is treated as fully valid -- with no known extent there
    is no padding to detect, so displaying the raw buffer as is beats crashing."""
    for d, axis in enumerate( axes ):
        extent = axis.numeric_extent( lambda sv: _shape_var_at( sv, axes, idx ) )
        if extent is None:
            continue          # no known extent -> no padding to detect: show the buffer as it is
        if idx[ d ] >= extent:
            return False
    return True


def _display_tree( raw, axes, d = 0, idx = () ):
    """Nested list over the full (padded) `raw`, `_BLANK` at every position that
    is padding rather than a real value (see `_cell_valid`)."""
    if d == raw.ndim:
        return raw[ idx ].item() if _cell_valid( axes, idx ) else _BLANK
    return [ _display_tree( raw, axes, d + 1, idx + ( i, ) ) for i in range( raw.shape[ d ] ) ]


def _assemble( value, caps, dtype, device ):
    """Build a padded rank-`len(caps)` buffer from `value`, FUNCTIONALLY: pad each
    block up to `caps` (extension), then `stack` the blocks (assembly). No in-place
    mutation, so it stays valid for Jax tracers / autodiff. Pad value is 0."""
    if not isinstance( value, _containers ):
        leaf = driver.array( value, dtype = dtype, device = device )
        pad_width = [ ( 0, caps[ i ] - leaf.shape[ i ] ) for i in range( len( caps ) ) ]
        return driver.pad( leaf, pad_width ) if any( a for _, a in pad_width ) else leaf

    children = [ _assemble( v, caps[ 1: ], dtype, device ) for v in value ]
    if len( children ) < caps[ 0 ]:
        block = driver.zeros( caps[ 1: ], dtype = dtype )
        children = children + [ block ] * ( caps[ 0 ] - len( children ) )
    return driver.stack( children, axis = 0 )
