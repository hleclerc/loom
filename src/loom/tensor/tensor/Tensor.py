from typing_extensions import overload
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING
import numpy

from ..util.Attribute import Attribute, resolve_attribute
from ..drivers.driver import driver
from ..devices.Device import Device

from .ReferenceShape import ReferenceShape
from .AbstractAxis import AbstractAxis
from .AxisList import AxisList
from .ShapeVar import ShapeVar
from .Dtype import Dtype
from .Axis import Axis


class Tensor( Attribute ):
    """
    Tensor declaration: a thin wrapper around the backend tensor of the chosen
    library (Jax, Torch, ...).

        t = Tensor( 17 )                        # rank 0, no declared axis
        t = Tensor[ { "dtype": int } ]( [ 1, 2 ] )
        t = Tensor[ x, y ]( [ [ 1, 2 ] ] )      # x, y being `Axis` objects

    The logical contract is the axis list; axis extents may depend on other axes
    (RAGGED axes), in which case the varying sizes live in the `ShapeVar`s of
    rank > 0. With no declared axis at all, there is nothing to solve and the
    buffer IS the contract (see `shape`). The physical contract (padding / order
    / alignment, per device) is to come, as template kwargs, kept separate from
    the axis list.

    Attributes
    * device: Device
    * dtype: Dtype
    * axes : list[ AbstractAxis ]
    * _shape: a `ReferenceShape` (or `None` if unknown), the LOGICAL sizes read from `value` ALONE,
        one `( sizes, dep_dims )` per array dimension of the value -- unpadded, and independent of the
        declared axes. `_raw` may be padded, so it only serves the CAPACITY; `_shape` serves the
        COUNT. A `ShapeVar` PULLS from it via the axis layout (see `register_in`).
    * _raw: the actual tensor (with values, trace, ...)

    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: ArrayLike | None ) -> None: ...


    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        self.device = Device.factory( template_kwargs.get( "device", None ) )
        self.dtype = Dtype.factory( template_kwargs.get( "dtype", None ) )
        self.axes = self._read_axes( template_args, scope )

        self._shape = None
        self._raw = None
        # set by `Aggregate` for a `ComputedAttribute`-declared field (see `Tensor.is_undefined`);
        # every other tensor (a plain input/output) leaves this `False`.
        self._computed_cache = False
        # the PHYSICAL arrangement of `_raw` relative to the logical axes: `None` means the plain
        # contiguous layout (buffer in logical order, no reorder, padding only from a ragged `_shape`),
        # which is every tensor today. A construction site (a kernel output, `fill`, ...) may set an
        # explicit `PhysicalLayout` (flattened/padded/reordered batch) -- ops carry it by reference.
        # It lives on the TENSOR, not the axes: the same axis sits differently in different buffers.
        self._layout = None

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
        return cls( template_args = other.axes, template_kwargs = { "dtype": other.dtype, "device": other.device } )

    @classmethod
    def full( cls, value, *, template_args = (), template_kwargs = {}, scope = None ) -> "Tensor":
        """A tensor filled with `value` everywhere, built straight from its axes
        (`Tensor[ axis ].full( v )`) -- no shape to pass: `shape` already reads it off
        the axes, the same way it would for any other tensor built on them."""
        res = cls( template_args = template_args, template_kwargs = template_kwargs, scope = scope )
        fill = value.tensor if isinstance( value, Tensor ) else value
        shape = res.shape
        # NB: a SYMBOLIC-fill path (a storageless C++ `FillTensor`, `res._fill = True` + a scalar
        # `_raw`) is prototyped (see FillTensor.h and the `is_fill` branches in Tensor / CallArg_Tensor)
        # but NOT yet wired in: it needs the fill to survive the symbolic algebra it flows through -- the
        # `target/mass * weights` of a second `normalized_version` (a `c * fill` must stay a fill), and
        # the backward RESIDUAL rank. Until then `full` MATERIALIZES: `[ n ]` weights are 80MB, already
        # small (the [nb_angles, n] blow-up is fixed by the shared `[ num_dirac ]` shape upstream).
        res._raw   = driver.full( shape, fill, dtype = res.dtype )
        res._shape = ReferenceShape.from_dense_shape( shape )
        return res

    def append_axis( self, axis ):
        """A new tensor sharing our buffer, with `axis` appended as one extra TRAILING axis --
        e.g. `SumOfDiracs1d.positions` (rank 1) reused as `SumOfDiracs`'s ( `num_dirac`, `dim` )
        positions once `dim`'s extent is known (see `SumOfDiracs1d.normalized_version`). Unset
        (`_raw is None`) stays unset: with no buffer yet, there is nothing to reshape."""
        res = type( self )( template_args = self.axes + [ axis ], template_kwargs = { "dtype": self.dtype, "device": self.device } )
        if self._raw is not None:
            res._raw = self._raw[ ..., None ]
            res._shape = self._shape.appended_dense( 1 )
        return res

    @property
    def is_symbolic_zero( self ) -> bool:
        return self._raw is not None and driver.is_symbolic_zero( self._raw )

    @property
    def is_fill( self ) -> bool:
        """A symbolic constant (`Tensor.full`): its `_raw` is a single scalar, its logical shape lives
        in the axes, and it lowers to a storageless `FillTensor`. Not a `TensorView` -- CallArg_Tensor
        binds the scalar buffer and spells the logical shape in the view."""
        return getattr( self, "_fill", False )

    @property
    def is_undefined( self ) -> bool:
        if self._raw is None:
            return True
        # a `ComputedAttribute` cache (e.g. `Image.cell_cum_mass`) is normally trusted once
        # populated -- invalidated only when a dependency is reassigned (`FieldDescriptor.__set__`).
        # That is not enough under `lax.scan`: differentiating through a scan retraces its body in
        # a SEPARATE trace to linearize/transpose it, and a cached JAX tracer from the ORIGINAL
        # trace is dead by then -- using it raises (or worse, silently corrupts) rather than just
        # being stale data. Recomputing costs nothing extra within a trace regardless (XLA's CSE
        # collapses the duplicate computation at compile time), so never trust a cached tracer.
        if self._computed_cache and driver.is_traced( self._raw ):
            return True
        return False

    @property
    def is_defined( self ) -> bool:
        return self._raw is not None

    def set( self, value ):
        if isinstance( value, Tensor ):
            self._raw = value._raw           # carries the kind along (buffer / symbolic zero / None / FILL)
            # adopt the source's reference shape: it describes the SAME buffer we just took, so it is
            # our logical shape too, whatever our own axes are (a `ShapeVar` reads it via the axis
            # layout, see `register_in`). `None` when the source has none (a kernel output whose
            # counts live in its ShapeVars) -- we then pull our capacity off the shared buffer.
            self._shape = value._shape
            # a symbolic FILL keeps its kind too: `_raw` is a scalar, `_shape` the logical extents, so
            # without this flag we would relower the scalar as a rank-0 TensorView (see `is_fill`).
            self._fill = getattr( value, "_fill", False )
            return

        # The LOGICAL shape is a pure fact about `value` (see `ReferenceShape`), read WITHOUT touching
        # its data. From it every ShapeVar PULLS its count (`register_in`); nothing is pushed. The
        # buffer is then a plain dense array, or -- when the value is jagged -- an ASSEMBLED padded
        # one whose per-dim capacity is (for now) the max size.
        self._shape = ReferenceShape.from_value( value )
        if self._shape.is_ragged():
            self._raw = _assemble( value, self._shape.capacities(), self.dtype, self.device )
        else:
            self._raw = driver.array( value, dtype = self.dtype, device = self.device )

    def set_raw( self, raw ):
        """Bind the buffer a kernel produced (a driver tensor). Sizes stay unobserved: an
        output's extents are the ones we ASKED for (`shape`), and its counts live in the
        ShapeVars the kernel wrote -- there is nothing to solve from the data.

        `raw` may also be a symbolic-zero cotangent handed back by the framework: it lands here the
        same way, and `is_symbolic_zero` recognizes it -- no special case."""
        self._raw = raw

    @property
    def buffer_layout( self ):
        """The single per-Tensor description of how `_raw` is laid out physically vs the logical axes
        (see `_layout`): the explicit one if set, else the plain CONTIGUOUS layout derived from the
        allocated extents -- so today's tensors answer exactly as before. This is where `capacity`,
        the ShapeVar capacity inversion and the FFI lowering will read the buffer's shape/strides
        from, in ONE place, unifying padding (ragged / batch) and physical order."""
        from .PhysicalLayout import PhysicalLayout
        if self._layout is not None:
            return self._layout
        return PhysicalLayout.contiguous( [ 0 ] * self.rank if self.raw is None else list( self.raw.shape ) )

    @property
    def capacity( self ):
        """What our buffer IS: the allocated extents per LOGICAL dimension. An input is bound at THIS
        size -- an output that wants to grow must not force us to inflate the input."""
        if self.is_fill:
            # a fill has no [shape] buffer; its logical extents ARE its capacity, and they are read
            # from its STORED reference shape -- NOT recomputed from the axes, which a backward residual
            # may not have resolved yet (the C++ view reads the true extent from a sibling at run time).
            return tuple( self._shape.capacities() )
        return tuple( self.buffer_layout.caps )

    @property
    def allocated_sizes( self ):
        """The allocated capacity per LOGICAL dimension -- what a `ShapeVar` inverts to learn the
        capacity it was allocated with. Read from `buffer_layout` (NOT `_raw.shape` directly), so a
        non-contiguous / flattened / padded buffer answers with the per-axis capacity rather than the
        raw physical extents. `None` while unbound. For the plain layout `caps == _raw.shape`."""
        if self.raw is None or self.is_fill:
            return None   # a fill backs no per-axis capacity a ShapeVar could invert (its count comes
                          # from a real sibling buffer); skip it, do not read the scalar's [] shape
        return [ numpy.array( c, dtype = int ) for c in self.buffer_layout.caps ]


    def _read_axes( self, axes, scope ):
        res = []
        for entry in axes:
            if isinstance( entry, str ):
                if entry.endswith( "..." ):
                    entry = entry[ :-3 ]
                entry = resolve_attribute( entry, scope )
            elif isinstance( entry, ShapeVar ):
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
            return list( self._raw.shape ) if self._raw is not None else []

        # each member contributes a LIST of extents (one for an `Axis`, `nb_dims`
        # for an unrolled `AxisList`); concatenation gives the tensor's extents.
        res = []
        for axis in self.axes:
            res += axis.max_list()
        return res

    @property
    def rank( self ):
        if not self.axes:
            return self._raw.ndim if self._raw is not None else 0
        return len( self.axes )

    @property
    def raw( self ):
        """The MATERIALIZED buffer, or `None` when there is nothing to bind -- a symbolic zero has
        no storage, so it reads as `None` here (that is how `is_bound` stays false for it), while
        `_raw` still holds the framework's zero object for `is_symbolic_zero` to recognize."""
        return None if self.is_symbolic_zero else self._raw

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

        With an explicit non-contiguous `_layout` (flattened / padded / reordered buffer) the plain
        slice does not apply: the LOGICAL view is GATHERED from the physical buffer via the layout's
        strides (a differentiable gather). This is the one physical->logical boundary; everything else
        reads `tensor`, so ops and results stay logical whatever the storage."""
        if self.raw is None:
            return None
        if self._layout is not None and not self._layout.is_identity:
            return self._gather_logical()
        return self.raw[ tuple( slice( 0, s ) for s in self.shape ) ]

    def _gather_logical( self ):
        """The logical dense view of a non-contiguously laid-out buffer: `flat[ offsets ]` where
        `offsets[ i0, ..., ik ] = sum_d i_d * stride_d` (element strides from `_layout`), so a
        flattened/padded/reordered buffer is read back in logical order. `offsets` is a static index
        grid; the gather rides the backend (differentiable)."""
        extents = self.shape
        strides = self._layout.strides
        offsets = numpy.zeros( tuple( extents ), dtype = int )
        for i, ( e, s ) in enumerate( zip( extents, strides ) ):
            shape = [ 1 ] * len( extents )
            shape[ i ] = e
            offsets = offsets + numpy.arange( e, dtype = int ).reshape( shape ) * int( s )
        return self._raw.reshape( -1 )[ offsets ]

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
        return type( self ).wrap( raw, names, dtype = self.dtype, device = self.device )

    @classmethod
    def wrap( cls, raw, names = None, dtype = None, device = None ):
        """A DETACHED tensor around an existing backend buffer: no aggregate, fresh DEFAULT axes
        (one per array dimension, named from `names`, observed straight from the dense buffer). This
        is how a `ShapeVar` hands its count back as a `Tensor`, and the FRESH-axis path for an op
        result that keeps no axis identity (`matmul`) -- the buffer is the whole contract, the axes
        just carry the names. An op that DOES preserve identity uses `_wrap_axes` instead."""
        res = cls( template_kwargs = { "dtype": dtype, "device": device } )
        res._raw = raw
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
        # still has a `_shape` to reshape from (`append_axis`). A symbolic zero carries no readable one.
        if raw is not None and not driver.is_symbolic_zero( raw ):
            res._shape = ReferenceShape.from_dense_shape( raw.shape )
        return res

    def _dim_names( self ):
        """One axis name (or `None`) per ARRAY dimension, read uniformly off the axes (an unrolled
        AxisList spreads its name over its spanned dimensions)."""
        return self._dim_attr( lambda axis: axis.name )

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
        """A dimension index, from an axis OBJECT (matched by identity -- the robust key), an int
        (returned as is), or an axis NAME (looked up in `_dim_names`)."""
        if isinstance( key, AbstractAxis ):
            for i, a in enumerate( self._dim_axes() ):
                if a is key:
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
        # the result always stays in `self.dtype`, like `set()` already guarantees for a fresh value.
        if not isinstance( other, Tensor ) and not driver.is_symbolic_zero( other ):
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
        """One axis OBJECT per array dimension when each dim maps to a DISTINCT axis (so a map by
        reference is unambiguous), else `None`. A rank-0 tensor qualifies with an empty list."""
        axes = self._dim_axes()
        if len( axes ) != self.rank:
            return None
        for i, a in enumerate( axes ):
            if any( a is b for b in axes[ :i ] ):
                return None
        return axes

    def _ref_binary( self, other, la, lb, op ):
        # canonical order (by IDENTITY): batch axes first, then first-seen (self, then other).
        order = []
        for want_batch in ( True, False ):
            for dims in ( la, lb ):
                for ax in dims:
                    if bool( getattr( ax, "is_batch", False ) ) == want_batch and not any( ax is o for o in order ):
                        order.append( ax )
        raw = op( _aligned_to( self.tensor, la, order ), _aligned_to( other.tensor, lb, order ) )
        return self._wrap_axes( raw, order )

    def _wrap_axes( self, raw, dim_axes ):
        """A detached tensor around `raw` whose dimensions ARE `dim_axes` -- the very axis OBJECTS, so
        identity (hence names and `is_batch`) rides along into the next op. Consecutive repeats (an
        `AxisList` over several dims) collapse to one entry. The axes are NOT re-registered: their
        ShapeVars are already sized, and `shape` reads the sizes straight off them."""
        axes = []
        for ax in dim_axes:
            if not axes or axes[ -1 ] is not ax:
                axes.append( ax )
        res = type( self )( template_kwargs = { "dtype": self.dtype, "device": self.device } )
        res.axes = axes
        res._raw = raw
        if raw is not None and not driver.is_symbolic_zero( raw ):
            res._shape = ReferenceShape.from_dense_shape( raw.shape )
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
        survivors = [ dims[ d ] for d, k in enumerate( key ) if not isinstance( k, int ) and d < len( dims ) ]
        return self._wrap_axes( result, survivors )

    def __repr__( self ):
        names  = self._dim_names()
        axes   = "" if all( n is None for n in names ) else f", axes={ names }"
        kind   = ", symbolic_zero" if self.is_symbolic_zero else ""
        header = f"Tensor( shape={ self.shape }{ axes }, dtype={ self.dtype.name }, device={ self.device }{ kind } )"
        if self.raw is None:
            return header

        raw = numpy.asarray( self.raw )
        # an unrolled AxisList is always fully dense (no reservation, no padding);
        # otherwise mask out padding cell by cell, from the axes' LIVE extents.
        tree = raw.tolist() if self._has_unroll() else _display_tree( raw, self.axes )
        width = max( ( len( _fmt_scalar( v ) ) for v in _leaves( tree ) if v is not _BLANK ), default = 0 )
        return header + "\n" + _render_tree( tree, width, raw.ndim )


# an axis is unrolled (spans several array dimensions) iff it is an `AxisList` -- the fact lives in
# the type, so a tensor stores only the axis, never a separate unroll flag (see `Tensor.__init__`).
def _is_unrolled( axis ):
    return isinstance( axis, AxisList )


def _aligned_to( arr, dims, order ):
    """`arr`, whose dimensions are the axis objects `dims`, viewed with its dimensions permuted into
    `order` and a size-1 axis inserted wherever `order` holds an axis `arr` does not have. After this
    both operands of an elementwise op share the SAME axis order, so a plain broadcast is a map by
    reference (a missing -- hence size-1 -- axis broadcasts). Matched by IDENTITY (`dims` distinct)."""
    pos = { id( ax ): i for i, ax in enumerate( dims ) }
    present = [ ax for ax in order if id( ax ) in pos ]
    arr = driver.transpose( arr, [ pos[ id( ax ) ] for ax in present ] )
    return arr[ tuple( slice( None ) if id( ax ) in pos else None for ax in order ) ]


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
        values = [ _shape_var_at( shape_var, axes, idx ) for shape_var in axis.coeffs ]
        if any( v is None for v in values ):
            continue
        extent = axis.offset + sum( coeff * v for coeff, v in zip( axis.coeffs.values(), values ) )
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
