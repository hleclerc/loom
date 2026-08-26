from .Parametrized import Parametrized
from .annotations import annotations
from .Attribute import Attribute
from .ComputedAttribute import ComputedAttribute
import inspect


def _unwrap_computed_attribute( type_attr ):
    """If type_attr is ComputedAttribute[Type, deps...], return (Type, deps).

    Otherwise return (type_attr, ()).
    """
    if isinstance( type_attr, Parametrized ) and type_attr.cls is ComputedAttribute:
        # args = (RealType, "dep1", "dep2", ...)
        real_type = type_attr.args[0] if type_attr.args else None
        dependencies = type_attr.args[1:] if len( type_attr.args ) > 1 else ()
        return real_type, dependencies
    return type_attr, ()


class Aggregate:
    """
    Base class for aggregates: classes whose fields are `Attribute` declarations.

    `Aggregate` works one level of abstraction above any concrete field type: it
    knows nothing about `ShapeVar` / `Axis` / `Tensor` (mere examples), only the
    `Attribute` protocol. Each annotation is a `Parametrized` schema; a fresh
    per-instance `Attribute` is built lazily by `get_attribute` and kept, under
    its field name, in the instance `__dict__`.

    Subclass it -- `class Cell( Aggregate ): ...` -- rather than decorating: the
    machinery is REAL methods (`__base_init__`, `apply_batch_axes`, `get_attribute`)
    and a real `__init__`, so a type checker / IDE sees them (no `TYPE_CHECKING`
    stubs to hand-maintain), and a subclass with its own construction just OVERRIDES
    `__init__` and calls `super().__base_init__( ... )` where it wants the fields set
    up (see `tests/python/test_Cell.py`) -- plain MRO, no bytecode sniffing. Per-field
    SET-only descriptors are installed by `__init_subclass__` at class-creation time.

    Each field can be *injected* at construction by passing its name as a kwarg;
    an `Attribute` value is shared (several instances then agree on the same
    value, e.g. a `ShapeVar` solved from the union of their tensors), any other
    value prescribes it:

        n = ShapeVar()
        a = Cell( nb_dims = n )   # a and b share nb_dims
        b = Cell( nb_dims = n )

    A field whose type is itself an `Aggregate` NESTS: it is built with the same
    kwargs, so what is written at the outer level reaches every field below it. A
    plain mapping under a field's name opens a scope for that field alone, which
    is how two nested aggregates get different values:

        Pair( nb_dims = 2, left = { "dtype": ... }, right = { "nb_dims": 3 } )

    A kwarg either SHARES an `Attribute` (the same object lands in both aggregates)
    or PRESCRIBES a value. Nothing else: notably, a capacity is NOT set here -- it
    is a decision about one allocation, so it is given to the call that allocates
    (`driver.call( ..., capacities = { ... } )`). A key matching no field here is
    not an error as long as some nested aggregate could consume it.

    Reads: `c.field` is a plain read of the instance `__dict__` -- so it hands back
    the `Attribute` OBJECT itself (a `ShapeVar`, an `Axis`, ...), value-on-read
    being the concrete type's business (`c.nb_dims.value`), not the aggregate's;
    `c.field = value` routes to `set` via the per-field descriptor.
    """

    # convention: every aggregate carries `batch_axes`, the (possibly empty) list of axes it is
    # batched over -- read uniformly by the methods (an auxiliary output is `RealTensor[ *batch_axes ]`)
    # and by `CallArgsAnalysis` (which folds them into `global_batch_indices`). A plain aggregate has
    # none; `apply_batch_axes` sets them. Left UNANNOTATED so it is not a field (no descriptor, and
    # `annotations()` -- hence the C++ lowering -- never sees it). The class-level `[]` is a mere
    # default: every instance reassigns `self.batch_axes` in `__base_init__`, so the shared list is
    # never mutated in place.
    batch_axes = []

    def __init_subclass__( cls, **kwargs ):
        # Track which computed attributes depend on each field, for invalidation.
        # Built at class time: scan annotations for ComputedAttribute instances.
        cls._computed_attr_deps = {}  # field_name -> list of ComputedAttribute names that depend on it
        for name, type_attr in annotations( cls ).items():
            sc = _field_cls( type_attr )
            if inspect.isclass( sc ) and issubclass( sc, ComputedAttribute ):
                # This field is a ComputedAttribute; record its dependencies.
                # We'll revisit when instantiating to link them properly.
                pass
        super().__init_subclass__( **kwargs )

        # one SET-only data descriptor per leaf `Attribute` field, to route its writes to `.set()`.
        # A nested aggregate has no `set` (you reach into it: `p.left.nb_dims = ...`), so it needs no
        # descriptor at all -- `p.left` is just the instance our constructor stored in `__dict__`.
        for name, type_attr in annotations( cls ).items():
            if _is_attribute_field( type_attr ):
                setattr( cls, name, FieldDescriptor( name ) )

    # ------------------ construction ------------------
    def __base_init__( self, **kwargs ):
        cls = type( self )
        anns = annotations( cls )

        # Track computed attributes and what they depend on, for cache invalidation.
        self._computed_attrs = {}  # field_name -> ComputedAttribute
        self._dependent_computed = {}  # dependency_name -> list of computed field names

        # batching is a reserved construction option, not a field: `batch_axes = [ ax, ... ]` adds
        # those axes on the LEFT of every tensor we declare. Popped here so it never reaches the
        # field logic; the tensors are (re)built with the axes prepended at the end, once the plain
        # fields exist -- the very same path a method (`init_as_hypercube( batch_axes = ... )`) takes
        # to batch a cell after construction (see `apply_batch_axes`).
        batch_axes = kwargs.pop( "batch_axes", None )
        self.batch_axes = []

        # a mapping under a FIELD's name scopes that field; everything else is visible to this
        # class AND to every aggregate nested below it.
        scoped = { n: v for n, v in kwargs.items() if n in anns and isinstance( v, dict ) }
        shared = { n: v for n, v in kwargs.items() if n not in scoped }

        # injections: share the passed `Attribute` (same object) rather than assign it. Stored
        # straight into `__dict__` (a raw write, past the set-only descriptor which would else
        # `.set()` it) -- the same slot every read then hands back. A `Tensor` is excluded: its
        # axes are OUR schema's (`RealTensor[ "num_vertex", "dim" ]`, resolved in OUR scope), never
        # the ones the caller's tensor happened to carry (e.g. from another aggregate's fields, or
        # an anonymous axis from `append_axis`) -- so it always goes through the schema-built path
        # below and adopts the value via `.set()` (buffer + `_shape` only) instead of by identity.
        for name, type_attr in anns.items():
            sc = _field_cls( type_attr )
            value = shared.get( name )
            if value is not None and inspect.isclass( sc ) and isinstance( value, sc ) and not _is_tensor_field( type_attr ):
                self.__dict__[ name ] = value

        # instantiate the fields; a nested aggregate inherits our scope, refined by its own
        for name, type_attr in anns.items():
            if name in self.__dict__:
                continue

            # Check if this is a ComputedAttribute declaration
            real_type, dependencies = _unwrap_computed_attribute( type_attr )

            if _is_aggregate( type_attr ):
                nested = type_attr( **{ **shared, **scoped.get( name, {} ) } )
                nested.name = name
                self.__dict__[ name ] = nested
            else:
                # Use real_type (unwrapped) if it's a ComputedAttribute
                if real_type is not None:
                    # It's a ComputedAttribute[RealType, deps]
                    # Create the real attribute (Tensor, ShapeVar, etc.)
                    attr = get_attribute( name, self, real_type )
                    if hasattr( attr, "_computed_cache" ):
                        attr._computed_cache = True   # see `Tensor.is_undefined`
                    # Also create a ComputedAttribute tracker for invalidation
                    computed_tracker = ComputedAttribute()
                    computed_tracker.name = name
                    self._computed_attrs[ name ] = computed_tracker
                    # Register dependencies
                    for dep_name in dependencies:
                        if dep_name not in self._dependent_computed:
                            self._dependent_computed[ dep_name ] = []
                        self._dependent_computed[ dep_name ].append( name )
                else:
                    # Regular attribute, no computed tracking
                    attr = get_attribute( name, self, type_attr )

        # batching happens BEFORE the prescriptions so a prescribed INPUT tensor is set onto the
        # already-batched schema: its leading dims then line up, axis for axis, with the batch axes
        # we prepended (the ShapeVars pull their counts positionally, see `Tensor.set`). Doing it
        # after would rebuild the tensor fields and DROP the values we just set.
        if batch_axes:
            self.apply_batch_axes( batch_axes )

        # prescriptions
        for key, value in shared.items():
            if key in anns:
                if self.__dict__[ key ] is not value:   # an injection is already in place
                    # a prescribed INPUT tensor whose leading dims do NOT match the batch extents is
                    # SHARED across the batch (e.g. `target_mass`, or a detector `origin`/`frame` that
                    # is the same at every angle): rebuild its field UNBATCHED so its C++ type omits
                    # the batch axis. The batch `AxisIndex` is optional, so the kernel's
                    # `x( batch_index )` simply lets the index fall through for such a member.
                    real_type, _ = _unwrap_computed_attribute( anns[ key ] )
                    is_tensor = _is_tensor_field( real_type if real_type is not None else anns[ key ] )
                    if self.batch_axes and is_tensor and not _carries_batch_dims( value, self.batch_axes, self.__dict__[ key ] ):
                        self._rebuild_field_unbatched( key )
                    get_attribute( key, self ).set( value )
            elif not any( _is_aggregate( t ) for t in anns.values() ):
                raise TypeError( f"'{ cls.__name__ }' has no field '{ key }' to initialize" )

    # By default `Aggregate()` builds every field. A subclass with its own construction just defines
    # `__init__` (plain override) and calls `self.__base_init__( ... )` where it wants the fields set
    # up (see `tests/python/test_Cell.py`).
    __init__ = __base_init__

    # ------------------ batching ------------------
    def apply_batch_axes( self, batch_axes ):
        """Batch this aggregate over `batch_axes`: (re)build every tensor field with those axes
        PREPENDED. Reachable at construction (`batch_axes = ...`, via `__base_init__`) or later,
        before the tensors are written (`c.init_as_hypercube( batch_axes = ... )`).

        The annotations -- the shared "scalar" schema -- are untouched; only the per-instance tensors
        gain the leading axes. Rebuilding (rather than mutating `specs` in place) is what keeps the
        dimension indices right: a fresh tensor registers each axis at its true position, and the
        unbatched one it replaces dies, taking its now-stale registrations with it. A nested
        aggregate is batched over the SAME axis objects -- joined iteration, no name to collide."""
        self.batch_axes = list( batch_axes )
        for name, type_attr in annotations( type( self ) ).items():
            # Unwrap ComputedAttribute to get the real type
            real_type, _ = _unwrap_computed_attribute( type_attr )
            type_to_check = real_type if real_type is not None else type_attr

            if _is_aggregate( type_to_check ):
                # a nested aggregate is co-iterated over the SAME axis objects -- but only if it does
                # not ALREADY carry them: an injected, already-batched input (e.g. `OtPlan1d`'s
                # `src_dist`) must not be re-batched, which would double the axes and drop its values.
                nested = get_attribute( name, self )
                if list( nested.batch_axes ) != list( self.batch_axes ):
                    nested.apply_batch_axes( self.batch_axes )
            elif _is_tensor_field( type_to_check ):
                attr = _batched_schema( type_to_check, self.batch_axes )( scope = self )
                attr.name = name
                if hasattr( attr, "_computed_cache" ):
                    attr._computed_cache = ( real_type is not None )   # see `Tensor.is_undefined`
                self.__dict__[ name ] = attr

    def _rebuild_field_unbatched( self, name ):
        """Replace the (batched) tensor field `name` with a fresh one built straight from its
        annotation -- i.e. WITHOUT the batch axes. Used when a prescribed value turns out to be
        shared across the batch (see `__base_init__`)."""
        type_attr = annotations( type( self ) )[ name ]
        real_type, _ = _unwrap_computed_attribute( type_attr )
        schema = real_type if real_type is not None else type_attr
        attr = schema( scope = self )
        attr.name = name
        if hasattr( attr, "_computed_cache" ):
            attr._computed_cache = ( real_type is not None )   # see `Tensor.is_undefined`
        self.__dict__[ name ] = attr
        return attr

    # the scope protocol (see `resolve_attribute`): what turns the NAME an attribute reads in a
    # declaration (`RealTensor[ "num_vertex" ]`) into the very object this instance holds.
    def get_attribute( self, name ):
        return get_attribute( name, self )


class FieldDescriptor:
    """SET-only data descriptor installed per `Aggregate` field.

    It intercepts WRITES only (`c.field = value` -> `attr.set(value)`); it has no `__get__`, so a
    read falls through to the instance `__dict__`, where `get_attribute` keeps the per-instance
    `Attribute`. `c.field` therefore returns that OBJECT (a `ShapeVar`, an `Axis`, ...): reading a
    value off it (`c.nb_dims.value`) is the concrete type's affair, not something the aggregate
    imposes. Class access (`Cls.field`) returns the descriptor itself, a handle on the schema.

    Having `__set__` makes it a DATA descriptor -- which would normally shadow the instance dict,
    but only through its `__get__`; with none, CPython's lookup skips straight to `__dict__`.
    """
    def __init__( self, name ):
        self.name = name

    def __set__( self, obj, value ):
        get_attribute( self.name, obj ).set( value )
        # Invalidate any computed attributes that depend on this field. `invalidate()` itself only
        # flips a flag on the SEPARATE tracker (`_computed_attrs`, kept apart for bookkeeping) --
        # the REAL attribute (`obj.__dict__[ computed_name ]`, what `obj.computed_name.is_undefined`
        # actually reads, e.g. `Image.mass`'s `current_mass.is_undefined` check) is reset here too,
        # so a lazy re-check after invalidation sees it as unresolved again instead of stale.
        if hasattr( obj, '_dependent_computed' ):
            for computed_name in obj._dependent_computed.get( self.name, [] ):
                obj._computed_attrs[ computed_name ].invalidate()
                real_attr = obj.__dict__.get( computed_name )
                if real_attr is not None and hasattr( real_attr, "set_raw" ):
                    real_attr.set_raw( None )


def _field_cls( type_attr ):
    return type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr


def _is_attribute_field( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Attribute )


def _is_aggregate( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Aggregate )


def _is_tensor_field( type_attr ):
    from ..tensor.Tensor import Tensor
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Tensor )


def _carries_batch_dims( value, batch_axes, batched_attr ):
    """Does `value` carry the batch axes as genuine leading dims (a per-batch-element input), or is
    it SHARED across the batch? Read from the shape ALONE -- never touches the data.

    Matching leading dims is NECESSARY but not SUFFICIENT: when a batch extent is 1 (or coincides
    with a shared member's own leading extent), a shared member's shape passes that test too -- e.g.
    a detector `origin` of shape [ 1 ] against a single-angle batch [ 1 ]. So we also require the
    FULL rank of a batched value: `batched_attr` already carries the prepended batch axes, so a
    genuine per-element input has ALL of its array dims (batch + own), whereas a shared value is
    short by exactly `len( batch_axes )`. When the batched rank is not yet knowable, fall back to the
    leading-dims verdict alone (the pre-existing behavior)."""
    from ..tensor.Tensor import Tensor
    import numpy
    sizes = [ int( ax.max ) for ax in batch_axes ]
    shape = list( value.shape ) if isinstance( value, Tensor ) else list( numpy.shape( value ) )
    if shape[ :len( sizes ) ] != sizes:
        return False
    dims = [ batched_attr._axis_array_dims( a ) for a in batched_attr.axes ]
    if any( d is None for d in dims ):
        return True
    return len( shape ) == sum( dims )


def _batched_schema( type_attr, axes ):
    """`type_attr` (a Tensor field schema) with `axes` prepended to its declared axes. A bare
    `RealTensor` becomes `RealTensor[ *axes ]`; a `RealTensor[ "num_vertex", "dim" ]` becomes
    `RealTensor[ *axes, "num_vertex", "dim" ]`, its dtype/device kwargs preserved.

    The declared CLASS rides along: batching adds leading axes, it does not change what the
    tensor is made of -- prepending an axis to an `IntTensor` field must not hand back a real one."""
    if isinstance( type_attr, Parametrized ):
        return Parametrized( type_attr.cls, *axes, *type_attr.args, dict( type_attr.kwargs ) )
    return Parametrized( type_attr, *axes )


def get_attribute( name, parent_inst, type_attr=None ):
    """Get or create an Attribute for a field.

    Args:
        name: field name
        parent_inst: the Aggregate instance
        type_attr: optional override of the annotation type (used for ComputedAttribute unwrapping)
    """
    # already instantiated ? (the per-instance `Attribute` lives under its field name in `__dict__`)
    attrs = parent_inst.__dict__
    if name in attrs:
        return attrs[ name ]

    # Resolve type_attr from annotation if not provided
    if type_attr is None:
        dct = annotations( parent_inst.__class__ )
        if name not in dct:
            raise ValueError( f"There's no attribute '{ name }' in '{ type( parent_inst ) }'" )
        type_attr = dct[ name ]

    if _is_attribute_field( type_attr ):
        # the parent is not a ctor argument, it is the SCOPE the declaration's names are
        # resolved in -- an `Attribute` is built the same way with or without one.
        res = type_attr( scope = parent_inst )
    else:
        res = type_attr()

    # the field name is the aggregate's to give: an `Attribute` does not know it (and may
    # outlive its parent -- a borrowed `Axis` still has to be nameable in the C++ code).
    res.name = name

    # a raw write, past the set-only `FieldDescriptor` (which would `.set()` the value instead).
    attrs[ name ] = res
    return res
