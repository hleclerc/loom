from ..util.Aggregate import get_attribute
from ..util.annotations import annotations
from .CallArg_Aggregate import CallArg_Aggregate
from .CallArg_Attr import CallArg_Attr
from .CallArg_Errors import CallArg_Errors, ERRORS_VAR_NAME
from .IoCategory import IoCategory
from .CallArg import CallArg


# namespace of the FFI buffer parameters, kept apart from the call's argument names
_FFI_PREFIX = "ffi_"


def batch_attr_name( axis_name ):
    """The FFI attribute carrying a batch axis's item count. Shared with `JaxFfi._render_call`,
    which declares it, and `batch_dim_expr`, which reads it."""
    return f"nb_batch_{ axis_name }"


class CallArgsAnalysis:
    """Lower the kwargs of `driver.call` into a tree of `CallArg`.

    The objects are built by the CALLER (`cell = Cell( nb_dims = 2 )`); this class never
    constructs anything. It walks them and says how each attribute reaches the kernel: which
    buffers to bind and in what order, what C++ struct to emit, what to allocate, and where the
    results go back. Prescribing and sharing belong to the aggregate's constructor, not here.

    Dispatch carries no type knowledge: an attribute is asked to lower ITSELF via `make_CallArg`
    (`Tensor`, `ShapeVar` and `CtShapeVar` each answer differently; an `Axis` answers `None` --
    it is a declaration, not data). An object that does not know how (a plain `@aggregate`)
    falls back to the default: walk its fields as a `CallArg_Aggregate`.

    `path` is the dotted attribute path from the call kwarg (`"cell.vertex_positions"`): it is
    what `output_attributes`, `input_exceptions` and `capacities` name, and naming an aggregate
    covers everything below it.

    CAPACITIES live here, and nowhere else. Sizing an output is a decision about ONE allocation
    -- so it is a parameter of the call, not state on the object (which would let an object
    claim a capacity of 64 while holding a buffer of 8). `capacity_of` resolves one, in order:

    1. what this call was given (`capacities = { "cell.nb_vertices": 8 }`);
    2. else what is ALREADY ALLOCATED for that var, read off the buffers of the tensors that
       use it -- a fact, not a decision, so a chained call need not restate it;
    3. else a count Python actually holds (a `CtShapeVar`, a prescribed value);
    4. else there is nothing to allocate from, and we say which attribute is missing.

    The DEVICE is here too: it is not a runtime parameter of the kernel but part of the type of
    everything the kernel touches (a buffer's memory space is in its `TensorView` type), so every
    node needs it while it emits its C++.
    """

    tensors: list   # the buffers to bind, in FFI order
    args: dict

    def __init__( self, args : dict, device, output_attributes = (), capacities = {}, output_attribute_exceptions = (), input_exceptions = (), batch_alignment = None, scratch_attributes = () ) -> None:
        self.device = device
        # SCRATCH outputs: allocated + written like any output in the FORWARD, but the BACKWARD does
        # NOT receive them as residuals (their forward values are transient per-thread garbage). It
        # re-allocates a fresh writable buffer under the same name and re-derives what it needs (a
        # re-sort). Read by `_call_backward`; the forward treats them as ordinary outputs.
        self.scratch_paths = list( scratch_attributes )
        # BYTE alignment for the flattened batch dimension of the tensors THIS call lays out (see
        # `PhysicalLayout` / `CallArg_Tensor`): the per-call value if given, else the device default.
        self.batch_alignment_bytes = device.resolve_batch_alignment( batch_alignment )
        self.output_paths = list( output_attributes )
        self.output_exceptions = list( output_attribute_exceptions )
        self.input_exceptions = list( input_exceptions )
        self.declared_outputs = set()
        self.declared_exceptions = set()
        self.declared_input_exceptions = set()
        self.type_names = {}
        self.batch_axes = []
        self.args = {}

        # paths resolve against the objects, so a capacity keyed by path becomes one keyed by
        # the ShapeVar itself: two aggregates sharing a ShapeVar cannot disagree about it. The
        # var gets to refuse a capacity that contradicts what it is (a `CtShapeVar` does).
        self.capacities = {}
        for path, capacity in capacities.items():
            shape_var = self._attribute_at( args, path )
            shape_var.accept_capacity( capacity )
            self.capacities[ shape_var ] = int( capacity )

        # the axes this call is batched over: any aggregate argument carries them (its `batch_axes`,
        # empty for a plain one). Their NAMES drive `global_batch_indices` (see `_batch_indices_decl`)
        # and `batch_dim_expr`; the leading dimension they add to each tensor is handled like any
        # other declared axis. Collected here, deduped in first-seen order.
        for inst in args.values():
            for axis in getattr( inst, "batch_axes", () ):
                if axis.name not in self.batch_axes:
                    self.batch_axes.append( axis.name )

        for name, inst in args.items():
            # an arg may lower to NOTHING: an `Axis` is a declaration, not data, so `make_CallArg`
            # answers None. Keep it out of the tree -- exactly as `CallArg_Aggregate` does for such
            # a field -- instead of parking a None every `nodes()` walk would then trip over. The
            # axis name still reaches the C++ through any tensor of the call that references it.
            ca = self.make_CallArg( name, name, inst )
            if ca is not None:
                self.args[ name ] = ca

        # what the kernel writes when something goes wrong -- built last, so it takes the FFI slot
        # after every argument's. It is a buffer of the call, not of an argument: no object of the
        # caller's has anything to do with it.
        self.errors = CallArg_Errors( self )

        # the tree is complete: hand out the names and ids that depend on the whole of it -- a PULL
        # over the nodes, never a push as each was built (see `nodes`, `_name_buffers`).
        self._name_buffers()
        self._number_error_vars()

        # a path that reached nothing is a typo, and a silent one would turn an output into an
        # unbound attribute (the kernel would write into the void). An exception that suppressed
        # no output is just as suspect -- a typo, or a hole carved where there was nothing.
        unused = [ p for p in self.output_paths if p not in self.declared_outputs ]
        if unused:
            raise ValueError( f"output_attributes: no such attribute: { ', '.join( unused ) }" )
        unused = [ e for e in self.output_exceptions if e not in self.declared_exceptions ]
        if unused:
            raise ValueError( f"output_attribute_exceptions: excludes nothing: { ', '.join( unused ) }" )
        unused = [ e for e in self.input_exceptions if e not in self.declared_input_exceptions ]
        if unused:
            raise ValueError( f"input_exceptions: no such attribute: { ', '.join( unused ) }" )

    # -- the buffers and error slots, derived from the tree --
    @property
    def tensors( self ):
        """The FFI buffers to bind, in handler order: every node that binds one, in a stable
        pre-order (args then the error buffer last -- which is exactly the arg/result order XLA
        wants). Pulled from the tree, not a list a buffer appends itself to."""
        return [ n for n in self.nodes() if n.is_ffi_buffer() ]

    @property
    def error_vars( self ):
        """The counts that can overflow a capacity, indexed by their `error_id` -- what a record in
        the error buffer points back to (`capacity_overflows`). The id IS the position here, and
        both are the tree order, so `error_vars[ id ]` is the very node numbered `id`."""
        return [ n for n in self.nodes() if getattr( n, "error_id", -1 ) >= 0 ]

    def _name_buffers( self ):
        """Give every FFI buffer a unique name in the handler's ONE flat namespace (shared with the
        call's arguments): named after the attribute, disambiguated when two collide. Only the
        buffer is renamed -- the struct member keeps the attribute name the C++ body uses. Done in
        binding order, once the tree is built, so the names are a function of the whole call."""
        named = []
        for tensor in self.tensors:
            name = _FFI_PREFIX + tensor.name
            while name in named:
                name += "_"
            tensor.ffi_name = name
            named.append( name )

    def _number_error_vars( self ):
        """Hand each count that can overflow its slot in the error buffer, in tree order: the id
        the C++ writes is the position the host reads back here."""
        error_id = 0
        for node in self.nodes():
            if getattr( node, "wants_error_id", lambda: False )():
                node.error_id = error_id
                error_id += 1

    def capacity_overflows( self ):
        """The capacities this call turned out to be too small for: `[ ( path, wanted, capacity ) ]`,
        empty if the call went through, `None` if we cannot know here (a traced buffer -- see
        `CallArg_Errors.capacity_overflows`)."""
        return self.errors.capacity_overflows( self.error_vars )

    # -- the generic traversal every derived collection folds over --
    def nodes( self ):
        """Every `CallArg` of this call, roots and their descendants, in a stable pre-order, then
        the call's own error buffer last (its FFI slot follows every argument's).

        This is the one place the tree is walked. A collection the analysis needs -- the axes to
        `DEFINE_AXIS`, the FFI attributes, ... -- is a fold over this, each node answering for
        ITSELF (`cpp_axis_names`, `jax_attrs`). Nothing is pushed onto a typed list as the tree is
        built: the analysis PULLS what it needs, when it needs it."""
        def walk( node ):
            yield node
            for child in node.children():
                yield from walk( child )
        for arg in self.args.values():
            yield from walk( arg )
        yield self.errors

    @property
    def axis_names( self ):
        """Every axis name spelled in the generated `.cpp`, deduped in first-seen order: each node
        contributes what its TYPE references (a tensor's dimensions, a count's batch axes), so
        every one gets a `DEFINE_AXIS`. Derived by folding the tree -- not accumulated during it."""
        seen = []
        for node in self.nodes():
            for name in node.cpp_axis_names():
                if name not in seen:
                    seen.append( name )
        return seen

    # -- what a `vmap` derives --
    def batched( self, axis_name, axis_size, batched_inputs ):
        """The same lowering, of the same objects, with one more axis: what a `vmap` calls.

        A COPY, not a mutation: the caller's own lowering must keep saying what its tensors are,
        since it is the one that writes the results back onto them (and it does so with the shapes
        the outer trace sees, batch axis stripped). The Python objects are shared -- only the
        lowering nodes are cloned.

        Every OUTPUT gains the axis (the kernel writes one per item); an input gains it only if the
        mapping gave it one (`batched_inputs`, the buffers the framework says it mapped). A value
        left out is not broadcast: it is shared by every item, and the kernel simply lets the batch
        index pass through it."""
        import copy

        mapping = {}
        res = copy.copy( self )
        res.args = { n: c._clone( mapping ) for n, c in self.args.items() }
        res.errors = self.errors._clone( mapping )
        res.batch_axes = list( self.batch_axes ) + [ axis_name ]
        # `tensors`, `error_vars` and `axis_names` are not carried: all three are derived from the
        # nodes, so they follow the CLONES on their own (each keeps its `ffi_name`/`error_id`), and
        # the batch axis reaches them through the buffers we are about to give it below.

        # les clones appartiennent à `res`, pas à nous : sans ça un noeud cloné irait demander
        # l'étendue de son NOUVEL axe de batch à une analyse qui ne le connaît pas.
        for node in res.nodes():
            node._rebind_analysis( res )

        for tensor in res.tensors:
            if not tensor.takes_batch_axis():
                continue
            if tensor.io_category.is_output or tensor.ffi_name in batched_inputs:
                tensor.add_batch_axis( axis_name, axis_size )

        return res

    def batch_axis_size( self, axis_name ):
        """How many ITEMS a batch axis holds, read HOST-side off a lowering node's LOGICAL shape.

        Never off a buffer's leading dimension. The physical layout flattens that dimension and
        PADS it to the device's batch alignment, so the buffer is routinely longer than the batch
        -- 32 slots for 20 cells. A batch extent is a COUNT, and a padded capacity is precisely
        what a count is not. The logical shapes all agree on the count (they are what this call
        allocated from), so any carrier answers the same thing."""
        for node in self.tensors:
            names = getattr( node, "axis_names", None ) or []
            if axis_name in names and len( names ) == len( node.shape ):
                return int( node.shape[ names.index( axis_name ) ] )
        raise ValueError( f"no lowering node carries the batch axis '{ axis_name }'" )

    def batch_dim_expr( self, axis_name ):
        """Where the extent of `axis_name` is read in the generated C++.

        A BATCH axis is a count like any other, so it crosses like any other simple host-known
        count: BY VALUE, as an FFI attribute (see `CallArg_ShapeVar.as_scalar` for the rule, and
        `_render_call` for where the attribute is declared). Not read off a buffer -- that leading
        dimension is a padded capacity, and iterating over it walked past the end of the count
        buffer, whose garbage then bounded a loop over the geometry. Not a literal either: that
        would recompile the library for every batch size.

        Any other axis still answers with a run-time extent off the first buffer carrying it."""
        if axis_name in self.batch_axes:
            return f"SI( { batch_attr_name( axis_name ) } )"
        for node in self.tensors:
            expr = node.batch_dim_expr( axis_name )
            if expr is not None:
                return expr
        raise ValueError( f"no buffer carries the axis '{ axis_name }'" )

    @staticmethod
    def _attribute_at( args, path ):
        """The `Attribute` object a dotted path designates (`"cell.nb_vertices"`)."""
        names = path.split( "." )
        if names[ 0 ] not in args:
            raise ValueError( f"'{ path }': no argument named '{ names[ 0 ] }' in this call" )
        inst = args[ names[ 0 ] ]
        for name in names[ 1: ]:
            inst = get_attribute( name, inst )
        return inst

    @property
    def cpp_memory_space( self ):
        """Where the buffers of this call already live -- part of the type of every view we emit."""
        return self.device.cpp_memory_space

    def capacity_of( self, shape_var, path ):
        """How much to allocate for `shape_var` in this call (see the class docstring)."""
        if shape_var in self.capacities:
            return self.capacities[ shape_var ]

        allocated = shape_var.allocated_capacity()
        if allocated is not None:
            return allocated

        static = shape_var.static_count()
        if static is not None:
            return static

        raise ValueError(
            f"'{ path }' has to be allocated, but '{ shape_var.name }' has no capacity: give it "
            f"to the call ( capacities = {{ '...{ shape_var.name }': n }} )"
        )

    def output_shape( self, tensor, path ):
        """The extents to allocate `tensor` with: its axes, evaluated on the capacities."""
        res = []
        for axis in tensor.axes:
            res += axis.capacity_list( lambda sv: self.capacity_of( sv, path ) )
        return res

    def io_category( self, path, has_value ) -> IoCategory:
        """Declared as an output, or else OBSERVED: holds data -> input, empty -> unbound.

        An exception wins over the output subtree that contains it: a path carved out of the
        outputs is observed like any other, as if it had never been declared.

        `input_exceptions` is the symmetric carve-out on the OTHER default: a path forced to
        `UNBOUND` even though it holds data -- the caller asserting the kernel has no business
        touching it, so it never crosses the FFI and never becomes a differentiable primal (a
        `NoneTensor`, exactly as if it had been empty and undeclared)."""
        for e in self.input_exceptions:
            if path == e or path.startswith( e + "." ):
                self.declared_input_exceptions.add( e )
                return IoCategory.UNBOUND
        for e in self.output_exceptions:
            if path == e or path.startswith( e + "." ):
                self.declared_exceptions.add( e )
                return IoCategory.INPUT if has_value else IoCategory.UNBOUND
        for p in self.output_paths:
            if path == p or path.startswith( p + "." ):
                self.declared_outputs.add( p )
                return IoCategory.OUTPUT
        return IoCategory.INPUT if has_value else IoCategory.UNBOUND

    def is_exact_output( self, path ) -> bool:
        """Whether `path` was named ITSELF, as opposed to being covered by an ancestor.

        Naming an aggregate is a convenience -- "the outputs are in there" -- so an attribute
        that cannot be one (a compile-time `CtShapeVar`) is simply skipped. Naming an attribute
        is an assertion about that attribute, and an impossible one must be told."""
        return path in self.output_paths

    def make_CallArg( self, path, name, inst ) -> CallArg | None:
        make = getattr( type( inst ), "make_CallArg", None )
        if make is not None:
            return make( self, path, name, inst )
        # a builtin cannot carry a `make_CallArg`: a bare `int` is a runtime attribute of the call.
        if isinstance( inst, int ):
            return CallArg_Attr( self, path, name, inst )
        # default lowering: a plain aggregate, walked field by field.
        return CallArg_Aggregate( self, path, name, inst )

    def attributes_of( self, inst ):
        """The declared attributes of an aggregate instance, as `Attribute` OBJECTS: `getattr`
        would hand back the read view (a ShapeVar's count, not the ShapeVar itself)."""
        return { name: get_attribute( name, inst ) for name in annotations( type( inst ) ) }

    def cpp_type_name( self, cls ):
        """C++ name of the struct template generated for the `@aggregate` class `cls`: its
        Python name, made unique should two distinct classes share it in the same call."""
        if cls not in self.type_names:
            name = cls.__name__
            while name in self.type_names.values():
                name += "_"
            self.type_names[ cls ] = name
        return self.type_names[ cls ]
