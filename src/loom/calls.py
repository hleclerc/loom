"""THE public entry point: run a C++/CUDA kernel, and the VOCABULARY of its arguments.

`driver` is the LOW layer -- it carries the framework ( Jax/Torch ), the device, the resolved types,
and it has no business showing up in a user's code. `ffi_call` is the same call, under the name of
what it does, and with another way of saying the inputs and the outputs.

WHAT CHANGES COMPARED TO `loom.ffi_call`, and why:

  * `name` is the FIRST argument. It is mandatory -- it names the functors, prefixes the compiled
    target and groups the compilation journal -- so it has no business in the middle of the data.

  * THE ROLE IS SAID ON THE VALUE, not in a list on the side. `loom.ffi_call` carries four lists of
    paths ( `output_attributes`, the two `exceptions`, `scratch_attributes` ) that have to be kept
    in agreement BY HAND with the kwarg names -- and a typo in which only showed up at the moment
    an attribute not found was reported. Here the role is CARRIED by the argument:
    `next = loom.out( next )` can no longer designate anything but itself.

  * IN DOING SO, the kernel's namespace is given back to it -- ENTIRELY. `loom.ffi_call` reserved
    EIGHT names ( `name`, `output_attributes`, `output_exceptions`, `input_exceptions`,
    `output_capacities`, `batch_alignment`, `has_dynamic_capacity`, `scratch_attributes` ): a
    kernel that wanted an argument called `name` could not have it, and `output_attribute` in the
    singular raised nothing -- it became an argument of the kernel. The arguments now living IN A
    GROUP, the first level of `args` only contains the groups: an argument called `name`,
    `output_attributes` or even `inputs` lives at `args.inputs.<its name>` and can no longer
    collide with anything.

THE GROUPS, that is to say what C++ sees:

    args.inputs.<name>        the inputs -- including a bare scalar or integer
    args.outputs.<name>       what the kernel writes
    args.scratch.<name>       the work buffers
    args.grad_of_inputs       ( adjoint ) the cotangent to WRITE
    args.grad_of_outputs      ( adjoint ) the cotangent coming IN
    args.machine, args.errors the call, not the user

A GROUP'S NAME ALWAYS DESIGNATES THE ROLE ON THE WAY FORWARD. That is what removes the knot of
`grad_inputs` / `grad_outputs`, where "inputs" could also be read as the role of the RETURN --
the two readings then swapping the meaning. The return READS `grad_of_outputs` and WRITES
`grad_of_inputs`, because an output cotangent is what we receive and an input cotangent what we
derive. The mirror is the content; there is nothing more to remember.

THE VOCABULARY:

    loom.out( x )                             x is WRITTEN by the kernel
    loom.out( cell, writes = ( "nb_vertices", ) )   ... these, and only these
    loom.out( cell, reads  = ( "weights", ) )       ... everything but these ( the same, from the
                                                    other end -- give the shortest list )
    loom.mutable( x )                         x is READ then RE-WRITTEN ( see `mutable` )
    loom.scratch( x )                         a work buffer: allocated and written on the way
                                              forward, not handed to the adjoint as a residual
    loom.unbound( cell, "cut_offsets" )       what this kernel has no business touching, even if
                                              it is filled

An argument without a marker is an INPUT, and a raw value goes in as is: a framework array, a numpy
array, a list, a float ( see `Tensor.as_tensor` -- loom reads its element type, which is a fact,
and leaves the scalar's size to the driver, which is a policy ).

WHAT THE CALL RETURNS: everything it WROTE -- `out` as well as `mutable` -- in the order of the
kwargs ( see `returned` ). A call that writes nothing returns `None`.
"""

# THE ROLES. Strings and not an enum: these four values never leave this file.
_OUT, _MUTABLE, _SCRATCH, _UNBOUND = "out", "mutable", "scratch", "unbound"


class Arg:
    """An argument PLUS the role it holds in the call.

    Built by `loom.out` & co, never directly. `members` restricts the role to sub-paths of the
    object -- what an aggregate asks for: a `Cell` of which the kernel only writes `nb_vertices`
    and `vertex_positions`. Empty, the role holds for the whole object. `excluded` says the SAME
    thing from the other end: the whole object holds the role, except these sub-paths ( see `out` ).
    """

    __slots__ = ( "kind", "value", "members", "excluded", "capacities" )

    def __init__( self, kind, value, members = (), excluded = (), capacities = {} ) -> None:
        if members and excluded:
            raise ValueError(
                "loom.out / loom.scratch: `writes` and `reads` say the SAME thing from both "
                "ends -- what the kernel writes, or what it leaves alone. Giving both opens the "
                "door to them contradicting each other: give only one." )
        self.kind = kind
        self.value = value
        self.members = tuple( members )
        self.excluded = tuple( excluded )
        self.capacities = dict( capacities )

    def paths( self, name ):
        """The paths this role designates, seen from the call: the argument, or its members."""
        return [ f"{ name }.{ m }" for m in self.members ] if self.members else [ name ]

    def excluded_paths( self, name ):
        """The paths CARVED OUT of this role: they keep the one they would have had without the
        declaration, that is to say INPUT for whatever carries a value."""
        return [ f"{ name }.{ m }" for m in self.excluded ]

    def capacity_paths( self, name ):
        """The capacities, re-keyed on the full path ( `nb_vertices` -> `cell.nb_vertices` )."""
        return { f"{ name }.{ p }": c for p, c in self.capacities.items() }


def out( value, *, writes = (), reads = (), capacities = {} ):
    """`value` is WRITTEN by the kernel: a fresh buffer, handed back to the object once the call is done.

    AND RETURNED BY THE CALL, so `return loom.ffi_call( ... )` is enough -- no more building the
    object, calling, then returning it on a third line ( see `returned` ).

    AN AGGREGATE OF WHICH THE KERNEL ONLY WRITES A PART is said from either end, as you like, and
    the shortest wins:

        loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ) )   these, and only these
        loom.out( cell, reads  = ( "weights", ) )                          everything but these

    THE TWO LISTS ARE EXCLUSIVE, and not hints: what is not written is observed like any input. A
    typo does not get through -- a name that designates nothing, or a `reads` that excludes
    nothing, is refused.

    `reads` is generally the list that does not rot: it names what the kernel must NOT overwrite,
    so it does not move when an output is added to the aggregate.

    `capacities` says HOW MUCH to allocate, when only the caller knows ( `loom.out( cell, capacities
    = { "nb_vertices": 8 } )` ); a capacity already materialized in a buffer does not have to be
    repeated, it is read from there."""
    return Arg( _OUT, value, writes, reads, capacities )


def mutable( value, *, capacities = {} ):
    """`value` is READ then RE-WRITTEN -- and `ffi_call` returns the new value.

    THE INPUTS AND OUTPUTS OF A CALL ARE DISJOINT, as in XLA: a kernel never writes what it reads.
    An in-place update is therefore TWO buffers plus a rebinding on the python side -- which this
    marker writes in our place. C++ sees both, under derived names:

        temperature = loom.mutable( temperature )
            -> args.inputs.temperature      what goes in
            -> args.outputs.temperature     what comes out
            ( and for the adjoint: `args.grad_of_outputs.temperature` going in,
              `args.grad_of_inputs.temperature` coming out )

    The output buffer is built "like" the input ( see `_empty_like` ). A CAPACITY only ever
    describes an ALLOCATION, and only the output is allocated: `nb_vertices` therefore designates
    the output side there, with no ambiguity to lift.

    What comes back is of the same kind as what was given: a framework array if an array was
    passed, a loom tensor if a tensor was passed."""
    return Arg( _MUTABLE, value, (), capacities )


def scratch( value, *, writes = (), reads = (), capacities = {} ):
    """A WORK buffer: allocated and written on the way forward like an output, but not handed to
    the adjoint as a residual -- what it carried on the way forward is per-thread transient, which
    the return re-allocates and re-derives itself. It is not returned by the call either: a
    transient is not a result.

    `writes` / `reads` as for `out`."""
    return Arg( _SCRATCH, value, writes, reads, capacities )


def unbound( value, *members ):
    """What THIS kernel has no business touching, even if the attribute is filled: nothing crosses
    the FFI, and -- not being a buffer -- nothing becomes a differentiable primal, so the adjoint
    has no cotangent to find for it. For the members an aggregate carries and that this kernel has
    no use for.

    The list is POSITIONAL here, and that is consistent: it names what is not bound, so it says
    the role directly. `out`, on the other hand, had to say WHICH of the two roles its list
    designated -- hence `writes` / `reads`."""
    return Arg( _UNBOUND, value, members )


def lower_args( args ):
    """THE VOCABULARY, translated into what the lowering expects.

    Returns `( data, outputs, scratches, unbound, capacities, groups, returns, exceptions )`:

        data     { path: object }        what crosses
        groups     { group: { member: path } }   the first level of `args` on the C++ side
        returns      [ ( object, what we were given | None ) ]   what the call RETURNS
        exceptions  [ path ]                the sub-paths CARVED OUT of an output ( `reads` )

    The PATH and the C++ NAME are separated here, and that is what makes `mutable` possible: two
    buffers carry the SAME C++ name in two groups, under two distinct paths -- a path being what
    the capacities and the outputs designate.

    Lives here, with the markers, and not in the driver: it is the same subject.
    """
    data, outputs, scratches, unbound = {}, [], [], []
    capacities = {}
    returns, exceptions = [], []
    groups = { "inputs": {}, "outputs": {}, "scratch": {} }

    for key, arg in args.items():
        if not isinstance( arg, Arg ):
            data[ key ] = arg
            groups[ "inputs" ][ key ] = key
            continue

        if arg.kind == _MUTABLE:
            input_path, output_path = f"{ key }_input", f"{ key }_output"
            # the two buffers carry the C++ name `key`, but each its own PATH -- and a path is
            # what the capacities and the outputs designate. Two equal paths would merge them,
            # silently: we refuse.
            for derived in ( input_path, output_path ):
                if derived in args:
                    raise ValueError(
                        f"'{ key } = loom.mutable( ... )' generates the path '{ derived }', which is "
                        f"already an argument of this call. Rename one of the two." )
            obj = _empty_like( arg.value, key )
            data[ input_path ] = arg.value
            data[ output_path ] = obj
            groups[ "inputs" ][ key ] = input_path
            groups[ "outputs" ][ key ] = output_path
            outputs.append( output_path )
            capacities.update( arg.capacity_paths( output_path ) )
            returns.append( ( obj, arg.value ) )
            continue

        data[ key ] = arg.value
        capacities.update( arg.capacity_paths( key ) )
        paths = arg.paths( key )
        if arg.kind == _UNBOUND:
            # it does not cross, but it remains AN ARGUMENT: the kernel sees it ( unbound ), so
            # it lives on the side where it would be read.
            unbound += paths
            groups[ "inputs" ][ key ] = key
        elif arg.kind == _SCRATCH:
            # a scratch is allocated and written like an output; what sets it apart is what the
            # ADJOINT gets from it, that is to say nothing -- not the place where it lives.
            outputs += paths
            scratches += paths
            exceptions += arg.excluded_paths( key )
            groups[ "scratch" ][ key ] = key
        else:
            outputs += paths
            exceptions += arg.excluded_paths( key )
            groups[ "outputs" ][ key ] = key
            returns.append( ( arg.value, None ) )

    return ( data, outputs, scratches, unbound, capacities,
             { g: m for g, m in groups.items() if m }, returns, exceptions )


def returned( returns ):
    """What a call RETURNS: its WRITTEN arguments, in the order they were given ( the value alone
    if there is only one, a tuple otherwise ), and `None` if there are none.

    `out` AND `mutable`, and that is what makes it possible to write `return loom.ffi_call( ... )`
    instead of building the object, calling, then returning it on a third line.

    What comes back is not the same object in both cases, and it cannot be: an `out` returns THE ONE
    WE WERE GIVEN -- it was already ours, the result was rewritten into it -- where a `mutable`
    returns the NEW buffer ( the inputs and outputs of a call are disjoint ), of the same kind as
    what had been given ( see `_same_kind` ).

    `scratch` returns nothing, and that is its definition: a transient is not a result."""
    if not returns:
        return None
    res = [ obj if given is None else _same_kind( obj, given ) for obj, given in returns ]
    return res[ 0 ] if len( res ) == 1 else tuple( res )


def ffi_call( name, *kernels, **kwargs ):
    """Runs `kernels` on the values passed as kwargs.

    IT IS `loom.ffi_call`, under the name of what it does: there is only one form of call, and
    `driver` remains the low layer that the user has no need to name.

    WHAT IS RETURNED: the WRITTEN arguments -- `loom.out` as well as `loom.mutable` -- in the order
    they were given ( see `returned` ). Hence the short form:

        return loom.ffi_call( "my_call", kernel, input = input,
                              output = loom.out( RealTensor[ n ]() ) )
    """
    from .drivers import framework_defaults
    framework = _framework_of( kwargs.values(), framework_defaults.framework_name() )
    if framework is None:
        return framework_defaults.call( name, *kernels, **kwargs )
    # the call runs where its buffers are, whatever `loom.default_framework` was set to for the values that
    # nobody said anything about
    with framework_defaults.using( framework ):
        return framework_defaults.call( name, *kernels, **kwargs )


# ---- which framework a call runs on -------------------------------------------------------------
# The default framework only says what to BUILD when nothing was specified: a call adapts to the
# framework of its buffers (the rule is in `drivers/promotion.py`, shared with the operations).

def _framework_of( values, default ):
    """The framework name a call should run on, or `None` to keep the default one."""
    from .drivers.promotion import promote
    tracing, plain = set(), set()
    for value in values:
        _scan( value, tracing, plain, set() )
    return promote( tracing, plain, default, for_call = True )


def _scan( value, tracing, plain, seen ):
    """Collect, into `tracing` / `plain`, the framework of every buffer reachable from `value`."""
    from .tensor.Tensor import Tensor
    from .tensor.storage.buffers import buffer_class_for
    from .util.Aggregate import Aggregate

    if isinstance( value, Arg ):
        value = value.value
    if value is None or isinstance( value, ( str, int, float, bool ) ) or id( value ) in seen:
        return
    seen.add( id( value ) )

    if isinstance( value, Tensor ):
        storage = value.storage          # a zero, a fill, an unbound tensor: no buffer, so no vote
        value = storage.buffer
        if value is None:
            return
    else:
        storage = None
    if isinstance( value, ( list, tuple ) ):
        for v in value:
            _scan( v, tracing, plain, seen )
    elif isinstance( value, dict ):
        for v in value.values():
            _scan( v, tracing, plain, seen )
    elif isinstance( value, Aggregate ):
        for v in vars( value ).values():
            _scan( v, tracing, plain, seen )
    else:
        if storage is None:
            cls = buffer_class_for( value )
            storage = cls( value ) if cls is not None else None
        if storage is not None and storage.framework is not None:
            ( tracing if storage.traces else plain ).add( storage.framework )


def _empty_like( value, name ):
    """The EMPTY twin of `value` -- the output buffer of a `mutable`.

    Two protocols, in this order: `_empty_like_me()`, which a composite object defines itself
    ( a `Cell` needs its dimension, its batch and its kernel type, which loom cannot guess );
    otherwise `Tensor.like`, which covers a loom tensor as well as a raw value."""
    make = getattr( value, "_empty_like_me", None )
    if make is not None:
        return make()

    from .tensor.Tensor import Tensor
    try:
        return Tensor.like( value )
    except TypeError as e:
        raise TypeError(
            f"'{ name } = loom.mutable( ... )' : loom does not know how to build the output buffer of a "
            f"{ type( value ).__name__ }. A tensor, or a value that has a shape, goes through "
            f"`Tensor.like`; a composite object must define `_empty_like_me()`." ) from e


def _same_kind( obj, given ):
    """What a `mutable` returns: the same kind as what we were given. A raw array went in, a raw
    array comes out; a loom tensor went in, the tensor comes out.

    `.value` AND NOT `.raw`. `raw` is the BUFFER, sized to the CAPACITY -- padding included,
    because that is what a kernel writes into ( the batch alignment is 128 bytes on CUDA, so a
    batch of 3 takes up 16 slots in fp64 ). Returning it would mean returning the padding too,
    silently, and the logical value is what we want. `value` crops to the shape."""
    from .tensor.Tensor import Tensor
    if isinstance( given, Tensor ) or not isinstance( obj, Tensor ):
        return obj
    return obj.value
