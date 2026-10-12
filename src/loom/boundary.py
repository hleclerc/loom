"""THE BOUNDARY between loom and the user: what a public function returns leaves the loom universe for
the caller's.

Inside, every value is a loom `Tensor` (a `RealTensor` around a jax / torch / numpy buffer): that is
what the internals compute with, and what they hand each other. The user, for their part, works in
ONE framework and expects its arrays back -- `sdot.w2( f, g )` on jax inputs is a jax array, so that
`jax.grad` of it works without knowing that loom exists.

    @loom.public
    def w2( f, g ):
        return plan( f, g ).cost          # a `RealTensor`, inside

    sdot.w2( f, g )                       # a jax array, when `f` and `g` were made of jax arrays
    sdot.w2.internal( f, g )              # the `RealTensor`: what another internal function calls

THE UNIVERSE of a call is read off its arguments, with the rule of `ffi_call` (`drivers/promotion.py`):
the framework of the buffers they hold -- an `Aggregate` (a distribution) is looked into --, a tracing
buffer winning over plain memory. A loom `Tensor` given AS AN ARGUMENT says that the caller lives in
loom: the result is then handed back as it is. Arguments that hold no buffer (lists, numbers) leave
it to the default framework.

THERE IS NO MODE. The conversion is done by `public.__call__`, always, and only there: an internal
function that calls a public one goes through `.internal`, and if it forgets, it pays one conversion
-- the result is wrapped again by the next call -- but gets a value that is right. The other direction
(internal code calling a USER callback) is not handled here: the call site exports what it passes,
with `export`."""
import functools


class public:
    """A public entry point: what it returns is `export`ed to the caller's universe. `.internal` is the
    same function without the conversion (and without reading the arguments), for loom-side callers.

    A method works too: `__get__` binds the instance, which then takes part in the universe as any
    argument does (`plan.barycenters()`, `plan.barycenters.internal()`)."""

    def __init__( self, fn, instance = () ):
        self.internal = fn
        self._instance = instance        # `( obj, )` once bound: the instance, which `args` no longer shows
        functools.update_wrapper( self, fn )

    def __call__( self, *args, **kwargs ):
        # read BEFORE the call: the call may well cache values of its own framework in its arguments
        universe = universe_of( self._instance + args, kwargs )
        return export( self.internal( *args, **kwargs ), universe )

    def __get__( self, obj, cls = None ):
        if obj is None:
            return self
        # at each access to the method: our `__dict__` (name, doc, `__wrapped__`) is copied as it is,
        # rather than made again by `update_wrapper`
        res = object.__new__( public )
        res.__dict__.update( self.__dict__ )
        res.internal = self.internal.__get__( obj, cls )
        res._instance = ( obj, )
        return res

    def __repr__( self ):
        return f"<public { getattr( self, '__qualname__', self.internal ) }>"


def export( value, universe ):
    """`value` as the user of `universe` expects it: a loom `Tensor` becomes an array of that framework
    (its LOGICAL value, padding cropped), a `ShapeVar` its count; tuples, lists and dicts are looked
    into. Anything else -- a plan, a distribution -- is handed back as it is: an object of the API
    stays in loom, its attributes are read with `.value`.

    `universe`: a framework name, or `"loom"` (nothing is converted) -- what `universe_of` says."""
    from .tensor.Tensor import Tensor
    from .tensor.ShapeVar import ShapeVar
    from .tensor.storage.buffers import buffer_class_for

    if isinstance( value, Tensor ):
        return value if universe == "loom" else _to_framework( value.value, universe )
    if isinstance( value, ShapeVar ):
        return value if universe == "loom" else value.value
    if isinstance( value, tuple ):
        return type( value )( *( export( v, universe ) for v in value ) ) if hasattr( value, "_fields" ) else \
               tuple( export( v, universe ) for v in value )
    if isinstance( value, list ):
        return [ export( v, universe ) for v in value ]
    if isinstance( value, dict ):
        return { k: export( v, universe ) for k, v in value.items() }
    if value is not None and not isinstance( value, ( str, int, float, bool ) ) and buffer_class_for( value ) is not None:
        # a bare array of a framework, computed inside: it crosses too
        return value if universe == "loom" else _to_framework( value, universe )
    return value


def universe_of( args = (), kwargs = None ):
    """The universe of a call made with `args` / `kwargs`: `"loom"` when one of them IS a loom
    `Tensor`, else the framework their buffers promote to (see `drivers/promotion.py`), else the
    default framework.

    An object can DECLARE its universe in a `_universe` attribute, which is then taken instead of
    looking into it -- what an object of the API does at construction (a distribution: see sdot's
    `Distribution.__base_init__`), because what it holds later also contains what the internals
    cached in it, of whatever framework they computed in."""
    from .tensor.Tensor import Tensor
    from .calls import _scan
    from .drivers.promotion import promote
    from .drivers import framework_defaults

    tracing, plain, seen = set(), set(), set()
    for value in ( *args, *( kwargs or {} ).values() ):
        if isinstance( value, Tensor ):
            return "loom"
        declared = getattr( value, "_universe", None ) if not isinstance( value, ( str, int, float, bool ) ) else None
        if declared == "loom":
            return "loom"
        if isinstance( declared, str ):
            plain.add( declared )
        else:
            _scan( value, tracing, plain, seen )
    if not tracing and plain == { "numpy" }:
        return "numpy"           # `promote` lets numpy give way to anything: here, nothing else was given
    default = framework_defaults.framework_name()
    return promote( tracing, plain, default ) or default


def _to_framework( raw, target ):
    """`raw` (an array of some framework, or `None` for a symbolic zero) as an array of `target`."""
    from .drivers import framework_defaults
    if raw is None or framework_defaults.framework_name_of( raw ) in ( target, None ):
        return raw
    from .tensor.storage.Storage import Storage
    return Storage.of( raw ).to_framework( target ).raw
