"""THE BOUNDARY between loom and the user: what a package exports leaves the loom universe for the
caller's.

Inside, every value is a loom `Tensor` (a `RealTensor` around a jax / torch / numpy buffer), and every
object is the package's own: that is what the internals compute with, and what they hand each other.
The user, for their part, works in ONE framework and expects its arrays back -- `sdot.w2( f, g )` on
jax inputs is a jax array, so that `jax.grad` of it works without knowing that loom exists.

THE BOUNDARY IS THE NAMESPACE OF THE PACKAGE, set in one line at the end of its `__init__.py`:

    loom.export_namespace( globals() )

which replaces every function and every class the package exports by its PUBLIC version. Nothing is
decorated: the internals import from the submodules ( `from .OtProblem import OtProblem` ), which are
not wrapped, and never see anything but loom values.

    sdot.w2( f, g )                       # a jax array, when `f` and `g` were made of jax arrays
    p = sdot.plan( f, g )                 # a VIEW of the plan: `p.barycenters` is a jax array

* A public function reads the UNIVERSE of its arguments, UNWRAPS them ( a view becomes the object it
  shows, a callback is wrapped so that what the internals give it is exported ), calls the internal
  function, and EXPORTS the result.
* An object of an exported class leaves as a VIEW ( `View` ): it keeps the universe, converts its
  attributes when they are read, and its methods are public functions in which it takes part as an
  argument of its universe. A name in `_` is read raw: the service door. What is written to a view is
  unwrapped and written to the object.
* A public class builds the internal object and returns its view; `isinstance` accepts both.
* An exception that crosses the boundary keeps its class ( `except` must still catch it ): its
  public attributes are exported, and it is raised again.

THE UNIVERSE of a call is read off its arguments, with the rule of `ffi_call` (`drivers/promotion.py`):
the framework of the buffers they hold -- an `Aggregate` (a distribution) is looked into --, a tracing
buffer winning over plain memory. A loom `Tensor` given AS AN ARGUMENT says that the caller lives in
loom: the result is then handed back as it is, and no view is made. Arguments that hold no buffer
(lists, numbers) make a NEUTRAL call ( universe `None` ): an array it returns is in the default framework,
and an object it returns has no universe -- it weighs nothing in the calls it enters ( `Polygon( [ ... ] )`
given to a function with numpy arrays leaves the result in numpy ), and what is read on it comes out in the
default framework of the moment."""
import functools
import inspect


# the view class of each exported class, by internal class -- for every package that exported its namespace
_VIEWS = {}


def export_namespace( namespace ):
    """Replaces, in `namespace` ( the `globals()` of a package's `__init__.py` ), the functions and the classes that the
    package defines and exports ( a name without `_` ) by their public versions. The exceptions keep their class."""
    package = namespace[ "__name__" ]

    def ours( value ):
        module = getattr( value, "__module__", None ) or ""
        return module == package or module.startswith( package + "." )

    names = [ n for n, v in namespace.items() if not n.startswith( "_" ) and ours( v ) ]
    # the classes first: what a function returns must find its view
    for n in names:
        v = namespace[ n ]
        if isinstance( v, type ) and not issubclass( v, BaseException ):
            namespace[ n ] = view_class( v )
    for n in names:
        v = namespace[ n ]
        if inspect.isfunction( v ):
            namespace[ n ] = public_function( v )


def public_function( fn ):
    """`fn` behind the boundary: the universe read off the arguments, the arguments unwrapped, the result exported."""
    if getattr( fn, "_loom_public", False ):
        return fn

    @functools.wraps( fn )
    def call( *args, **kwargs ):
        return _call( fn, universe_of( args, kwargs, neutral = True ), args, kwargs )

    call._loom_public = True
    return call


def view_class( cls ):
    """The public class of `cls`: building it builds a `cls` and returns its `View`. Made once per class."""
    res = _VIEWS.get( cls )
    if res is None:
        res = _ViewMeta( cls.__name__, ( View, ), {
            "__slots__"    : (),
            "_loom_cls"    : cls,
            "__doc__"      : cls.__doc__,
            "__module__"   : cls.__module__,
            "__qualname__" : cls.__qualname__,
        } )
        try:
            res.__signature__ = inspect.signature( cls )
        except ( TypeError, ValueError ):
            pass
        _VIEWS[ cls ] = res
    return res


class _ViewMeta( type ):
    """What is read on a public CLASS is read on the internal one: a class method, a static method, an unbound method
    come out as public functions; anything else as it is. `isinstance` accepts a view as well as an internal object."""

    def __getattr__( vcls, name ):
        cls = type.__getattribute__( vcls, "__dict__" ).get( "_loom_cls" )
        if cls is None:
            raise AttributeError( name )
        value = getattr( cls, name )
        if name.startswith( "_" ) or isinstance( value, type ):
            return _VIEWS.get( value, value )
        if callable( value ):
            return public_function( value )
        return value

    def __instancecheck__( vcls, obj ):
        if type.__instancecheck__( vcls, obj ):
            return True
        cls = type.__getattribute__( vcls, "__dict__" ).get( "_loom_cls" )
        return cls is not None and isinstance( unwrap( obj ), cls )


class View( metaclass = _ViewMeta ):
    """An internal object, seen from the universe of its user ( see the module ). `_loom_obj` is the object,
    `_universe` the framework its values are converted to ( `None`: the default one, at each read )."""

    __slots__ = ( "_loom_obj", "_universe" )

    def __new__( vcls, *args, **kwargs ):
        return _call( vcls._loom_cls, universe_of( args, kwargs, neutral = True ), args, kwargs )

    def __init__( self, *args, **kwargs ):
        pass                                  # all done by `__new__`

    def __init_subclass__( cls, **kwargs ):
        for base in cls.__bases__:
            if "_loom_cls" in base.__dict__:
                inner = base.__dict__[ "_loom_cls" ]
                raise TypeError( f"{ base.__name__ } is the PUBLIC class of `{ inner.__module__ }.{ inner.__qualname__ }`, which "
                                 "cannot be derived: derive the internal class, imported from its module" )
        super().__init_subclass__( **kwargs )

    @staticmethod
    def _make( obj, universe ):
        res = object.__new__( _view_class_of( type( obj ) ) )
        object.__setattr__( res, "_loom_obj", obj )
        object.__setattr__( res, "_universe", universe )
        return res

    def __getattr__( self, name ):
        obj = object.__getattribute__( self, "_loom_obj" )
        value = getattr( obj, name )
        if name.startswith( "_" ):
            return value
        if inspect.ismethod( value ) and value.__self__ is obj:
            return _bound( value, self )
        return export( value, object.__getattribute__( self, "_universe" ) )

    def __setattr__( self, name, value ):
        setattr( object.__getattribute__( self, "_loom_obj" ), name, unwrap( value ) )

    def __delattr__( self, name ):
        delattr( object.__getattribute__( self, "_loom_obj" ), name )

    def __dir__( self ):
        return dir( object.__getattribute__( self, "_loom_obj" ) )

    def __repr__( self ):
        return repr( object.__getattribute__( self, "_loom_obj" ) )

    def __str__( self ):
        return str( object.__getattribute__( self, "_loom_obj" ) )

    def __eq__( self, other ):
        return object.__getattribute__( self, "_loom_obj" ) is unwrap( other )

    def __hash__( self ):
        return id( object.__getattribute__( self, "_loom_obj" ) )

    def __bool__( self ):
        return bool( object.__getattribute__( self, "_loom_obj" ) )

    def __len__( self ):
        return len( object.__getattribute__( self, "_loom_obj" ) )

    def __iter__( self ):
        universe = object.__getattribute__( self, "_universe" )
        return ( export( v, universe ) for v in object.__getattribute__( self, "_loom_obj" ) )

    def __getitem__( self, key ):
        return export( object.__getattribute__( self, "_loom_obj" )[ unwrap( key ) ], object.__getattribute__( self, "_universe" ) )

    def __call__( self, *args, **kwargs ):
        obj = object.__getattribute__( self, "_loom_obj" )
        return _call( obj, universe_of( ( self, *args ), kwargs, neutral = True ), args, kwargs )


def _view_class_of( cls ):
    """the view class of the nearest exported class among the ancestors of `cls`, or None"""
    for c in cls.__mro__:
        res = _VIEWS.get( c )
        if res is not None:
            return res
    return None


def _bound( method, view ):
    """a method of the object behind `view`, as a public function: the view takes part in the universe of the call"""
    @functools.wraps( method )
    def call( *args, **kwargs ):
        return _call( method, universe_of( ( view, *args ), kwargs, neutral = True ), args, kwargs )
    return call


def _call( fn, universe, args, kwargs ):
    """`fn( *args, **kwargs )` across the boundary, in `universe`"""
    args = tuple( _unwrap( a, universe ) for a in args )
    kwargs = { k: _unwrap( v, universe ) for k, v in kwargs.items() }
    try:
        res = fn( *args, **kwargs )
    except BaseException as e:
        if universe != "loom":
            for k, v in list( vars( e ).items() ):
                if not k.startswith( "_" ):
                    setattr( e, k, export( v, universe ) )
        raise
    return export( res, universe )


def unwrap( value ):
    """What the internals see of `value`: the object behind a view, containers looked into. Anything else as it is."""
    return _unwrap( value, "loom" )


def _unwrap( value, universe ):
    if isinstance( value, View ):
        return object.__getattribute__( value, "_loom_obj" )
    if isinstance( value, ( list, tuple, dict ) ):
        items = value.values() if isinstance( value, dict ) else value
        if not any( isinstance( v, ( View, list, tuple, dict ) ) or _is_callback( v ) for v in items ):
            return value          # nothing to unwrap: not even a copy
        if isinstance( value, dict ):
            return { k: _unwrap( v, universe ) for k, v in value.items() }
        res = [ _unwrap( v, universe ) for v in value ]
        if isinstance( value, list ):
            return res
        return type( value )( *res ) if hasattr( value, "_fields" ) else type( value )( res )
    if universe != "loom" and _is_callback( value ):
        return _callback( value, universe )
    return value


def _is_callback( value ):
    """a function of the user: what the internals call is then exported to them"""
    return ( inspect.isfunction( value ) or inspect.ismethod( value ) or isinstance( value, functools.partial ) ) \
        and not getattr( value, "_loom_public", False )


def _callback( fn, universe ):
    @functools.wraps( fn )
    def call( *args, **kwargs ):
        return _unwrap( fn( *export( args, universe ), **export( kwargs, universe ) ), universe )
    call._loom_public = True     # already at the boundary: not wrapped again
    return call


def export( value, universe ):
    """`value` as the user of `universe` expects it: a loom `Tensor` becomes an array of that framework
    (its LOGICAL value, padding cropped), a `ShapeVar` its count, an object of an exported class its `View`;
    tuples, lists and dicts are looked into. Anything else is handed back as it is.

    `universe`: a framework name, `"loom"` (nothing is converted), or `None` ( neutral: the default
    framework at the time of the conversion ) -- what `universe_of` says."""
    if universe == "loom":
        return value

    from .tensor.Tensor import Tensor
    from .tensor.ShapeVar import ShapeVar
    from .tensor.storage.buffers import buffer_class_for

    if isinstance( value, Tensor ):
        return _to_framework( value.value, universe )
    if isinstance( value, ShapeVar ):
        return value.value
    if value is None or isinstance( value, ( str, int, float, bool, View, type ) ):
        return value
    if isinstance( value, tuple ):
        return type( value )( *( export( v, universe ) for v in value ) ) if hasattr( value, "_fields" ) else \
               tuple( export( v, universe ) for v in value )
    if isinstance( value, list ):
        return [ export( v, universe ) for v in value ]
    if isinstance( value, dict ):
        return { k: export( v, universe ) for k, v in value.items() }
    if _view_class_of( type( value ) ) is not None:
        return View._make( value, universe )
    if buffer_class_for( value ) is not None:
        # a bare array of a framework, computed inside: it crosses too
        return _to_framework( value, universe )
    return value


def universe_of( args = (), kwargs = None, neutral = False ):
    """The universe of a call made with `args` / `kwargs`: `"loom"` when one of them IS a loom
    `Tensor`, else the framework their buffers promote to (see `drivers/promotion.py`), else the
    default framework -- or `None` when `neutral` ( a constructor: see the module ).

    An object can DECLARE its universe in a `_universe` attribute, which is then taken instead of
    looking into it -- what a view does, and what an object of the API does at construction (a
    distribution: see sdot's `Distribution.__base_init__`), because what it holds later also contains
    what the internals cached in it, of whatever framework they computed in. A neutral view weighs nothing."""
    from .tensor.Tensor import Tensor
    from .calls import _scan
    from .drivers.promotion import promote
    from .drivers import framework_defaults

    tracing, plain, seen = set(), set(), set()
    for value in ( *args, *( kwargs or {} ).values() ):
        if isinstance( value, Tensor ):
            return "loom"
        if isinstance( value, View ):
            declared = object.__getattribute__( value, "_universe" )
            if declared is not None:
                plain.add( declared )
            continue
        declared = getattr( value, "_universe", None ) if not isinstance( value, ( str, int, float, bool ) ) else None
        if declared == "loom":
            return "loom"
        if isinstance( declared, str ):
            plain.add( declared )
        else:
            _scan( value, tracing, plain, seen )
    if not tracing and plain == { "numpy" }:
        return "numpy"           # `promote` lets numpy give way to anything: here, nothing else was given
    if neutral and not tracing and not plain:
        return None
    default = framework_defaults.framework_name()
    return promote( tracing, plain, default ) or default


def _to_framework( raw, target ):
    """`raw` (an array of some framework, or `None` for a symbolic zero) as an array of `target` ( `None`: the default one )."""
    from .drivers import framework_defaults
    if target is None:
        target = framework_defaults.framework_name()
    if raw is None or framework_defaults.framework_name_of( raw ) in ( target, None ):
        return raw
    from .tensor.storage.Storage import Storage
    return Storage.of( raw ).to_framework( target ).raw
