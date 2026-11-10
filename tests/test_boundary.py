from errand import test

# What a package exports hands back the caller's arrays, and views of its objects: loom tensors stay inside
# ( see `loom/boundary.py` ). Each test builds a small "package": a namespace given to `export_namespace`.


def _package( **content ):
    """a namespace as the `globals()` of a package `pkg`, whose functions and classes are `content`"""
    import loom
    ns = { "__name__": "pkg" }
    for name, value in content.items():
        value.__module__ = "pkg"
        ns[ name ] = value
    loom.export_namespace( ns )
    return ns


if test( "a_public_function_returns_the_framework_of_its_arguments" ):
    import numpy
    import loom
    from loom import RealTensor

    def double( x ):
        return RealTensor( x ) * 2

    pub = _package( double = double )[ "double" ]
    res = pub( numpy.arange( 3.0 ) )
    assert isinstance( res, numpy.ndarray ), type( res )
    assert res.tolist() == [ 0, 2, 4 ]

    # the internal function, imported from its module, keeps the loom tensor
    assert isinstance( double( numpy.arange( 3.0 ) ), loom.Tensor )

    # a caller that gives loom tensors lives in loom: nothing is converted
    assert isinstance( pub( RealTensor( numpy.arange( 3.0 ) ) ), loom.Tensor )

    try:
        import jax.numpy as jnp
    except ImportError:
        jnp = None
    if jnp is not None:
        import jax
        res = pub( jnp.arange( 3.0 ) )
        assert isinstance( res, jax.Array ), type( res )
        # and it stays a node of jax's graph
        assert jax.grad( lambda x: pub( x ).sum() )( jnp.arange( 3.0 ) ).tolist() == [ 2, 2, 2 ]


if test( "the_internals_only_see_loom_tensors" ):
    # the data is HOMOGENIZED at the entry: whatever the framework, the internal function receives loom
    # tensors ( containers looked into ), and python scalars as they are -- one code for every framework
    import numpy
    import loom

    seen = {}

    def inside( x, pair, scale ):
        seen.update( x = x, pair = pair, scale = scale )
        return x * scale + pair[ 0 ]

    pub = _package( inside = inside )[ "inside" ]
    res = pub( numpy.arange( 3.0 ), [ numpy.ones( 3 ) ], 2.0 )
    assert isinstance( seen[ "x" ], loom.Tensor ), type( seen[ "x" ] )
    assert isinstance( seen[ "pair" ][ 0 ], loom.Tensor ), type( seen[ "pair" ][ 0 ] )
    assert seen[ "scale" ] == 2.0 and isinstance( seen[ "scale" ], float )
    assert isinstance( res, numpy.ndarray ) and res.tolist() == [ 1, 3, 5 ], res


if test( "inside_a_call_a_loom_tensor_counts_by_its_buffer" ):
    # the tensors the boundary made say nothing of the caller: inside, `universe_of` reads their buffers
    # ( a loom tensor given by a caller, outside, still says "loom" )
    import numpy
    import loom

    def inside( x ):
        return loom.universe_of( ( x, ) )

    pub = _package( inside = inside )[ "inside" ]
    assert pub( numpy.arange( 3.0 ) ) == "numpy"
    assert loom.universe_of( ( loom.Tensor.as_tensor( numpy.arange( 3.0 ) ), ) ) == "loom"


if test( "a_public_function_called_from_inside_stays_in_loom" ):
    # after `export_namespace`, the globals of the module are the PUBLIC versions: a function of the package
    # calling another one by its name must still get loom values back, not arrays of the caller
    import numpy
    import loom

    def double( x ):
        return x * 2

    ns = _package( double = double )

    def outer( x ):
        y = ns[ "double" ]( x )          # the public name, as the globals of the module would give it
        assert isinstance( y, loom.Tensor ), type( y )
        return y

    pub = _package( outer = outer )[ "outer" ]
    res = pub( numpy.arange( 3.0 ) )
    assert isinstance( res, numpy.ndarray ) and res.tolist() == [ 0, 2, 4 ], res


if test( "an_object_whose_getattr_raises_is_left_alone" ):
    # entering a call must not read public attributes of the package's objects: a lazy `__getattr__`
    # may raise something else than `AttributeError` for them ( only the `_` names are the boundary's )
    class Lazy:
        def __getattr__( self, name ):
            if name.startswith( "_" ):
                raise AttributeError( name )
            raise ValueError( "not known yet" )

    lazy = Lazy()
    pub = _package( inside = lambda o: o )[ "inside" ]
    assert pub( lazy ) is lazy


if test( "containers_are_looked_into_foreign_objects_are_not" ):
    import numpy
    from loom import RealTensor

    class Foreign:
        pass

    box = Foreign()

    def several( x ):
        t = RealTensor( x )
        return t, [ t + 1 ], { "k": t }, box

    several = _package( several = several )[ "several" ]
    a, ( b, ), d, o = several( numpy.zeros( 2 ) )
    assert all( isinstance( v, numpy.ndarray ) for v in ( a, b, d[ "k" ] ) )
    assert o is box


if test( "an_object_of_an_exported_class_leaves_as_a_view" ):
    import numpy
    import loom
    from loom import RealTensor, Aggregate

    class Holder( Aggregate ):
        """holds values"""
        values : RealTensor

        def __init__( self, values ):
            self.__base_init__( values = values )

        def twice( self ):
            return self.values * 2

        def same( self, other ):
            return other is self

        @staticmethod
        def make( n ):
            return Holder( numpy.arange( float( n ) ) )

    def build( x ):
        return Holder( x )

    ns = _package( Holder = Holder, build = build )
    PubHolder = ns[ "Holder" ]

    h = ns[ "build" ]( numpy.arange( 2.0 ) )
    assert isinstance( h, loom.View ) and isinstance( h, PubHolder )
    assert isinstance( h.values, numpy.ndarray )                        # an attribute, converted
    assert isinstance( h.twice(), numpy.ndarray )                       # a method, a public function
    assert h.same( h )                                                  # a view given back is unwrapped
    assert isinstance( h._loom_obj, Holder )

    # the public class builds, and `isinstance` accepts the internal objects too
    h2 = PubHolder( numpy.arange( 3.0 ) )
    assert isinstance( h2, PubHolder ) and isinstance( Holder( numpy.zeros( 1 ) ), PubHolder )
    assert PubHolder.__doc__ == "holds values"
    assert isinstance( PubHolder.make( 2 ), PubHolder )                 # a static method, public too

    # what is written to a view goes to the object, unwrapped
    h.other = h2
    assert h._loom_obj.other is h2._loom_obj

    # given loom tensors, the caller lives in loom: the object itself
    assert isinstance( ns[ "build" ]( RealTensor( numpy.arange( 2.0 ) ) ), Holder )

    # a public class cannot be derived
    try:
        class Derived( PubHolder ):
            pass
        assert False, "derived"
    except TypeError:
        pass


if test( "an_object_built_from_nothing_is_neutral" ):
    import numpy
    import loom
    from loom import RealTensor

    class Settings:
        def __init__( self, tol = 1e-8 ):
            self.tol = tol

        def scaled( self, x ):
            return RealTensor( x ) * self.tol

    Pub = _package( Settings = Settings )[ "Settings" ]
    s = Pub( tol = 2.0 )
    assert object.__getattribute__( s, "_universe" ) is None
    assert s.tol == 2.0
    # a neutral view weighs nothing: the numpy argument decides
    assert isinstance( s.scaled( numpy.arange( 2.0 ) ), numpy.ndarray )


if test( "a_callback_receives_the_callers_values" ):
    import numpy
    import loom
    from loom import RealTensor

    def each( x, callback ):
        t = RealTensor( x )
        for k in range( 2 ):
            callback( k, t * k )
        return t

    each = _package( each = each )[ "each" ]
    seen = []
    each( numpy.arange( 3.0 ), lambda k, v: seen.append( v ) )
    assert all( isinstance( v, numpy.ndarray ) for v in seen ), [ type( v ) for v in seen ]


if test( "an_exception_carries_views" ):
    import numpy
    from loom import RealTensor, Aggregate

    class Result( Aggregate ):
        values : RealTensor

        def __init__( self, values ):
            self.__base_init__( values = values )

    class Failed( RuntimeError ):
        def __init__( self, result ):
            super().__init__( "failed" )
            self.result = result

    def fail( x ):
        raise Failed( Result( x ) )

    ns = _package( Result = Result, Failed = Failed, fail = fail )
    assert ns[ "Failed" ] is Failed                                     # an exception keeps its class
    try:
        ns[ "fail" ]( numpy.arange( 2.0 ) )
        assert False
    except Failed as e:
        assert isinstance( e.result, ns[ "Result" ] ) and isinstance( e.result.values, numpy.ndarray )


if test( "a_declared_universe_is_not_moved_by_what_the_internals_cache" ):
    import numpy
    import loom
    from loom import RealTensor, Aggregate

    class Source( Aggregate ):
        values : RealTensor
        cache  : RealTensor

        def __init__( self, values ):
            self._universe = loom.universe_of( ( values, ) )
            self.__base_init__( values = values )

    def total( src ):
        res = RealTensor( src.values.value ).sum()
        src.cache = RealTensor( loom.array( [ 1.0 ] ) )     # cached in the default framework
        return res

    ns = _package( Source = Source, total = total )
    src = ns[ "Source" ]( numpy.arange( 3.0 ) )
    for _ in range( 2 ):
        assert isinstance( ns[ "total" ]( src ), ( numpy.ndarray, numpy.generic ) ), loom.universe_of( ( src, ) )


if test( "the_internals_run_on_the_driver_of_the_callers_framework" ):
    # Jax tracers in a Torch process: what the internals decide from the driver ( `loom.resolved_framework()`, the device:
    # the card or the CPU ) follows the arguments, as their calls do -- sdot built its tree for the card ( torch's ) and ran
    # the call on Jax's CPU. Numpy gives way to the default
    import numpy
    import loom
    from loom.drivers import framework_defaults

    def seen( x ):
        return str( loom.resolved_framework() )

    pub = _package( seen = seen )[ "seen" ]
    default = framework_defaults.framework_name()
    assert pub( numpy.arange( 3.0 ) ) == default
    others = []
    try:
        import jax.numpy as jnp
        others.append( ( "jax", jnp.arange( 3.0 ) ) )
    except ImportError:
        pass
    try:
        import torch
        others.append( ( "torch", torch.arange( 3.0 ) ) )
    except ImportError:
        pass
    for name, x in others:
        assert pub( x ) == name, ( default, name, pub( x ) )
    assert framework_defaults.framework_name() == default            # ( and back to the default after the call )
