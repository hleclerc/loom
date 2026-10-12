from errand import test

# A public function hands back the caller's arrays: loom tensors stay inside (see `loom/boundary.py`).

if test( "a_public_function_returns_the_framework_of_its_arguments" ):
    import numpy
    import loom
    from loom import RealTensor

    @loom.public
    def double( x ):
        return RealTensor( x ) * 2

    res = double( numpy.arange( 3.0 ) )
    assert isinstance( res, numpy.ndarray ), type( res )
    assert res.tolist() == [ 0, 2, 4 ]

    # internal code calls `.internal`, and keeps the loom tensor
    assert isinstance( double.internal( numpy.arange( 3.0 ) ), loom.Tensor )

    # a caller that gives loom tensors lives in loom: nothing is converted
    assert isinstance( double( RealTensor( numpy.arange( 3.0 ) ) ), loom.Tensor )

    try:
        import jax.numpy as jnp
    except ImportError:
        jnp = None
    if jnp is not None:
        import jax
        res = double( jnp.arange( 3.0 ) )
        assert isinstance( res, jax.Array ), type( res )
        # and it stays a node of jax's graph
        assert jax.grad( lambda x: double( x ).sum() )( jnp.arange( 3.0 ) ).tolist() == [ 2, 2, 2 ]


if test( "containers_are_looked_into_objects_are_not" ):
    import numpy
    import loom
    from loom import RealTensor

    class Box:
        pass

    box = Box()

    @loom.public
    def several( x ):
        t = RealTensor( x )
        return t, [ t + 1 ], { "k": t }, box

    a, ( b, ), d, o = several( numpy.zeros( 2 ) )
    assert all( isinstance( v, numpy.ndarray ) for v in ( a, b, d[ "k" ] ) )
    assert o is box


if test( "a_method_takes_its_instance_into_the_universe" ):
    import numpy
    import loom
    from loom import RealTensor, Aggregate

    class Holder( Aggregate ):
        values : RealTensor

        def __init__( self, values ):
            self.__base_init__( values = values )

        @loom.public
        def twice( self ):
            return self.values * 2

    h = Holder( numpy.arange( 2.0 ) )
    assert isinstance( h.twice(), numpy.ndarray )
    assert isinstance( h.twice.internal(), loom.Tensor )
    assert Holder.twice.internal is not None and Holder.twice.__name__ == "twice"


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

    @loom.public
    def total( src ):
        res = RealTensor( src.values.value ).sum()
        src.cache = RealTensor( loom.array( [ 1.0 ] ) )     # cached in the default framework
        return res

    src = Source( numpy.arange( 3.0 ) )
    for _ in range( 2 ):
        assert isinstance( total( src ), numpy.ndarray ), loom.universe_of( ( src, ) )
