from loom.tensor import ReferenceShape
from loom.testing import test
import numpy


if test( "dense" ):
    # every dimension's extent is uniform: `sizes` is 0-d everywhere, no dependency
    shape = ReferenceShape.from_value( [ [ 1, 2 ], [ 3, 4 ] ] )

    assert len( shape ) == 2
    assert not shape.is_ragged()

    assert shape.dep_dims( 0 ) == ()
    assert shape.sizes( 0 ).tolist() == 2

    assert shape.dep_dims( 1 ) == ()
    assert shape.sizes( 1 ).tolist() == 2

    assert shape.capacities() == [ 2, 2 ]


if test( "scalar" ):
    # no array dimension at all
    shape = ReferenceShape.from_value( 5 )

    assert len( shape ) == 0
    assert not shape.is_ragged()
    assert shape.capacities() == []


if test( "ragged, single dependency" ):
    # the 2nd dimension's extent varies along the 1st
    shape = ReferenceShape.from_value( [ [ 10, 11 ], [ 12 ] ] )

    assert len( shape ) == 2
    assert shape.is_ragged()

    assert shape.dep_dims( 0 ) == ()
    assert shape.sizes( 0 ).tolist() == 2

    assert shape.dep_dims( 1 ) == ( 0, )
    assert shape.sizes( 1 ).tolist() == [ 2, 1 ]

    assert shape.capacities() == [ 2, 2 ]


if test( "ragged, dependency on two outer dimensions" ):
    # the innermost dimension's extent varies along BOTH outer dimensions ; the outer two stay
    # dense since they are uniform (2 batches, 2 cells per batch)
    value = [ [ [ 1, 2 ], [ 3 ] ], [ [ 4 ], [ 5, 6, 7 ] ] ]
    shape = ReferenceShape.from_value( value )

    assert len( shape ) == 3
    assert shape.is_ragged()

    assert shape.dep_dims( 0 ) == ()
    assert shape.sizes( 0 ).tolist() == 2

    assert shape.dep_dims( 1 ) == ()
    assert shape.sizes( 1 ).tolist() == 2

    assert shape.dep_dims( 2 ) == ( 0, 1 )
    assert shape.sizes( 2 ).tolist() == [ [ 2, 1 ], [ 1, 3 ] ]

    assert shape.capacities() == [ 2, 2, 3 ]


if test( "leaves are read data-free, from `.shape` alone" ):
    # a leaf need not be a scalar: anything with a `.shape` is treated as an array leaf, and only
    # that metadata is read -- no data is ever touched
    value = [ numpy.zeros( ( 2, ) ), numpy.zeros( ( 3, ) ) ]
    shape = ReferenceShape.from_value( value )

    assert len( shape ) == 2

    assert shape.dep_dims( 0 ) == ()
    assert shape.sizes( 0 ).tolist() == 2

    assert shape.dep_dims( 1 ) == ( 0, )
    assert shape.sizes( 1 ).tolist() == [ 2, 3 ]


if test( "from_dense_shape" ):
    # a padding-free buffer shape is recorded as fully dense, whatever its rank
    shape = ReferenceShape.from_dense_shape( ( 4, 5, 6 ) )

    assert len( shape ) == 3
    assert not shape.is_ragged()

    for d, expected in enumerate( ( 4, 5, 6 ) ):
        assert shape.dep_dims( d ) == ()
        assert shape.sizes( d ).tolist() == expected

    assert shape.capacities() == [ 4, 5, 6 ]
