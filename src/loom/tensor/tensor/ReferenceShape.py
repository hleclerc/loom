import numpy


class ReferenceShape:
    """The LOGICAL (unpadded) shape of a value: one `( sizes, dep_dims )` entry per array dimension,
    describing only the nesting/`.shape` structure of that value -- nothing else (no axes, no buffer).

      * `sizes`    -- a numpy array of the extents along that dimension: 0-d when the dimension is
                      dense, else an array indexed by `dep_dims` (a RAGGED dimension);
      * `dep_dims` -- the (outer) dimensions its extent varies along, an empty tuple when dense.

    A dense rank-2 value gives `[ ( array(2), () ), ( array(3), () ) ]`; a value whose 2nd dimension
    is ragged along the 1st gives `[ ( array(2), () ), ( array([ 2, 1 ]), ( 0, ) ) ]`.

    Immutable once built (`from_value` / `from_dense_shape`), so it can be shared -- e.g. `Tensor`
    uses it as the source of truth its `ShapeVar`s pull their counts from, independent of the
    (possibly padded) buffer and of the declared axes."""

    def __init__( self, dims ):
        self.dims = dims   # list of ( numpy array, tuple of dependency dimension indices )

    def __len__( self ):
        return len( self.dims )

    def sizes( self, dim ):
        """Extents along array dimension `dim`: a 0-d array (dense) or one indexed by `dep_dims(dim)`."""
        return self.dims[ dim ][ 0 ]

    def dep_dims( self, dim ):
        """The array dimensions `dim`'s extent varies along (empty when `dim` is dense)."""
        return self.dims[ dim ][ 1 ]

    def is_ragged( self ):
        """Whether any dimension's extent varies (its `sizes` is not 0-d): then the value is jagged
        and its buffer must be ASSEMBLED with padding rather than built as a plain dense array."""
        return any( sizes.ndim > 0 for sizes, _ in self.dims )

    def capacities( self ):
        """One capacity per dimension for a padded allocation: the max extent along each."""
        return [ int( sizes.max() ) for sizes, _ in self.dims ]

    @classmethod
    def from_value( cls, value ):
        """Read the reference shape off `value` WITHOUT touching its data (a GPU tensor is never
        moved): only the list nesting and each leaf's `.shape` metadata are inspected. A dimension
        whose extent is uniform across its outer dimensions is recorded as DENSE (0-d `sizes`); a
        non-uniform one keeps its per-segment extents and the dimensions they vary along."""
        tree = _shape_tree( value )
        dims = []
        for d in range( _tree_ndim( tree ) ):
            sizes, dropped = _collapse_uniform( _query( tree, d ) )
            dims.append( ( numpy.array( sizes, dtype = int ), tuple( range( dropped, d ) ) ) )
        return cls( dims )

    @classmethod
    def from_dense_shape( cls, shape ):
        """A fully DENSE reference shape from a padding-free buffer shape -- what `Tensor.wrap`
        records around an op result or a `ShapeVar`'s backend array (their buffers carry no padding,
        so their logical sizes ARE their shape)."""
        return cls( [ ( numpy.array( s, dtype = int ), () ) for s in shape ] )

    def appended_dense( self, size = 1 ):
        """A new `ReferenceShape` with one more DENSE array dimension of extent `size` appended --
        what `Tensor.append_axis` needs for a literal axis whose extent is not read off any data."""
        return ReferenceShape( self.dims + [ ( numpy.array( size, dtype = int ), () ) ] )


# ---- reading a value's size structure, data-free ----

# containers recursed into by `_shape_tree` (a whitelist: anything else is a leaf)
_containers = ( list, tuple )


def _shape_tree( value ):
    """Recursive size descriptor of `value`, WITHOUT touching its data.

    A whitelisted container becomes a list of child descriptors; anything else is treated as an
    array and described by its `.shape` (metadata only -- a GPU tensor is never moved); a scalar has
    shape `()`."""
    if isinstance( value, _containers ):
        return [ _shape_tree( v ) for v in value ]
    shape = getattr( value, "shape", None )
    return tuple( shape ) if shape is not None else ()


def _tree_ndim( tree ):
    """Number of array dimensions the shape tree describes: the list-nesting depth plus the rank of
    its leaf (a leaf `.shape` tuple). A jagged tree is still rectangular in DEPTH, so the first child
    suffices."""
    if isinstance( tree, list ):
        return 1 + ( _tree_ndim( tree[ 0 ] ) if tree else 0 )
    return len( tree )


def _query( tree, d ):
    """Size structure along array dimension `d`, read from a shape tree: a scalar when the outer
    dimensions are dense, a nested list of ints when they are ragged (nested over dims `0..d-1`)."""
    if isinstance( tree, list ):
        return len( tree ) if d == 0 else [ _query( child, d - 1 ) for child in tree ]
    return tree[ d ]


def _collapse_uniform( sizes ):
    """Drop the OUTERMOST dimensions of `sizes` that are uniform (their extent does not actually vary
    along them), returning `( collapsed, dropped )`. `dropped` counts how many outer dimensions were
    removed, so the caller knows which dimensions the survivors are indexed by. A scalar (dense) is
    returned untouched."""
    dropped = 0
    while isinstance( sizes, list ) and all( s == sizes[ 0 ] for s in sizes ):
        sizes = sizes[ 0 ]
        dropped += 1
    return sizes, dropped
