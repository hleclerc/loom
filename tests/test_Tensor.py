from loom import ShapeVar, Axis, AxisList, Tensor, Aggregate, driver
from loom.testing import test
import numpy

if test( "basic" ):
    class Cell( Aggregate ):
        vertex_positions : Tensor[ "num_vertex", "dim" ]
        vertex_indices   : Tensor[ "num_vertex", "dim", { "dtype": int } ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : ShapeVar



    nb_dims = ShapeVar()
    c = Cell( nb_dims = nb_dims )

    c.vertex_positions = [ [ 1, 2 ] ]
    # Vérifier que le dtype est bien extracté

    info( c.nb_vertices.value )

    # `c.nb_vertices` is the `ShapeVar` itself; its `.value` is the solved count
    assert c.nb_vertices.value == 1
    assert c.nb_dims.value == 2



if test( "ragged" ):
    class Mesh( Aggregate ):
        cell_vertices   : Tensor[ "cell", "vtx" ]

        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]      # ragged: depends on `cell`

        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]             # rank-1 (one count per cell)



    m = Mesh()
    m.cell_vertices = [ [ 10, 11 ], [ 12 ] ]   # cell 0 has 2 vertices, cell 1 has 1

    # sizes are read from the nesting only (no data touched)
    assert m.nb_cells.value == 2
    assert list( m.nb_vtx_per_cell.value ) == [ 2, 1 ]

    # values assembled into a padded rank-2 buffer (pad = 0), functionally
    import numpy
    raw = numpy.asarray( m.cell_vertices.raw )
    assert raw.shape == ( 2, 2 )
    assert raw.tolist() == [ [ 10, 11 ], [ 12, 0 ] ]

    info( m.nb_vtx_per_cell )


if test( "AxisList" ):
    class Image( Aggregate ):
        values  : Tensor[ "img_pos..." ]
        knots   : Tensor[ "dim", "num_knot" ]

        num_knot: Axis[ "extent + 1" ]          # ragged over `dim` via `extent`
        img_pos : AxisList[ "dim", "extent" ]    # unrolled into `nb_dims` static axes
        dim     : Axis[ "nb_dims" ]

        nb_dims : ShapeVar
        extent  : ShapeVar[ "dim" ]             # rank-1 (one count per dim)



    m = Image( values = driver.random( [ 2, 1 ] ), knots = [ [ 0, 1, 2 ], [ 0, 1 ] ] )
    assert list( m.extent.value ) == [ 2, 1 ]
    assert m.nb_dims.value == 2


if test( "axis_parsing" ):
    class Dell( Aggregate ):
        x: ShapeVar
        y: ShapeVar

        a1: Axis[ "x" ]
        a2: Axis[ "2 * x + 3" ]
        a3: Axis[ "x - 5" ]
        a4: Axis[ "3 * x + 2 * y - 1" ]
        a5: Axis[ "- x + 10" ]


    c = Dell()

    # Test simple variable
    assert len( c.a1.coeffs ) == 1 and c.a1.offset == 0

    # Test coefficient + constant
    assert len( c.a2.coeffs ) == 1 and c.a2.offset == 3
    coeff2 = list(c.a2.coeffs.values())[0]
    assert coeff2 == 2

    # Test subtraction
    assert len( c.a3.coeffs ) == 1 and c.a3.offset == -5

    # Test multiple variables
    assert len( c.a4.coeffs ) == 2 and c.a4.offset == -1

    # Test negative variable
    assert len( c.a5.coeffs ) == 1 and c.a5.offset == 10
    coeff5 = list(c.a5.coeffs.values())[0]
    assert coeff5 == -1

    info( "All axis parsing tests passed" )



# if test( "basic_tensor" ):
#     @aggregate
#     class Cell:
#         nb_dims = ShapeVar()
#         num     = Axis( nb_dims + 1 )
#         dim     = Axis( nb_dims )

#         frame   = Tensor( num, dim )

#         def __init__( self ) -> None:
#             self.pouet = 32

#     c = Cell()
#     # frame shape == [ num, dim ] == [ nb_dims + 1, nb_dims ] -> [ 3, 2 ] solves nb_dims = 2
#     c.frame = [ [ 0, 0 ], [ 1, 1 ], [ 2, 2 ] ]
#     assert c.nb_dims == 2
#     assert c.pouet == 32

#     # a prescribed value wins over what the tensors imply
#     c.nb_dims = 1222
#     assert c.nb_dims == 1222


# if test( "shared" ):
#     @aggregate
#     class Celm:
#         nb_dims = ShapeVar()
#         num     = Axis( nb_dims + 1 )
#         dim     = Axis( nb_dims )
#         frame   = Tensor( num, dim )

#     n = ShapeVar()
#     a = Celm( nb_dims = n )        # a and b share the same nb_dims cell
#     b = Celm( nb_dims = n )

#     a.frame = [ [ 0, 0 ], [ 1, 1 ], [ 2, 2 ] ]   # only a is given a value
#     assert a.nb_dims == 2
#     assert b.nb_dims == 2          # b sees it through the shared cell

# from typing import Any

# class MonTypeTemplate:
#     def __init__(self, params: Any):
#         self.params = params

#     # Permet la syntaxe MonTypeTemplate[int] ou MonTypeTemplate["X"]
#     def __class_getitem__(cls, item, **kw ):
#         # En production, vous pouvez retourner un objet proxy ou une instance spécialisée
#         return cls(params=item)

# @aggregate
# class Test:
#     a : MonTypeTemplate[ 132, a = 2 ]

if test( "indep" ):
    inp = Tensor( 17 ) # rank 0
    out = Tensor()

    yo = Tensor[ dict( dtype = int ) ]( [ 17, 18 ] ) # rank 1
    assert yo.shape == [ 2 ]
    assert out.shape == []

    nx = ShapeVar()
    ny = ShapeVar()
    x  = Axis( nx )   # outside an aggregate, an Axis takes the ShapeVar itself (no name to resolve)
    y  = Axis( ny )
    ya = Tensor[ x, y, dict( dtype = int ) ]( [ [ 17, 18 ] ] ) # rank 2 with named Axes

    # the ShapeVars are solved from the tensor, exactly as they would be in an aggregate
    assert nx.value == 1
    assert ny.value == 2

    info( inp )
    info( yo )
    info( ya )

    ya.value = [ [ 18, 19, 20 ] ] # reassign
    assert ny.value == 3
    info( ya )

    info( nx )
    info( ny )


# ---- the "sexy" Tensor API: operators, reductions, slicing, array protocol ----

def _named_2x3():
    """A rank-2 int tensor with named axes `row` (2) and `col` (3), value 1..6."""
    x = Axis( ShapeVar(), name = "row" )
    y = Axis( ShapeVar(), name = "col" )
    return Tensor[ x, y, dict( dtype = int ) ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )


if test( "tensor_ops" ):
    a = Tensor[ dict( dtype = int ) ]( [ 1, 2, 3 ] )
    b = Tensor[ dict( dtype = int ) ]( [ 10, 20, 30 ] )

    # scalar / tensor operands, left and right
    assert numpy.asarray( a + b ).tolist() == [ 11, 22, 33 ]
    assert numpy.asarray( b - a ).tolist() == [ 9, 18, 27 ]
    assert numpy.asarray( a * 2 ).tolist() == [ 2, 4, 6 ]
    assert numpy.asarray( 2 * a ).tolist() == [ 2, 4, 6 ]
    assert numpy.asarray( -a ).tolist()    == [ -1, -2, -3 ]

    # comparisons yield a boolean Tensor
    assert numpy.asarray( a >= 2 ).tolist() == [ False, True, True ]

    # result of an op is a Tensor, chainable
    assert numpy.asarray( ( a + b ) * 2 ).tolist() == [ 22, 44, 66 ]

    info( a + b )


if test( "tensor_reduce" ):
    t = _named_2x3()

    assert int( t.sum() ) == 21
    assert int( t.max() ) == 6
    assert int( t.min() ) == 1

    # reduce BY AXIS NAME -- drops that axis, keeps the other
    per_col = t.sum( "row" )                       # sum over rows -> one value per col
    assert numpy.asarray( per_col ).tolist() == [ 5, 7, 9 ]
    assert per_col._dim_names() == [ "col" ]

    # reduce BY POSITION
    per_row = t.sum( 1 )
    assert numpy.asarray( per_row ).tolist() == [ 6, 15 ]
    assert per_row._dim_names() == [ "row" ]


if test( "tensor_index" ):
    t = _named_2x3()

    r0 = t[ 0 ]                                    # first row -> the "col" axis survives
    assert numpy.asarray( r0 ).tolist() == [ 1, 2, 3 ]
    assert r0._dim_names() == [ "col" ]

    c1 = t[ :, 1 ]                                 # second column -> the "row" axis survives
    assert numpy.asarray( c1 ).tolist() == [ 2, 5 ]
    assert c1._dim_names() == [ "row" ]

    by_name = t[ "col", 2 ]                        # select column 2 by axis name
    assert numpy.asarray( by_name ).tolist() == [ 3, 6 ]
    assert by_name._dim_names() == [ "row" ]


if test( "tensor_transpose" ):
    t = _named_2x3()                               # axes `row` (2), `col` (3)

    tt = t.T                                        # reverse -> `col` (3), `row` (2)
    assert numpy.asarray( tt ).tolist() == [ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ]
    assert tt._dim_names() == [ "col", "row" ]

    # explicit permutation, by axis NAME -- the names follow the move
    by_name = t.transpose( "col", "row" )
    assert numpy.asarray( by_name ).tolist() == numpy.asarray( tt ).tolist()
    assert by_name._dim_names() == [ "col", "row" ]

    # matmul chains through it: t (2x3) @ t.T (3x2) -> 2x2
    g = t @ t.T
    assert numpy.asarray( g ).tolist() == [ [ 14, 32 ], [ 32, 77 ] ]


if test( "tensor_ref_broadcast" ):
    from loom import new_batch_axis

    row = Axis( ShapeVar(), name = "row" )
    col = Axis( ShapeVar(), name = "col" )
    m = Tensor[ row, col, dict( dtype = int ) ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )   # row(2) x col(3)
    v = Tensor[ col, dict( dtype = int ) ]( [ 10, 20, 30 ] )                       # shares the `col` OBJECT

    # an elementwise op MAPS by axis REFERENCE: the shared `col` object lines up, `row` (only in `m`)
    # broadcasts. Operand order sets the order of the NON-batch axes (first-seen), so `v * m` lists
    # `col` first -- the same tensor as `m * v`, merely stored transposed (downstream aligns by ref).
    assert ( m * v )._dim_names() == [ "row", "col" ]
    assert numpy.asarray( m * v ).tolist() == [ [ 10, 40, 90 ], [ 40, 100, 180 ] ]
    assert ( v * m )._dim_names() == [ "col", "row" ]
    assert numpy.asarray( v * m ).tolist() == [ [ 10, 40 ], [ 40, 100 ], [ 90, 180 ] ]

    # a DIFFERENT axis object does NOT align, even with the SAME name (reference, not name): the two
    # `col`s are distinct axes, so the op is their outer product over two separate dimensions.
    col2 = Axis( ShapeVar(), name = "col" )
    w = Tensor[ col2, dict( dtype = int ) ]( [ 1, 2, 3 ] )
    ow = v * w
    assert numpy.asarray( ow ).shape == ( 3, 3 )
    assert ow._dim_names() == [ "col", "col" ]

    # a BATCH axis sorts FIRST in the result layout, whichever operand carries it (and however the
    # non-batch axes are arranged). No manual reshape: a per-batch value spreads over the rest.
    b = new_batch_axis( 2 )
    bt = Tensor[ b, dict( dtype = int ) ]( [ 100, 200 ] )                          # batch(2)
    r = m * bt
    assert r._dim_names() == [ b.name, "row", "col" ]                              # batch leading
    assert ( bt * m )._dim_names() == [ b.name, "row", "col" ]
    assert numpy.asarray( r )[ 0 ].tolist() == [ [ 100, 200, 300 ], [ 400, 500, 600 ] ]
    assert numpy.asarray( r )[ 1 ].tolist() == [ [ 200, 400, 600 ], [ 800, 1000, 1200 ] ]


if test( "tensor_dot" ):
    # `dot` CONTRACTS over a shared axis object, assuming no axis order (unlike `@`). Here `xy` (2) is
    # shared; the free axes `a` (2) and `b` (3) survive -> a plain matrix product, but by REFERENCE.
    xy = Axis( ShapeVar(), name = "xy" )
    a  = Axis( ShapeVar(), name = "a" )
    b  = Axis( ShapeVar(), name = "b" )
    left  = Tensor[ a, xy, dict( dtype = int ) ]( [ [ 1, 0 ], [ 0, 2 ] ] )          # a(2) x xy(2)
    right = Tensor[ b, xy, dict( dtype = int ) ]( [ [ 1, 1 ], [ 2, 0 ], [ 0, 3 ] ] )  # b(3) x xy(2)

    out = left.dot( right, over = xy )                                              # contracts xy
    assert out._dim_names() == [ "a", "b" ]                                         # free axes survive
    # out[i,j] = sum_xy left[i,xy]*right[j,xy]
    assert numpy.asarray( out ).tolist() == [ [ 1, 2, 0 ], [ 2, 0, 6 ] ]
    # matches the positional matmul left @ right.T, but chosen by axis, not by order
    assert numpy.asarray( out ).tolist() == numpy.asarray( left @ right.T ).tolist()


if test( "tensor_physical_layout_view" ):
    # a Tensor whose buffer is laid out NON-contiguously (batch axis flattened + padded) still reads
    # back its LOGICAL values: `.tensor` gathers them through the layout (the physical<->logical
    # boundary). Everything else reads `.tensor`, so ops/results stay logical whatever the storage.
    from loom.tensor import PhysicalLayout
    from loom.tensor import ReferenceShape
    from loom import driver

    logical = numpy.array( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], dtype = float )   # logical [2,3]
    raw = numpy.zeros( ( 4, 3 ) ); raw[ :2 ] = logical                     # batch(2)->flat padded to 4
    L = PhysicalLayout.of( [ 2, 3 ], [ True, False ], alignment_bytes = 32, itemsize = 8 )
    assert L.buffer_shape == [ 4, 3 ] and L.strides == [ 3, 1 ] and L.caps == [ 2, 3 ]

    b = Axis( ShapeVar( 2 ), name = "b" )
    c = Axis( ShapeVar( 3 ), name = "c" )
    t = Tensor[ b, c ]()
    t._raw   = driver.array( raw )
    t._shape = ReferenceShape.from_dense_shape( [ 2, 3 ] )
    t._layout = L

    assert list( t.shape ) == [ 2, 3 ]
    assert t.capacity == ( 2, 3 )                     # per LOGICAL dim: the padding lives in the flat phys dim
    assert numpy.asarray( t.tensor ).tolist() == [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]   # padding + flatten peeled off

    # an elementwise op reads `.tensor`, so it works transparently on the laid-out tensor
    assert numpy.asarray( t + t ).tolist() == [ [ 2, 4, 6 ], [ 8, 10, 12 ] ]


if test( "tensor_protocol" ):
    s = Tensor( 17 )                               # rank 0
    assert int( s ) == 17
    assert float( s ) == 17.0

    v = Tensor[ dict( dtype = int ) ]( [ 5, 6, 7 ] )
    assert len( v ) == 3
    assert [ int( x ) for x in v ] == [ 5, 6, 7 ]  # iteration yields sub-tensors
    assert numpy.asarray( v ).tolist() == [ 5, 6, 7 ]
    assert bool( ( v == v ).all() )                # elementwise eq, then reduce to a scalar bool


if test( "tensor_repr" ):
    t = _named_2x3()
    info( t )
    r = repr( t )
    assert "shape=[2, 3]" in r
    assert "row" in r and "col" in r              # named axes shown in the header


if test( "tensor_symbolic_zero" ):
    # a symbolic zero is the framework's shaped-but-bufferless zero, stored straight in `_raw` --
    # the single source of truth for a tensor's kind (buffer / symbolic zero / None).
    z = driver.symbolic_zero( [ 2, 3 ] )
    assert driver.is_symbolic_zero( z )
    assert not driver.is_symbolic_zero( driver.array( [ 1.0, 2.0 ] ) )

    t = Tensor()
    t.set_raw( z )
    assert t.is_symbolic_zero
    assert t.raw is None                          # nothing to bind -> a ZeroTensor, unbound
    assert t.tensor is None
    assert t.shape == [ 2, 3 ]                    # shape still readable, from the zero object
    assert "symbolic_zero" in repr( t )

    # writing a real buffer makes it a plain bound tensor -- no flag to reset, `_raw` says it all
    t.set_raw( driver.array( [ [ 1.0, 2, 3 ], [ 4, 5, 6 ] ] ) )
    assert not t.is_symbolic_zero
    assert t.raw is not None


# ---- `set` in its various forms (list / backend array / Tensor, dense / ragged, reassignment) ----

if test( "set_list_reassign" ):
    ni = ShapeVar()
    t  = Tensor[ Axis( ni ), dict( dtype = int ) ]()   # no value yet
    assert t.shape == [ None ]                          # one axis, extent still unresolved

    t.set( [ 1, 2, 3 ] )                               # observe from a python list
    assert numpy.asarray( t ).tolist() == [ 1, 2, 3 ]
    assert t.shape == [ 3 ] and ni.value == 3

    t.set( [ 4, 5 ] )                                  # reassign: the observation follows
    assert numpy.asarray( t ).tolist() == [ 4, 5 ]
    assert t.shape == [ 2 ] and ni.value == 2


if test( "set_backend_array" ):
    ni, nj = ShapeVar(), ShapeVar()
    t = Tensor[ Axis( ni ), Axis( nj ), dict( dtype = int ) ]()
    t.set( driver.array( [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ] ) )   # a backend array, not a list
    assert ni.value == 3 and nj.value == 2
    assert t.shape == [ 3, 2 ]
    assert numpy.asarray( t ).tolist() == [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]


if test( "set_ragged_reassign" ):
    class Mesh( Aggregate ):
        cell_vertices   : Tensor[ "cell", "vtx" ]
        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]
        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]

    m = Mesh()
    m.cell_vertices = [ [ 10, 11 ], [ 12 ] ]
    assert m.nb_cells.value == 2 and list( m.nb_vtx_per_cell.value ) == [ 2, 1 ]

    # reassign to a different ragged structure: the per-segment counts (which live on the ShapeVar,
    # not on the tensor) are re-observed from scratch.
    m.cell_vertices = [ [ 1 ], [ 2, 3, 4 ], [ 5, 6 ] ]
    assert m.nb_cells.value == 3 and list( m.nb_vtx_per_cell.value ) == [ 1, 3, 2 ]
    assert numpy.asarray( m.cell_vertices.raw ).shape == ( 3, 3 )   # padded to the new max


if test( "set_from_tensor" ):
    # a dense source, and a DESTINATION with its OWN (independent) axes: setting from the tensor
    # adopts its buffer AND re-observes the destination's axes, so its shape resolves.
    src = Tensor[ Axis( ShapeVar() ), Axis( ShapeVar() ), dict( dtype = int ) ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )
    mi, mj = ShapeVar(), ShapeVar()
    dst = Tensor[ Axis( mi ), Axis( mj ), dict( dtype = int ) ]()
    dst.set( src )
    assert numpy.asarray( dst ).tolist() == [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
    assert dst.shape == [ 2, 3 ] and mi.value == 2 and mj.value == 3

    # setting from a symbolic-zero tensor carries the KIND across (no buffer to re-observe)
    zero = Tensor.like( src )
    zero.set_raw( driver.symbolic_zero( [ 2, 3 ] ) )
    holder = Tensor.like( src )
    holder.set( zero )
    assert holder.is_symbolic_zero and holder.raw is None


# ---- `ShapeVar.value` is a `Tensor` (int-able at rank 0, reduces, iterates); `.raw` is the array --

if test( "shapevar_value_is_tensor" ):
    n = ShapeVar()
    t = Tensor[ Axis( n ), dict( dtype = int ) ]( [ 5, 6, 7 ] )

    v = n.value
    assert isinstance( v, Tensor )
    assert v.shape == []                 # a plain count -> a rank-0 Tensor
    assert int( v ) == 3                 # ... convertible to int
    assert n.value == 3                  # ... comparable through the array protocol
    assert n.max == 3                    # ShapeVar.max still works (reads `raw`)
    assert n.raw is not None             # the backend-array escape hatch stays available

    assert ShapeVar().value is None      # unresolved -> no Tensor yet


if test( "shapevar_ragged_value_is_tensor" ):
    class Mesh( Aggregate ):
        cell_vertices   : Tensor[ "cell", "vtx" ]
        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]
        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]

    m = Mesh()
    m.cell_vertices = [ [ 10, 11 ], [ 12 ] ]

    per_cell = m.nb_vtx_per_cell.value   # a ragged count -> a rank-1 Tensor, dim named after `cell`
    assert isinstance( per_cell, Tensor )
    assert per_cell._dim_names() == [ "cell" ]
    assert numpy.asarray( per_cell ).tolist() == [ 2, 1 ]
    assert int( per_cell.max() ) == 2
    assert [ int( x ) for x in per_cell ] == [ 2, 1 ]   # iterates into per-segment counts

    # `.raw` is the escape hatch to the backend array (parallels `Tensor.raw`)
    assert numpy.asarray( m.nb_vtx_per_cell.raw ).tolist() == [ 2, 1 ]
    assert m.nb_vtx_per_cell.value.raw is not None


if test( "tensor_reduce_ragged" ):
    class Mesh( Aggregate ):
        cell_vertices   : Tensor[ "cell", "vtx" ]
        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]
        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]

    m = Mesh()
    m.cell_vertices = [ [ 1.0, 2.0 ], [ 3.0 ] ]   # cell 1 is padded -> a hole at [ 1, 1 ]
    t = m.cell_vertices

    # a reduction over the ragged axis must IGNORE the padding hole (not fold the 0 in)
    assert numpy.asarray( t.min( "vtx" ) ).tolist()  == [ 1.0, 3.0 ]   # masked; unmasked -> [1, 0]
    assert numpy.asarray( t.prod( "vtx" ) ).tolist() == [ 2.0, 3.0 ]   # masked; unmasked -> [2, 0]
    assert numpy.asarray( t.mean( "vtx" ) ).tolist() == [ 1.5, 3.0 ]   # masked; unmasked -> [1.5, 1.5]
    assert numpy.asarray( t.sum( "vtx" ) ).tolist()  == [ 3.0, 3.0 ]
    assert int( t.min() ) == 1                                          # global min over real cells
