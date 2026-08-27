from loom import ShapeVar, ShapeArray, Axis, AxisList, Tensor, Aggregate, driver, RealTensor, IntTensor, BoolTensor
from loom.testing import test
import numpy

if test( "basic" ):
    class Cell( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]
        vertex_indices   : IntTensor[ "num_vertex", "dim" ]

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
        cell_vertices   : RealTensor[ "cell", "vtx" ]

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
        values  : RealTensor[ "img_pos..." ]
        knots   : RealTensor[ "dim", "num_knot" ]

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

#         frame   = RealTensor( num, dim )

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
#         frame   = RealTensor( num, dim )

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
    inp = RealTensor( 17 ) # rank 0
    out = RealTensor()

    yo = IntTensor( [ 17, 18 ] ) # rank 1
    assert yo.shape == [ 2 ]
    assert out.shape == []

    nx = ShapeVar()
    ny = ShapeVar()
    x  = Axis( nx )   # outside an aggregate, an Axis takes the ShapeVar itself (no name to resolve)
    y  = Axis( ny )
    ya = IntTensor[ x, y ]( [ [ 17, 18 ] ] ) # rank 2 with named Axes

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
    return IntTensor[ x, y ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )


if test( "tensor_ops" ):
    a = IntTensor( [ 1, 2, 3 ] )
    b = IntTensor( [ 10, 20, 30 ] )

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
    m = IntTensor[ row, col ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )   # row(2) x col(3)
    v = IntTensor[ col ]( [ 10, 20, 30 ] )                       # shares the `col` OBJECT

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
    w = IntTensor[ col2 ]( [ 1, 2, 3 ] )
    ow = v * w
    assert numpy.asarray( ow ).shape == ( 3, 3 )
    assert ow._dim_names() == [ "col", "col" ]

    # a BATCH axis sorts FIRST in the result layout, whichever operand carries it (and however the
    # non-batch axes are arranged). No manual reshape: a per-batch value spreads over the rest.
    b = new_batch_axis( 2 )
    bt = IntTensor[ b ]( [ 100, 200 ] )                          # batch(2)
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
    left  = IntTensor[ a, xy ]( [ [ 1, 0 ], [ 0, 2 ] ] )          # a(2) x xy(2)
    right = IntTensor[ b, xy ]( [ [ 1, 1 ], [ 2, 0 ], [ 0, 3 ] ] )  # b(3) x xy(2)

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
    from loom.tensor import ReferenceShape, Storage
    from loom import driver

    logical = numpy.array( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], dtype = float )   # logical [2,3]
    raw = numpy.zeros( ( 4, 3 ) ); raw[ :2 ] = logical                     # batch(2)->flat padded to 4
    L = PhysicalLayout.of( [ 2, 3 ], [ True, False ], alignment_bytes = 32, itemsize = 8 )
    assert L.buffer_shape == [ 4, 3 ] and L.strides == [ 3, 1 ] and L.caps == [ 2, 3 ]

    b = Axis( ShapeVar( 2 ), name = "b" )
    c = Axis( ShapeVar( 3 ), name = "c" )
    t = RealTensor[ b, c ]()
    # ONE statement says how this value is backed: the buffer, the logical sizes it was read from,
    # and the physical layout relating the two -- rather than three fields that must agree.
    t.storage = Storage.of( driver.array( raw ), ReferenceShape.from_dense_shape( [ 2, 3 ] ), L )

    assert list( t.shape ) == [ 2, 3 ]
    assert t.capacity == ( 2, 3 )                     # per LOGICAL dim: the padding lives in the flat phys dim
    assert numpy.asarray( t.tensor ).tolist() == [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]   # padding + flatten peeled off

    # an elementwise op reads `.tensor`, so it works transparently on the laid-out tensor
    assert numpy.asarray( t + t ).tolist() == [ [ 2, 4, 6 ], [ 8, 10, 12 ] ]


if test( "tensor_protocol" ):
    s = RealTensor( 17 )                               # rank 0
    assert int( s ) == 17
    assert float( s ) == 17.0

    v = IntTensor( [ 5, 6, 7 ] )
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

    t = RealTensor()
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
    t  = IntTensor[ Axis( ni ) ]()   # no value yet
    assert t.shape == [ None ]                          # one axis, extent still unresolved

    t.set( [ 1, 2, 3 ] )                               # observe from a python list
    assert numpy.asarray( t ).tolist() == [ 1, 2, 3 ]
    assert t.shape == [ 3 ] and ni.value == 3

    t.set( [ 4, 5 ] )                                  # reassign: the observation follows
    assert numpy.asarray( t ).tolist() == [ 4, 5 ]
    assert t.shape == [ 2 ] and ni.value == 2


if test( "set_backend_array" ):
    ni, nj = ShapeVar(), ShapeVar()
    t = IntTensor[ Axis( ni ), Axis( nj ) ]()
    # a backend array, not a list -- and an INT one: `driver.array` defaults to the driver's
    # ftype, and binding a real buffer to a tensor declared `int` is refused (it would be
    # reinterpreted, not converted, once the FFI spells its element type in C++).
    t.set( driver.array( [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ], dtype = int ) )
    assert ni.value == 3 and nj.value == 2
    assert t.shape == [ 3, 2 ]
    assert numpy.asarray( t ).tolist() == [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]


if test( "set_ragged_reassign" ):
    class Mesh( Aggregate ):
        cell_vertices   : RealTensor[ "cell", "vtx" ]
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
    src = IntTensor[ Axis( ShapeVar() ), Axis( ShapeVar() ) ]( [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] )
    mi, mj = ShapeVar(), ShapeVar()
    dst = IntTensor[ Axis( mi ), Axis( mj ) ]()
    dst.set( src )
    assert numpy.asarray( dst ).tolist() == [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
    assert dst.shape == [ 2, 3 ] and mi.value == 2 and mj.value == 3

    # setting from a symbolic-zero tensor carries the KIND across (no buffer to re-observe)
    zero = Tensor.like( src )
    zero.set_raw( driver.symbolic_zero( [ 2, 3 ] ) )
    holder = Tensor.like( src )
    holder.set( zero )
    assert holder.is_symbolic_zero and holder.raw is None


# ---- `ShapeVar.value` is a HOST count (`ShapeArray`); `as_tensor()` is the device one ----------

if test( "shapevar_value_is_a_host_count" ):
    n = ShapeVar()
    t = IntTensor[ Axis( n ) ]( [ 5, 6, 7 ] )

    v = n.value
    assert isinstance( v, ShapeArray )
    assert v.shape == ()                 # a plain count -> rank 0
    assert int( v ) == 3                 # ... convertible to int
    assert n.value == 3                  # ... and comparable
    assert [ 0 ] * n.value == [ 0, 0, 0 ] # ... and usable as a SIZE, which is the whole point
    assert n.max == 3                    # ShapeVar.max still works (reads `raw`)
    assert n.raw is not None             # the backend-array escape hatch stays available

    assert ShapeVar().value is None      # unresolved -> no count yet

    # the device form is asked for explicitly, and it is an `IntTensor` like any other
    assert isinstance( n.as_tensor(), IntTensor )
    assert int( n.as_tensor() ) == 3
    assert ShapeVar().as_tensor() is None


if test( "shapevar_ragged_value_is_tensor" ):
    class Mesh( Aggregate ):
        cell_vertices   : RealTensor[ "cell", "vtx" ]
        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]
        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]

    m = Mesh()
    m.cell_vertices = [ [ 10, 11 ], [ 12 ] ]

    per_cell = m.nb_vtx_per_cell.value   # a ragged count -> rank 1, dim named after `cell`
    assert isinstance( per_cell, ShapeArray )
    assert per_cell.names == [ "cell" ]
    assert per_cell.tolist() == [ 2, 1 ]
    assert int( per_cell.max() ) == 2
    assert [ int( x ) for x in per_cell ] == [ 2, 1 ]   # iterates into per-segment counts

    # a count stays a count under count arithmetic, and stops being one under a ratio
    assert ( per_cell + 1 ).tolist() == [ 3, 2 ]
    assert isinstance( per_cell + 1, ShapeArray )
    assert not isinstance( per_cell / 2, ShapeArray )

    # `.raw` is the escape hatch to the backend array; `as_tensor()` its tidy device form
    assert numpy.asarray( m.nb_vtx_per_cell.raw ).tolist() == [ 2, 1 ]
    assert m.nb_vtx_per_cell.as_tensor()._dim_names() == [ "cell" ]


if test( "tensor_reduce_ragged" ):
    class Mesh( Aggregate ):
        cell_vertices   : RealTensor[ "cell", "vtx" ]
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


if test( "dtype_is_a_contract_not_a_label" ):
    # A `Tensor`'s dtype is not decoration: `CallArg_Tensor` spells it as the C++ element type of
    # the buffer it binds, so a buffer of another type is REINTERPRETED by the kernel rather than
    # converted. Every route by which a buffer becomes a tensor's therefore enforces it.
    ni = ShapeVar()
    idx = IntTensor[ Axis( ni ) ]()

    # widening is silent (that is a conversion, and it loses nothing)
    idx.set( [ True, False, True ] )
    assert numpy.asarray( idx ).tolist() == [ 1, 0, 1 ]

    # losing the fractional part is NOT: it is refused, from a list...
    try:
        idx.set( [ 1.5, 2.5 ] )
        assert False, "binding a real value to an int tensor must be refused"
    except TypeError:
        pass

    # ...and from another Tensor (the route that used to bind a float64 buffer under an `int`
    # declaration, leaving the kernel to read those bytes as int64).
    try:
        idx.set( RealTensor( [ 1.5, 2.5 ] ) )
        assert False, "binding a real buffer to an int tensor must be refused"
    except TypeError:
        pass

    # a claim made on `wrap` is CHECKED against the buffer, never believed
    try:
        Tensor.wrap( RealTensor( [ 1.0, 2.0 ] ).raw, dtype = int )
        assert False, "wrap must not label a real buffer as int"
    except TypeError:
        pass


if test( "op_result_dtype_follows_the_op" ):
    # A derived tensor DECLARES nothing -- it is a result -- so its dtype is the one its buffer
    # actually came out with. Inheriting the operand's is what used to label a bool buffer `TF`.
    t = RealTensor( [ 1.0, 2.0, 3.0 ] )
    assert t.dtype.floating_point

    mask = t > 1.5
    assert mask.dtype.boolean                                   # a comparison yields booleans
    assert numpy.asarray( mask ).tolist() == [ False, True, True ]

    idx = IntTensor( [ 1, 2, 3 ] )
    assert ( idx * 2 ).dtype.integer                            # int stays int
    assert ( idx / 2 ).dtype.floating_point                     # a true division does not

    # an integer tensor is compared against the value it is GIVEN, not against that value cast
    # down to its own type (`> 1.5` used to be answered as `> 1`).
    assert numpy.asarray( idx > 1.5 ).tolist() == [ False, True, True ]


if test( "element_type_is_the_class" ):
    # What a tensor is MADE OF is its class, not a kwarg: `RealTensor` / `IntTensor` /
    # `BoolTensor`, one per dtype kind. The class is what answers the questions that depend on
    # the element type -- differentiability first among them.
    assert isinstance( RealTensor( [ 1.0 ] ), Tensor )   # `Tensor` stays the abstract base
    assert RealTensor( [ 1.0 ] ).is_differentiable
    assert not IntTensor( [ 1 ] ).is_differentiable

    # a size is a driver POLICY, not part of the declaration -- unless pinned
    assert RealTensor( [ 1.0 ] ).dtype.name == "TF"
    assert RealTensor[ dict( size = 32 ) ]( [ 1.0 ] ).dtype.name == "FP32"

    # a dtype that contradicts the class is two declarations, not a narrower one
    try:
        IntTensor[ dict( dtype = float ) ]()
        assert False, "an IntTensor cannot be declared real"
    except TypeError:
        pass

    # the old spelling still lands on the right class (so declarations can migrate gradually)
    assert type( Tensor( [ 1.0 ] ) ) is RealTensor
    assert type( Tensor[ dict( dtype = int ) ]( [ 1 ] ) ) is IntTensor

    # a tensor built AROUND a buffer takes the class its buffer calls for -- nothing is inherited
    # from the operand, which is how `>` comes back a `BoolTensor` without anyone saying so.
    assert type( RealTensor( [ 1.0, 2.0 ] ) > 1.5 ) is BoolTensor
    assert type( IntTensor( [ 1, 2 ] ) / 2 ) is RealTensor
    assert type( Tensor.wrap( IntTensor( [ 1, 2 ] ).raw ) ) is IntTensor


if test( "a_count_that_lives_on_the_device_is_refused_on_the_host" ):
    # The case the host/device split exists for: a kernel WRITES a count, so under a `jit` that
    # count is a tracer. Nothing can un-trace it, so `value` refuses it right there -- instead of
    # handing back something that fails later in whatever tried to size a buffer with it.
    n = ShapeVar()

    def probe( x ):
        n.set_count( x )                    # what a kernel does with a count it produced
        try:
            n.value
            refused = 0.0
        except TypeError:
            refused = 1.0
        # the DEVICE form stays available -- it is the right answer here, just not a host one
        assert n.as_tensor() is not None
        return driver.array( refused )

    assert float( driver.jit( probe )( driver.array( 3, dtype = int ) ) ) == 1.0

    # outside any trace the very same count reads back on the host, no ceremony
    n.set_count( driver.array( 3, dtype = int ) )
    assert int( n.value ) == 3


if test( "storage_is_a_kind_not_a_flag" ):
    # HOW a value is backed is an object, one variant per way it can be (see `tensor/storage.py`),
    # and each answers the physical questions its own way. Nothing tests for a kind.
    from loom.tensor import Storage, Unbound, Buffer, SymbolicZero, Fill
    from loom.tensor import ReferenceShape

    n = ShapeVar()
    t = RealTensor[ Axis( n ) ]()
    assert isinstance( t.storage, Unbound )          # declared, holds nothing -> a NoneTensor
    assert not t.is_defined and t.raw is None and t.tensor is None
    assert t.allocated_sizes is None

    t.set( [ 1.0, 2.0, 3.0 ] )
    assert isinstance( t.storage, Buffer )
    assert t.is_defined and t.capacity == ( 3, )
    assert numpy.asarray( t.tensor ).tolist() == [ 1.0, 2.0, 3.0 ]

    # a symbolic zero HOLDS a value (it reads as 0) yet backs no buffer -- which is exactly why it
    # binds nothing across the FFI. One variant, no flag, no special case at the call sites.
    z = RealTensor[ Axis( ShapeVar( 3 ) ) ]()
    z.set_raw( driver.symbolic_zero( [ 3 ] ) )
    assert isinstance( z.storage, SymbolicZero )
    assert z.is_defined and z.is_symbolic_zero
    assert z.raw is None                              # nothing to bind...
    assert z.storage.raw is not None                  # ...though the framework's object is held

    # a fill is STATED, never inferred: one scalar looks like any other rank-0 buffer
    f = RealTensor[ Axis( ShapeVar( 4 ) ) ].filled_with( 2.5, ReferenceShape.from_dense_shape( [ 4 ] ) )
    assert isinstance( f.storage, Fill ) and f.is_fill
    assert f.capacity == ( 4, )                       # its logical extents ARE its capacity
    assert f.allocated_sizes is None                  # ... and back no capacity a ShapeVar inverts
    assert numpy.asarray( f.tensor ).tolist() == [ 2.5 ] * 4

    # binding a value to another tensor carries its KIND along, and retypes what backs it
    holder = RealTensor[ Axis( ShapeVar( 4 ) ) ]()
    holder.set( f )
    assert isinstance( holder.storage, Fill )         # a fill stays a fill
    holder.set( z )
    assert isinstance( holder.storage, SymbolicZero ) # a symbolic zero stays one

    # and `Storage.of` is the ONE place a kind is decided from a value
    assert isinstance( Storage.of( None ), Unbound )
    assert isinstance( Storage.of( driver.array( 1.0 ) ), Buffer )


if test( "a_literal_extent_is_enough_standalone" ):
    # Outside an aggregate there is no scope to resolve a name through, so an axis has to be
    # handed over as an object. It should not also cost a `ShapeVar` to spell: an integer IS an
    # extent, and it mints its own count behind the scenes -- the short form is a shortcut, not a
    # second kind of axis.
    t = RealTensor[ 2, 3 ]( [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ] )
    assert t.shape == [ 2, 3 ] and t.capacity == ( 2, 3 )

    i = Axis( 3, name = "i" )
    x = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )
    y = RealTensor[ i ]( [ 4.0, 5.0, 6.0 ] )

    # the axis OBJECT is shared, so the two line up by identity and contract by name
    assert float( x.dot( y, "i" ) ) == 32.0
    assert ( x + y )._dim_names() == [ "i" ]

    # an axis the other side does not have is broadcast -> an outer product, both names kept
    j = Axis( 2, name = "j" )
    assert ( x * RealTensor[ j ]( [ 1.0, 10.0 ] ) ).shape == [ 3, 2 ]

    # the long spelling still means exactly the same thing
    assert RealTensor[ Axis( ShapeVar( 4 ) ) ]().shape == [ 4 ]


if test( "an_axis_can_be_given_just_a_name" ):
    # Standalone there is no scope a name could be RESOLVED in -- so a bare name is not a reference
    # to a count, it IS the axis's name, and the count is minted for it. That is the whole ceremony
    # gone: what one wants standalone is an axis to match ops by, not a count to declare.
    i = Axis( "i" )
    x = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )
    y = RealTensor[ i ]( [ 4.0, 5.0, 6.0 ] )

    assert i.name == "i"
    assert x.shape == [ 3 ]                          # the extent is solved from the value
    assert float( x.dot( y, "i" ) ) == 32.0

    # the minted count is named after the axis, mechanically: `nb_<thing>`, a leading `num_`
    # dropped. No pluralization -- an irregular plural would make the name unguessable.
    ( count, ) = i.coeffs
    assert count.name == "nb_i" and int( count.value ) == 3
    ( cell_count, ) = Axis( "num_cell" ).coeffs
    assert cell_count.name == "nb_cell"

    # two axes are still distinct OBJECTS, so they outer-product rather than collapse
    j = Axis( "j" )
    z = RealTensor[ j ]( [ 1.0, 10.0 ] )
    assert ( x * z ).shape == [ 3, 2 ]

    # inside a scope a string keeps its usual meaning (a count declared as a field), and an
    # EXPRESSION standalone is still refused -- it genuinely needs a scope
    try:
        Axis( "2 * n + 1" )
        assert False, "an affine expression cannot be resolved without a scope"
    except TypeError:
        pass


if test( "where_selects_by_axis_identity" ):
    import loom

    i, j = Axis( "i" ), Axis( "j" )
    x = RealTensor[ i ]( [ -1.0, 2.0, -3.0 ] )
    y = RealTensor[ i ]( [ 10.0, 20.0, 30.0 ] )

    mask = x > 0
    assert type( mask ) is BoolTensor

    # either branch may be a plain scalar, or another tensor
    assert numpy.asarray( mask.where( x, 0.0 ) ).tolist() == [ 0.0, 2.0, 0.0 ]
    assert numpy.asarray( mask.where( x, y ) ).tolist() == [ 10.0, 2.0, 30.0 ]

    # and the three operands align BY AXIS, like any elementwise op: a per-row condition selects
    # across a whole matrix with no reshaping.
    m = RealTensor[ i, j ]( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ] )
    r = mask.where( m, 0.0 )
    assert r.shape == [ 3, 2 ] and r._dim_names() == [ "i", "j" ]
    assert numpy.asarray( r ).tolist() == [ [ 0.0, 0.0 ], [ 3.0, 4.0 ], [ 0.0, 0.0 ] ]

    assert numpy.asarray( loom.where( mask, x, y ) ).tolist() == [ 10.0, 2.0, 30.0 ]


if test( "every_operation_has_both_forms" ):
    # a method and a free function are the same operation, kept in step on purpose: some
    # expressions read better chained, others called.
    import loom

    i, j = Axis( "i" ), Axis( "j" )
    m = RealTensor[ i, j ]( [ [ 1.0, 4.0 ], [ 9.0, 16.0 ] ] )
    v = RealTensor[ i ]( [ -1.0, 2.0 ] )
    w = RealTensor[ i ]( [ 10.0, 20.0 ] )

    same = lambda a, b: numpy.asarray( a ).tolist() == numpy.asarray( b ).tolist()

    assert same( loom.dot( v, w, "i" ), v.dot( w, "i" ) )
    assert same( loom.sum ( m, "i" ), m.sum ( "i" ) )
    assert same( loom.prod( m, "i" ), m.prod( "i" ) )
    assert same( loom.min ( m, "i" ), m.min ( "i" ) )
    assert same( loom.max ( m, "i" ), m.max ( "i" ) )
    assert same( loom.mean( m, "i" ), m.mean( "i" ) )
    assert same( loom.all ( m > 2, "i" ), ( m > 2 ).all( "i" ) )
    assert same( loom.any ( m > 2, "i" ), ( m > 2 ).any( "i" ) )
    assert same( loom.sqrt( m ), m.sqrt() )
    assert same( loom.abs ( v ), abs( v ) )
    assert same( loom.clip( v, -0.5, 1.0 ), v.clip( -0.5, 1.0 ) )
    assert same( loom.stop_gradient( v ), v.stop_gradient() )
    assert same( loom.transpose( m ), m.transpose() )
    assert same( loom.arcsin( RealTensor[ i ]( [ 0.0, 0.5 ] ) ),
                 RealTensor[ i ]( [ 0.0, 0.5 ] ).arcsin() )


if test( "a_result_keeps_its_extents_after_its_operand_is_gone" ):
    import gc

    i, j = Axis( "i" ), Axis( "j" )
    x = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )

    # the right operand is a TEMPORARY: an axis reads its extent off the tensors that use it, and
    # those are held weakly, so without the result registering itself the extent would vanish with
    # the temporary.
    r = x * RealTensor[ j ]( [ 1.0, 10.0 ] )
    gc.collect()
    assert r.shape == [ 3, 2 ] and r._dim_names() == [ "i", "j" ]

    # registering is only sound because an op preserves its axes' extents. A partial slice does
    # NOT, so it hands over a DERIVED axis: the same axis MEANT, over a count of its own.
    s = x[ 0:2 ]
    assert numpy.asarray( s ).tolist() == [ 1.0, 2.0 ]
    assert s.shape == [ 2 ]                    # the SLICED size, not the axis's
    assert s._dim_names() == [ "i" ]           # ... still readable as `i`
    assert int( i.max ) == 3                   # ... and the shared axis is untouched
    assert RealTensor[ i ]( [ 7.0, 8.0, 9.0 ] ).shape == [ 3 ]

    # a FULL slice changes nothing, so the axis object survives it
    assert x[ : ]._dim_axes()[ 0 ] is i

    # the usage list does not grow without bound: dead entries are compacted away
    ( count, ) = i.coeffs
    for _ in range( 500 ):
        x + x
    gc.collect()
    x + x
    assert len( count.usages ) < 32, len( count.usages )


if test( "a_window_into_a_dimension_remembers_where_it_starts" ):
    # A slice is not a fresh dimension, and it is not the original one either: `v[ 10:20 ]` is
    # `num_vertex + 10` -- the SAME dimension, read from another origin. Keeping the offset is what
    # separates the two questions an axis answers:
    #   * which dimension is this?      -> selection stays by name, wherever the window starts
    #   * do our positions correspond?  -> only the same window of the same dimension maps
    num_vertex = Axis( "num_vertex" )
    v = RealTensor[ num_vertex ]( [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ] )
    w = RealTensor[ num_vertex ]( [ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0 ] )

    # the same window of the same dimension: elementwise
    assert numpy.asarray( v[ 2:4 ] * w[ 2:4 ] ).tolist() == [ 40.0, 90.0 ]
    assert ( v[ 2:4 ] * w[ 2:4 ] ).shape == [ 2 ]

    # DIFFERENT windows of the same dimension are REFUSED. They index the same thing, so they are
    # not independent axes to broadcast -- and they start at different places, so they do not line
    # up either. Silently pairing item 2 with item 0 is the one outcome that must not happen.
    for bad in ( lambda: v[ 2:4 ] * w[ 0:2 ], lambda: v[ 2:4 ] * w, lambda: v[ 2:4 ] + w[ 1:3 ] ):
        try:
            bad()
            assert False, "two different windows of one dimension must not combine"
        except ValueError:
            pass

    # genuinely distinct axes still broadcast, exactly as before
    assert ( v * RealTensor[ Axis( "j" ) ]( [ 1.0, 2.0 ] ) ).shape == [ 6, 2 ]

    # the offset composes through a second slice, into the ORIGINAL dimension
    assert "num_vertex+3" in repr( v[ 2:6 ][ 1:3 ] )
    assert "num_vertex*2" in repr( v[ ::2 ] )

    # ... and a window is still the `num_vertex` dimension for selection, by name or by object
    assert float( v[ 2:4 ].sum( "num_vertex" ) ) == 5.0
    assert float( v[ 2:4 ].sum(  num_vertex  ) ) == 5.0

    # an index ARRAY is not an affine window -- its positions bear no fixed relation to the
    # original's, so it becomes a dimension of its own rather than a mislabelled window
    assert v[ numpy.array( [ 4, 1 ] ) ]._dim_names() == [ None ]

    assert int( num_vertex.max ) == 6            # the dimension itself is never touched


if test( "slicing_keeps_the_meaning_and_drops_the_size" ):
    # An axis has two lifetimes, and slicing is where they part: `x[ 0:2 ]` is still the `i`
    # dimension (so two independent slices must still multiply ELEMENTWISE), but it is no longer
    # 3 long -- and the shared `i` must not be taught otherwise. Hence a DERIVED axis: same
    # identity, own count.
    i, j = Axis( "i" ), Axis( "j" )
    a = RealTensor[ i ]( [ 1.0, 2.0, 3.0 ] )
    b = RealTensor[ i ]( [ 10.0, 20.0, 30.0 ] )

    sa, sb = a[ 0:2 ], b[ 0:2 ]
    assert sa.shape == [ 2 ] and sb.shape == [ 2 ]
    assert sa._dim_axes()[ 0 ] is not sb._dim_axes()[ 0 ]            # different objects...
    assert sa._dim_axes()[ 0 ].identity is i.identity                # ... in the same dimension
    assert sa._dim_axes()[ 0 ].coordinate == sb._dim_axes()[ 0 ].coordinate   # ... same window

    # so they map by reference, elementwise -- NOT as an outer product
    assert numpy.asarray( sa * sb ).tolist() == [ 10.0, 40.0 ]
    assert ( sa * sb ).shape == [ 2 ]

    # and a sliced matrix still lines its rows up with a sliced vector
    m = RealTensor[ i, j ]( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ] )
    r = m[ 0:2 ] * sa
    assert r.shape == [ 2, 2 ] and r._dim_names() == [ "i", "j" ]

    # a narrowed dimension is still selectable by name AND by the original axis object
    assert numpy.asarray( m[ 0:2 ].sum( "i" ) ).tolist() == [ 4.0, 6.0 ]
    assert numpy.asarray( m[ 0:2 ].sum(  i  ) ).tolist() == [ 4.0, 6.0 ]

    assert int( i.max ) == 3                                        # never poisoned


if test( "a_window_keeps_its_bounds_as_expressions" ):
    # A window is `( lo, hi, step )`, both bounds AFFINE positions in the dimension's own space, and
    # the extent is derived from them rather than stored. That is what lets an open-ended slice say
    # "ends where the dimension ends" instead of freezing the size it happened to have.
    nb_vertex = ShapeVar(); nb_vertex.name = "nb_vertex"
    num_vertex = Axis( nb_vertex ); num_vertex.name = "num_vertex"
    v = RealTensor[ num_vertex ]( [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ] )

    def window_of( t ):
        ax = t._dim_axes()[ 0 ]
        return repr( ax.lo ), repr( ax.hi ), ax.step

    assert window_of( v[ 2:4 ] ) == ( "2", "4", 1 )                    # both bounds literal
    assert window_of( v[ 2:  ] ) == ( "2", "nb_vertex", 1 )            # ends where the dimension does
    assert window_of( v[  :-1] ) == ( "0", "nb_vertex - 1", 1 )        # one short of its end
    assert window_of( v[ ::2 ] ) == ( "0", "nb_vertex", 2 )
    assert v[ : ]._dim_axes()[ 0 ] is num_vertex                       # the whole of it IS the axis

    # a stepped window's size is a ceiling division -- a partial last stride still holds an item
    assert v[ ::2 ].shape == [ 3 ] and v[ ::4 ].shape == [ 2 ]
    # ... and an empty slice holds nothing, not a negative amount
    assert v[ 4:2 ].shape == [ 0 ]


if test( "a_window_can_teach_its_dimension_the_count" ):
    # Because the relation to the dimension is KEPT, inference flows back through a window: four
    # items starting at 2 means the dimension holds six. Nothing had to be told twice.
    nb_vertex = ShapeVar(); nb_vertex.name = "nb_vertex"
    num_vertex = Axis( nb_vertex ); num_vertex.name = "num_vertex"

    assert nb_vertex.value is None
    tail = RealTensor[ num_vertex.windowed( slice( 2, None ) ) ]( [ 9.0, 9.0, 9.0, 9.0 ] )
    assert int( nb_vertex.value ) == 6                    # extent = nb_vertex - 2 = 4

    # a window with LITERAL bounds says nothing about the dimension, and must not pretend to:
    # its extent is the constant 2, which constrains nothing.
    nb_m = ShapeVar(); nb_m.name = "nb_m"
    num_m = Axis( nb_m ); num_m.name = "num_m"
    RealTensor[ num_m.windowed( slice( 2, 4 ) ) ]( [ 1.0, 2.0 ] )
    assert nb_m.value is None

    # neither does a STEPPED one: a sampled view genuinely does not determine what it sampled.
    nb_k = ShapeVar(); nb_k.name = "nb_k"
    num_k = Axis( nb_k ); num_k.name = "num_k"
    stepped = num_k.windowed( slice( None, None, 2 ) )
    assert stepped.extent is None                          # not an affine expression at all
    assert stepped.solve_single( nb_k, 3 ) is None


if test( "a_dimension_is_an_identity_with_no_size" ):
    # `AxisId` is WHICH dimension, and nothing else. Sizeless on purpose: a size belongs to a
    # WINDOW on a dimension, and a dimension has no privileged window -- `num_vertex` and
    # `num_vertex+2` are two views of one thing, neither of them "the" one.
    from loom.tensor import AxisId

    num_vertex = Axis( "num_vertex" )
    assert isinstance( num_vertex.identity, AxisId )
    assert not hasattr( num_vertex.identity, "lo" )          # no bounds, no extent, no count

    # every window into it shares that identity...
    v = RealTensor[ num_vertex ]( [ 0.0, 1.0, 2.0, 3.0 ] )
    assert v[ 1:3 ]._dim_axes()[ 0 ].identity is num_vertex.identity
    assert v[ 1:3 ][ 1: ]._dim_axes()[ 0 ].identity is num_vertex.identity

    # ... and the name lives there too, so a window is not a second thing to name
    assert v[ 1:3 ]._dim_names() == [ "num_vertex" ]
    num_vertex.name = "renamed"
    assert v[ 1:3 ]._dim_names() == [ "renamed" ]

    # identity is by REFERENCE, never by name: two dimensions both called `i` are two dimensions
    assert Axis( "i" ).identity is not Axis( "i" ).identity


if test( "a_dimension_can_be_shared_across_aggregates" ):
    # Two aggregates of the same kind mint their own dimension, so they do NOT line up -- which is
    # right by default: two unrelated meshes both have vertices, and vertex 3 of one is not vertex 3
    # of the other. When they ARE the same dimension, say so by passing the `AxisId`.
    #
    # That is the difference with injecting an `Axis`: an axis is a WINDOW, so sharing one shares
    # the size too. An `AxisId` is only WHICH dimension, so each instance keeps a count of its own.
    from loom.tensor import AxisId

    class Cell( Aggregate ):
        positions   : RealTensor[ "num_vertex" ]
        num_vertex  : Axis[ "nb_vertices" ]
        nb_vertices : ShapeVar

    plain_a, plain_b = Cell(), Cell()
    plain_a.positions = [ 1.0, 2.0, 3.0 ]
    plain_b.positions = [ 10.0, 20.0 ]
    assert plain_a.num_vertex.identity is not plain_b.num_vertex.identity
    assert ( plain_a.positions * plain_b.positions ).shape == [ 3, 2 ]      # distinct -> outer

    vertex = AxisId( "num_vertex" )
    a = Cell( num_vertex = vertex ); a.positions = [ 1.0, 2.0, 3.0 ]
    b = Cell( num_vertex = vertex ); b.positions = [ 10.0, 20.0, 30.0 ]

    assert a.num_vertex.identity is b.num_vertex.identity is vertex
    assert numpy.asarray( a.positions * b.positions ).tolist() == [ 10.0, 40.0, 90.0 ]

    # each keeps its OWN count -- the dimension carries no size to impose
    assert int( a.nb_vertices.value ) == 3
    c = Cell( num_vertex = vertex ); c.positions = [ 1.0, 2.0 ]
    assert int( c.nb_vertices.value ) == 2 and int( a.nb_vertices.value ) == 3

    # same dimension, different extents: they align, and then the shapes disagree -- which is the
    # honest error, not a silent broadcast
    try:
        a.positions * c.positions
        assert False, "3 items and 2 items of one dimension cannot combine"
    except TypeError:
        pass

    # an AxisId is about a dimension, so it is refused on a field that is not an axis
    try:
        Cell( nb_vertices = AxisId( "x" ) )
        assert False, "an AxisId is not a count"
    except TypeError:
        pass
