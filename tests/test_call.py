import loom
from loom import CtShapeVar, ShapeVar, Axis, Tensor, Aggregate, driver, RealTensor, IntTensor
from loom.compilation.FfiCode import FfiCode
from errand import test

# An `@aggregate` instance is built BEFORE the call and passed as a plain kwarg. Inputs and
# outputs are DISJOINT (as in XLA): a kernel never writes what it reads, so there is no
# aliasing, and no reconciling of two sizes under one name. What looks like a mutation is a
# Python-side rebinding, done by the caller between two calls.
#
# Each attribute of a passed object falls in exactly one category:
#
# * named in `output_attributes` -> an OUTPUT: a fresh buffer, allocated at the capacity this
#   call decided, rebound onto the attribute once the call returns.
# * holds a value -> an INPUT, bound at the size its data actually has.
# * empty and not declared -> UNBOUND: nothing crosses the FFI, and the kernel is not handed a
#   degenerate view to test but a `NoneTensor`, a TYPE with no data (an attribute may simply be
#   optional and unused -- see `partial_init`).
#
# Nothing is returned: outputs are written back onto the instances we were given (the
# aggregate is our own Python object -- Jax only ever sees the tensors inside it).
#
# The io category is not just bookkeeping: it is what `make_available` uses to move a member to
# the device and back. It is a property of the KERNEL, though, not of the data -- chain two
# kernels over one object and they may read and write different parts of it. So the generated
# struct holds none: it is handed a POLICY (`Cell_io`, the same shape, one category per member)
# where `run_parallel` expects a category. This call's own policy comes ready-made as
# `<arg>_io`, but a body may hand another one -- or a plain tag, which then holds for every
# member (`InpList(), cell`). A bare tensor has no members, so a plain tag is all it takes.
#
# A `ShapeVar` holds a COUNT: how many items are used. A kernel reads it or writes it, so it
# lives in a device buffer; under `jit` Python does not know it, and it can therefore never
# size anything (an XLA shape cannot depend on a device value).
#
# What sizes a buffer is a CAPACITY -- and a capacity is NOT state on the object: it is a
# decision about ONE allocation, so it is given to the CALL that allocates. An object only
# ever says what it IS; the call says what it allocates. A capacity already materialized in a
# buffer is read back from it, so a chained call need not restate it.

if test( "basic" ):
    # NB the class is `Cell1`, not `Cell`: a bare `Cell` would match the hand-written
    # `src/cpp/sdot/Cell.h`, and the aggregate would build on THAT struct instead of a generated
    # one. A name with no `sdot/<Name>.h` gets its struct generated whole (see `_emit_full_header`).
    class Cell1( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    # the ctor only prescribes what the cell IS: `nb_dims` is compile-time known (its count is
    # its size). `nb_vertices` has no count yet -- the kernel is what writes it.
    cell = Cell1( nb_dims = 2 )

    # the body runs on a DEVICE, once per item of the call's batch (a single item here: a `vmap`
    # is what will give it axes). `FfiCode` wraps it in a named functor and a
    # `run_parallel` over every argument of the call, on the device's queue (a typedef, since the
    # memory space a pointer lives in is part of its type).
    #
    # An object handed to the kernel is an ARGUMENT of `run_parallel`, not a capture: that is
    # what lets `make_available` retype its pointers into the memory space the kernel reads. It
    # is preceded by its io policy (`cell_io`, generated from what this call does with `cell`),
    # and made available member by member -- an input is not copied back, an output not copied in.
    #
    # `batch_index` is applied to a value exactly like any other index: it is a multi-index of
    # NAMED coordinates (`vmap_0 = i`), so it selects an axis by name -- and a value that is not
    # mapped along that axis ignores it. Unbatched, it is the EMPTY multi-index, and indexing by
    # it is a no-op. Hence one body, batched or not.
    loom.ffi_call(
        "test_call_basic",
        FfiCode.per_item( code = """
        outputs.cell.nb_vertices( batch_index ).set( 1 );
        outputs.cell.vertex_positions( batch_index, dim = 0, num_vertex = 0 ) = 1;
        outputs.cell.vertex_positions( batch_index, dim = 1, num_vertex = 0 ) = 2;
        """ ),
        # la capacite en sommets => `vertex_positions` est alloue en 8x2
        cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 8 } ),
    )

    # the kernel wrote the count, and the count is what makes the tensor read 1x2 -- while the
    # BUFFER it was written into is the 8x2 the call asked for.
    assert cell.nb_vertices.value == 1
    assert cell.vertex_positions.shape == [ 1, 2 ]
    assert cell.vertex_positions.capacity == ( 8, 2 )
    assert cell.vertex_positions.raw.tolist()[ 0 ] == [ 1, 2 ]

    # a `Tensor` needs no wrapper aggregate to be an argument: here it borrows `cell`'s axis,
    # hence `cell`'s ShapeVar. That ShapeVar's capacity is not restated: it is READ BACK from
    # `cell.vertex_positions`, which is allocated 8x2 -- so `res` gets 8 too. `cell` is a
    # read-only input of this second call.
    #
    # `res` is a bare tensor: no members, so no policy either -- a plain tag says it all.
    res = RealTensor[ cell.num_vertex ]()

    loom.ffi_call(
        "test_call_basic_res",
        FfiCode.per_item( code = """
        outputs.res( batch_index, num_vertex = 0 ) = inputs.cell.nb_vertices( batch_index );
        """ ),
        cell = cell,
        res = loom.out( res ),
    )

    # the capacity was not restated, and `res` still got 8: it was read back from the buffer of
    # `cell.vertex_positions`, which the first call allocated.
    assert res.capacity == ( 8, )
    assert res.raw.tolist()[ 0 ] == 1


if test( "partial_init" ):
    # not every declared tensor has to be filled: `vertex_indices` is neither given a value nor
    # declared as an output, so nothing is bound for it. It does not become a degenerate
    # `TensorView` to be tested at runtime -- it lowers to a `NoneTensor`, a distinct TYPE with
    # no data, so the kernel discriminates at COMPILE time (and a `static_assert` can forbid
    # touching it outright).
    class Cell2( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]
        vertex_indices   : IntTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    cell = Cell2( nb_dims = 2 )

    loom.ffi_call(
        "test_partial_init",
        FfiCode.per_item( code = """
        outputs.cell.nb_vertices( batch_index ).set( 1 );
        // a fresh output buffer is NOT guaranteed zero-initialized (see
        // `ProjectedSumOfDiracs::zero_position_grad`'s docstring for the general fact) --
        // write every element the assertion below reads, rather than relying on a
        // leftover-memory default.
        outputs.cell.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 1;
        outputs.cell.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 0;
        
        static_assert( outputs.cell.vertex_positions.is_valid );
        static_assert( ! outputs.cell.vertex_indices  .is_valid );
        """ ),
        cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 8 } ),
    )

    assert cell.nb_vertices.value == 1
    assert cell.vertex_positions.raw.tolist()[ 0 ] == [ 1, 0 ]

    # nothing was bound for `vertex_indices`, and nothing came back for it either.
    assert cell.vertex_indices.raw is None


if test( "input_exceptions" ):
    # `vertex_positions` holds data by the second call (the first one wrote it), so it would
    # default to an INPUT there -- `input_exceptions` is the symmetric carve-out of
    # `output_attribute_exceptions`: it forces the attribute back to `UNBOUND` regardless, so the
    # kernel sees a `NoneTensor` exactly as if it had never been filled. Nothing crosses the FFI
    # for it, and the underlying data stays untouched (an asserted "this kernel has no business
    # touching it", not a mutation).
    class Cell3( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    cell = Cell3( nb_dims = 2 )

    loom.ffi_call(
        "test_input_exceptions_init",
        FfiCode.per_item( code = """
        outputs.cell.nb_vertices( batch_index ).set( 1 );
        outputs.cell.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 1;
        outputs.cell.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 2;
        """ ),
        cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 8 } ),
    )

    loom.ffi_call(
        "test_input_exceptions_use",
        FfiCode.per_item( code = """
        static_assert( ! inputs.cell.vertex_positions.is_valid );
        """ ),
        cell = loom.unbound( cell, "vertex_positions" ),
    )

    # the exception only kept the second call from binding it -- the data itself is untouched.
    assert cell.vertex_positions.raw.tolist()[ 0 ] == [ 1, 2 ]


if test( "two_instances" ):
    # the same aggregate, twice in one call, with different compile-time shape vars: `Cell` is
    # generated as a C++ template, instantiated once per argument.
    class Cell3( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    flat = Cell3( nb_dims = 2 )
    volu = Cell3( nb_dims = 3 )

    loom.ffi_call(
        "two_instances",
        FfiCode.per_item( code = """
        // an index applies to a whole aggregate just as well as to one of its members:
        // `f( batch_index ).nb_vertices` and `outputs.flat.nb_vertices( batch_index )` are the
        // same thing. Handy when every member takes the same index.
        auto f = outputs.flat( batch_index );
        f.nb_vertices.set( 1 );
        f.vertex_positions( num_vertex = 0, dim = 0 ) = 1;
        f.vertex_positions( num_vertex = 0, dim = 1 ) = 2;
        
        outputs.volu.nb_vertices( batch_index ).set( 1 );
        // see `partial_init`'s comment above: write every dim, not just the nonzero one.
        outputs.volu.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 0;
        outputs.volu.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 0;
        outputs.volu.vertex_positions( batch_index, num_vertex = 0, dim = 2 ) = 3;
        """ ),
        flat = loom.out( flat, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 8 } ),
        volu = loom.out( volu, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 4 } ),
    )

    # one class, two instantiations: the compile-time `nb_dims` differ, and so do the capacities.
    assert flat.nb_vertices.value == 1 and volu.nb_vertices.value == 1
    assert flat.vertex_positions.capacity == ( 8, 2 )
    assert volu.vertex_positions.capacity == ( 4, 3 )
    assert flat.vertex_positions.raw.tolist()[ 0 ] == [ 1, 2 ]
    assert volu.vertex_positions.raw.tolist()[ 0 ] == [ 0, 0, 3 ]


if test( "nested" ):
    # an aggregate field whose type is itself an aggregate: `Cell` is generated as its own C++
    # template, and `Pair` holds two instantiations of it (and forwards their parameters).
    class Cell4( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    class Pair( Aggregate ):
        left  : Cell4
        right : Cell4



    # a mapping under a field's name scopes a prescription to that field alone.
    pair = Pair( left = { "nb_dims": 2 }, right = { "nb_dims": 3 } )

    loom.ffi_call(
        "test_call_nested",
        FfiCode.per_item( code = """
        // indexing an aggregate indexes its members -- a nested one included, recursively.
        auto p = outputs.pair( batch_index );
        
        // every coordinate of the one written row is set explicitly: a fresh output buffer
        // is NOT guaranteed zero-initialized on every device (XLA's GPU allocator does not
        // zero fresh device memory, unlike a first-touch CPU page).
        p.left.nb_vertices.set( 1 );
        p.left.vertex_positions( num_vertex = 0, dim = 0 ) = 0;
        p.left.vertex_positions( num_vertex = 0, dim = 1 ) = 1;
        
        p.right.nb_vertices.set( 1 );
        p.right.vertex_positions( num_vertex = 0, dim = 0 ) = 0;
        p.right.vertex_positions( num_vertex = 0, dim = 1 ) = 0;
        p.right.vertex_positions( num_vertex = 0, dim = 2 ) = 2;
        """ ),
        # nommer l'agregat couvre tout ce qu'il y a dessous
        pair = loom.out( pair, capacities = { "left.nb_vertices": 8, "right.nb_vertices": 4 } ),
    )

    assert pair.left.nb_vertices.value == 1 and pair.right.nb_vertices.value == 1
    assert pair.left .vertex_positions.raw.tolist()[ 0 ] == [ 0, 1 ]
    assert pair.right.vertex_positions.raw.tolist()[ 0 ] == [ 0, 0, 2 ]

if test( "vmap" ):
    # a `vmap` maps the call over a new axis, and the KERNEL is what runs it: the batched call is
    # one launch of one (re)compiled kernel over N items, not N calls. The body does not change --
    # `batch_index` was already there, empty. Nothing here is Jax-specific: `driver.vmap` is what
    # the driver in use provides (Jax today, Torch later), and the test only ever sees driver
    # arrays.
    class Cell5( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar



    noyau = FfiCode.per_item( code = """
    auto c = outputs.cell( batch_index );
    c.nb_vertices.set( 1 );
    c.vertex_positions( num_vertex = 0, dim = 0 ) = inputs.scale( batch_index, dim = 0 );
    c.vertex_positions( num_vertex = 0, dim = 1 ) = inputs.scale( batch_index, dim = 1 );
    """ )
    def positions_of( raw_scale ):
        cell = Cell5( nb_dims = 2 )

        # a bare tensor input, borrowing `cell`'s `dim` axis: one scale per dimension.
        scale = RealTensor[ cell.dim ]()
        scale.set( raw_scale )

        loom.ffi_call(
            "test_call_vmap",
            noyau,
            cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 4 } ),
            scale = scale,
        )
        return cell.vertex_positions.raw

    # unmapped: one cell, the batch multi-index is empty and indexing by it is a no-op.
    one = positions_of( driver.array( [ 1, 2 ] ) )
    assert one.tolist()[ 0 ] == [ 1, 2 ]

    # mapped: three cells at once. Each item reads ITS row of `scale` -- the input gained a batch
    # axis and the kernel selects it by name -- and writes ITS slice of the output.
    many = driver.vmap( positions_of )( driver.array( [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ] ) )
    assert many.shape == ( 3, 4, 2 )   # 3 batch items x the capacity this call asked for x dim
    assert [ item.tolist()[ 0 ] for item in many ] == [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]


if test( "capacity_overflow" ):
    # a capacity is a GUESS -- only the kernel knows how many items it produces. So the kernel is
    # allowed to ask for more than it was given: `ShapeVarView::operator=` sees the count exceed
    # the capacity it was handed, records it in the call's error buffer (which is not a ShapeVar
    # business: anything that fails records there, see `support/containers/ErrorBuffer.h`), and
    # CLAMPS the count -- so whatever the body writes next stays inside the buffers this call
    # allocated. Python then reserves more and runs again, until it fits. Nothing of a failed run
    # survives: an output is a fresh buffer every time.
    class Cell6( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar   # written by the kernel: what it produced
        nb_wanted        : ShapeVar   # read by it: how many it is going to produce
        nb_dims          : CtShapeVar



    noyau = FfiCode.per_item( code = """
    auto c = outputs.cell( batch_index );
    
    // the count may not fit -- and then what one READS BACK is the capacity, never more,
    // which is what makes the loop below safe whatever happens.
    c.nb_vertices.set( c.nb_wanted );
    for ( SI n = 0; n < SI( c.nb_vertices ); ++n )
        c.vertex_positions( num_vertex = n, dim = 0 ) = n;
    """ )
    def cell_of( nb_wanted, capacity ):
        cell = Cell6( nb_dims = 2, nb_wanted = nb_wanted )
        loom.ffi_call(
            "test_call_overflow",
            noyau,
            cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": capacity } ),
        )
        return cell

    # it fits: one run, and the capacity stays the one that was asked for.
    cell = cell_of( 2, 8 )
    assert cell.nb_vertices.value == 2
    assert cell.vertex_positions.capacity == ( 8, 2 )

    # 3 vertices into a buffer of 2 -> a second run, with `max( 3, 2 * 2 ) = 4`: a capacity that
    # was exceeded once tends to be exceeded again, so we make ROOM rather than fit the count.
    cell = cell_of( 3, 2 )
    assert cell.nb_vertices.value == 3
    assert cell.vertex_positions.capacity == ( 4, 2 )
    # les 3 que le kernel a ECRITS, et rien de plus : la 4e case est allouee mais jamais ecrite,
    # et un tampon de sortie est FRAIS, pas remis a zero (seul un output PARTAGE d'un appel batche
    # est semé, cf. `CallArg_Tensor.cpp_seed_member`). L'affirmer valait 0 marchait tant que
    # l'allocateur rendait des pages deja nulles -- donc au gre de ce qui avait tourne avant :
    # ici on lit 1e-323, ailleurs 6.5e-310.
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ][ :3 ] == [ 0, 1, 2 ]

    # 5 into a buffer of 1 -> `max( 5, 2 * 1 ) = 5`: this time it is the count that decides.
    cell = cell_of( 5, 1 )
    assert cell.nb_vertices.value == 5
    assert cell.vertex_positions.capacity == ( 5, 2 )
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ] == [ 0, 1, 2, 3, 4 ]


if test( "der" ):
    # a differentiable call: `code` computes the output, `backward` the input gradients. The
    # backward is generated as an ORDINARY kernel call -- its inputs are the forward inputs and
    # outputs plus the output cotangents (`grad_for_out`), its output the input cotangent
    # (`grad_for_inp`). Jax reaches it through a `custom_vjp` rule.
    #
    # An INTEGER tensor is non-differentiable and non-perturbable: it never gets a `grad_for_`.
    # A symbolically-zero output cotangent lowers to a `ZeroTensor` (`grad_for_out.surely_null`
    # is a compile-time true), and a non-perturbed input gradient to a `NoneTensor`
    # (`grad_for_inp.is_valid` a compile-time false) -- either lets the body drop a term at
    # compile time rather than move or multiply a buffer of zeros.
    avant = FfiCode.per_item( code = """
            outputs.out = 2 * inputs.inp + 100;
        """ )
    arriere = FfiCode.per_item( """
            if ( ! grad_of_outputs.out.surely_null && grad_of_inputs.inp.is_valid )
                grad_of_inputs.inp = 2 * grad_of_outputs.out;
        """ )
    def fwd_of( x ):
        inp = RealTensor()
        inp.set( x )
        out = RealTensor()
        loom.ffi_call(
            "test_call_der",
            avant,
            arriere,
            out = loom.out( out ),
            inp = inp,
        )
        return out.raw

    # forward: 2 * 17 + 100 = 134
    assert float( fwd_of( driver.array( 17.0 ) ) ) == 134

    # backward: d( 2 * inp + 100 ) / d inp = 2
    g = driver.grad( fwd_of )( driver.array( 17.0 ) )
    assert float( g ) == 2


if test( "der_symbolic_zero" ):
    # two outputs, and a loss that uses only one of them: the cotangent of the UNUSED output is a
    # symbolic zero, so `grad_for_out_b` reaches the backward kernel as a `ZeroTensor` -- read as
    # 0, no buffer. The body multiplies by it and the term simply vanishes.
    avant = FfiCode.per_item( code = """
            outputs.out_a = 2 * inputs.inp;
            outputs.out_b = 3 * inputs.inp;
        """ )
    arriere = FfiCode.per_item( """
            grad_of_inputs.inp = 2 * grad_of_outputs.out_a + 3 * grad_of_outputs.out_b;
        """ )
    def only_a( x ):
        inp = RealTensor()
        inp.set( x )
        out_a = RealTensor()
        out_b = RealTensor()
        loom.ffi_call(
            "test_call_der_sz",
            avant,
            arriere,
            out_a = loom.out( out_a ),
            out_b = loom.out( out_b ),
            inp = inp,
        )
        return out_a.raw   # `out_b` is never used: its cotangent is a symbolic zero

    # d( 2 * inp ) / d inp = 2 -- the `3 * grad_for_out_b` term drops (ZeroTensor)
    g = driver.grad( only_a )( driver.array( 5.0 ) )
    assert float( g ) == 2


if test( "der_non_perturbed" ):
    # two float inputs, but only one is a function of the differentiated variable: the other is a
    # constant, so Jax does not perturb it. Its gradient is never requested, so `grad_for_bias`
    # reaches the backward kernel as a `NoneTensor` -- `is_valid` is a compile-time false, and
    # the body simply does not compute it (nor is a buffer allocated for it).
    avant = FfiCode.per_item( code = """
            outputs.out = inputs.inp + inputs.bias;
        """ )
    arriere = FfiCode.per_item( """
            // the perturbation is a COMPILE-TIME fact here: `grad_of_inputs.inp` is a real
            // gradient buffer, `grad_of_inputs.bias` a `NoneTensor` (inputs.bias is never perturbed).
            static_assert( grad_of_inputs.inp .is_valid );
            static_assert( ! grad_of_inputs.bias.is_valid );
            
            // a `NoneTensor` has no `operator=`, so its write must be dropped at COMPILE
            // time -- `if constexpr` on `is_valid`, not a runtime `if`.
            if constexpr ( grad_of_inputs.inp.is_valid )
                grad_of_inputs.inp = grad_of_outputs.out;
            if constexpr ( grad_of_inputs.bias.is_valid )
                grad_of_inputs.bias = grad_of_outputs.out;
        """ )
    def loss( x ):
        inp = RealTensor()
        inp.set( x )
        bias = RealTensor()
        bias.set( driver.array( 100.0 ) )   # a constant: not a function of `x`, so non-perturbed
        out = RealTensor()
        loom.ffi_call(
            "test_call_der_np",
            avant,
            arriere,
            out = loom.out( out ),
            inp = inp,
            bias = bias,
        )
        return out.raw

    assert float( loss( driver.array( 5.0 ) ) ) == 105

    # d( inp + bias ) / d inp = 1 -- `bias` is never perturbed, so `grad_for_bias` is a NoneTensor
    g = driver.grad( loss )( driver.array( 5.0 ) )
    assert float( g ) == 1


if test( "der_shape_var" ):
    # a differentiable tensor whose shape is driven by a `ShapeVar`: the gradient of an input is
    # allocated at the input's capacity, read back from its buffer through the axis they share.
    n = ShapeVar()
    ax = Axis( n )
    ax.name = "n"   # a standalone axis: stamp the name the generated C++ uses (`DEFINE_AXIS( n )`)

    avant = FfiCode.per_item( code = """
            outputs.out( n = 0 ) = 2 * inputs.vec( n = 0 );
            outputs.out( n = 1 ) = 3 * inputs.vec( n = 1 );
        """ )
    arriere = FfiCode.per_item( """
            if ( grad_of_inputs.vec.is_valid && ! grad_of_outputs.out.surely_null ) {
                grad_of_inputs.vec( n = 0 ) = 2 * grad_of_outputs.out( n = 0 );
                grad_of_inputs.vec( n = 1 ) = 3 * grad_of_outputs.out( n = 1 );
            }
        """ )
    def loss( x ):
        vec = RealTensor[ ax ]()
        vec.set( x )            # length-2 vector -> `n` is solved to 2 from the data
        out = RealTensor[ ax ]()
        loom.ffi_call(
            "test_call_der_sv",
            avant,
            arriere,
            out = loom.out( out ),
            vec = vec,
        )
        return out.raw.sum()    # loss = 2*vec[0] + 3*vec[1]

    assert float( loss( driver.array( [ 1.0, 1.0 ] ) ) ) == 5

    # d loss / d vec = [ 2, 3 ], and the gradient buffer is sized like `vec` (capacity 2)
    g = driver.grad( loss )( driver.array( [ 1.0, 1.0 ] ) )
    assert [ float( v ) for v in g ] == [ 2, 3 ]


if test( "der_aggregate" ):
    # differentiating through an AGGREGATE argument: the backward gets a `grad_for_cell` of the
    # same class, mirrored member by member -- `grad_for_cell.data` is the gradient of the input
    # `cell.data`, allocated at its capacity (the shared `nn` resolves it). The residual `cell`
    # re-enters under its forward name; non-tensor members (`n`, `nn`) are shared.
    class Vec( Aggregate ):
        data : RealTensor[ "n" ]
        n    : Axis[ "nn" ]
        nn   : CtShapeVar


    avant = FfiCode.per_item( code = """
            outputs.out = 2 * inputs.cell.data( n = 0 ) + 3 * inputs.cell.data( n = 1 );
        """ )
    arriere = FfiCode.per_item( """
            if ( ! grad_of_outputs.out.surely_null && grad_of_inputs.cell.data.is_valid ) {
                grad_of_inputs.cell.data( n = 0 ) = 2 * grad_of_outputs.out;
                grad_of_inputs.cell.data( n = 1 ) = 3 * grad_of_outputs.out;
            }
        """ )
    def loss( x ):
        cell = Vec( nn = 2 )
        cell.data = x           # a float INPUT member
        out = RealTensor()          # a bare scalar output
        loom.ffi_call(
            "test_call_der_agg",
            avant,
            arriere,
            out = loom.out( out ),
            cell = cell,
        )
        return out.raw          # loss = 2*data[0] + 3*data[1]

    assert float( loss( driver.array( [ 1.0, 1.0 ] ) ) ) == 5

    # d loss / d cell.data = [ 2, 3 ], returned through `grad_for_cell.data`
    g = driver.grad( loss )( driver.array( [ 1.0, 1.0 ] ) )
    assert [ float( v ) for v in g ] == [ 2, 3 ]


if test( "batch_alignment_forced" ):
    # PHYSICAL LAYOUT, end to end. Our OWN batch machinery (`batch_axes`, not a jax vmap) runs the
    # kernel over a leading axis. With a hardware BYTE alignment, the flattened batch dimension is
    # PADDED so its byte size aligns -- the output buffer is physically larger than the logical
    # batch count. The kernel still iterates only the real items (`global_batch_indices` is the
    # prescribed batch size, not the capacity), and the result reads back LOGICALLY identical.
    #
    # This exercises the non-identity path completely: a padded `jax_out_spec`, the 4-arg
    # `tensor_view` (logical extents + physical BYTE strides), and `Tensor.value`'s gather.
    from loom.tensor import new_batch_axis

    class Cell7( Aggregate ):
        scale            : RealTensor[ "dim" ]
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar


    noyau = FfiCode.per_item( code = """
    auto c = outputs.cell( batch_index );
    c.nb_vertices.set( 1 );
    c.vertex_positions( num_vertex = 0, dim = 0 ) = c.scale( dim = 0 );
    c.vertex_positions( num_vertex = 0, dim = 1 ) = c.scale( dim = 1 );
    """ )
    def run( alignment ):
        cell = Cell7( nb_dims = 2, batch_axes = [ new_batch_axis( 3 ) ] )
        cell.scale = driver.array( [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ] )
        loom.ffi_call(
            "test_call_batch_align",
            noyau,
            cell = loom.out( cell, "nb_vertices", "vertex_positions", capacities = { "nb_vertices": 4 } ),
            batch_alignment = alignment,
        )
        return cell.vertex_positions

    plain  = run( 1 )
    padded = run( 32 )    # 32 bytes / 8 (fp64) = 4 items -> batch of 3 rounds up to 4

    # padding really happened: the physical buffer gained rows (leading dim 4, not 3) ...
    assert padded.raw.shape[ 0 ] == 4
    assert plain.raw.shape[ 0 ] == 3

    # ... yet the LOGICAL value read back is byte-for-byte the one the unpadded run produced:
    # each batch item wrote its own `scale` row into vertex 0.
    want = [ [ [ 1, 2 ] ], [ [ 3, 4 ] ], [ [ 5, 6 ] ] ]
    assert plain.value.tolist()  == want
    assert padded.value.tolist() == want


if test( "physical_axis_reorder" ):
    # PHASE 3: a physical-order REORDER. The input tensor is stored COLUMN-MAJOR (its two axes
    # permuted in memory), a layout with NON-MONOTONIC strides. The kernel indexes it by LOGICAL
    # name (`row=`, `col=`) and must recover the right values -- proving the 4-arg `tensor_view`
    # honours an arbitrary per-axis stride, so a hardware reorder is pure performance.
    import numpy
    from loom.tensor import PhysicalLayout
    from loom.tensor import ReferenceShape
    from loom.tensor import Storage
    from loom import Axis, ShapeVar, Tensor

    noyau = FfiCode.per_item( code = """
    outputs.out( batch_index, row = 0, col = 0 ) = inputs.m( batch_index, row = 0, col = 0 );
    outputs.out( batch_index, row = 0, col = 1 ) = inputs.m( batch_index, row = 0, col = 1 );
    outputs.out( batch_index, row = 1, col = 0 ) = inputs.m( batch_index, row = 1, col = 0 );
    outputs.out( batch_index, row = 1, col = 1 ) = inputs.m( batch_index, row = 1, col = 1 );
    """ )
    row = Axis( ShapeVar( 2 ), name = "row" )
    col = Axis( ShapeVar( 2 ), name = "col" )

    # logical m = [ [ 1, 2 ], [ 3, 4 ] ], but laid out column-major: phys strides [ row=1, col=2 ].
    L = PhysicalLayout.of( [ 2, 2 ], [ False, False ], phys_num = [ 1, 0 ] )
    assert L.buffer_shape == [ 2, 2 ] and L.strides == [ 1, 2 ] and not L.is_identity

    m = RealTensor[ row, col ]()
    m.storage = Storage.of( driver.array( [ [ 1, 3 ], [ 2, 4 ] ] ),   # the physical (col-major) buffer
                            ReferenceShape.from_dense_shape( [ 2, 2 ] ), L )
    assert numpy.asarray( m.value ).tolist() == [ [ 1, 2 ], [ 3, 4 ] ]   # reads back logical

    out = RealTensor[ row, col ]()
    loom.ffi_call(
        "test_call_phys_reorder",
        noyau,
        m = m,
        out = loom.out( out ),
    )

    # the kernel read the permuted input by name and copied it: the logical value is preserved.
    assert numpy.asarray( out.value ).tolist() == [ [ 1, 2 ], [ 3, 4 ] ]


if test( "fill_crosses_as_a_storageless_FillTensor" ):
    # A FILL is a value whose every element reads the same scalar. It crosses the FFI as ONE rank-0
    # buffer -- not as an [n] array -- and its logical extents are not baked into the source either:
    # the C++ view reads them off a SIBLING argument that carries the same axis. So one compiled
    # kernel serves any fill value and any size.
    from loom.tensor import Fill
    import numpy

    n   = ShapeVar()
    num = Axis( n, name = "num" )

    x = RealTensor[ num ]( [ 10.0, 20.0, 30.0, 40.0 ] )
    f = RealTensor[ num ].filled_with( 2.5 )
    assert isinstance( f.storage, Fill ) and f.is_fill
    assert f.raw.shape == ()            # ONE scalar backs it...
    assert f.capacity == ( 4, )         # ...over four logical elements

    out = RealTensor[ num ]()

    loom.ffi_call(
        "test_call_fill",
        FfiCode.per_item( code = """
        // the same scalar whatever the index -- indexing a fill ignores the index
        outputs.out( batch_index, num = 0 ) = inputs.x( batch_index, num = 0 ) * inputs.f( batch_index, num = 0 );
        outputs.out( batch_index, num = 1 ) = inputs.x( batch_index, num = 1 ) * inputs.f( batch_index, num = 3 );
        // its logical extent, filled in from the sibling buffer that carries `num`
        outputs.out( batch_index, num = 2 ) = inputs.f.size();
        // and it is a distinct TYPE, not a TensorView the kernel has to test
        static_assert( ! std::is_same_v< decltype( inputs.f ), decltype( inputs.x ) > );
        outputs.out( batch_index, num = 3 ) = 0;
        """ ),
        x = x,
        f = f,
        out = loom.out( out ),
    )

    assert numpy.asarray( out.value ).tolist() == [ 25.0, 50.0, 4.0, 0.0 ]


if test( "a_plain_count_crosses_by_value_not_through_a_buffer" ):
    # A count is ONE integer. When the host knows it and the kernel only READS it, sending it
    # through a device buffer costs an allocation, a transfer and a dereference per read, and buys
    # nothing -- the value is uniform over the whole call. So it travels as an FFI attribute and
    # lands in the kernel as a `ScalarValue<SI>`, in registers.
    #
    # A buffer is kept exactly where it is unavoidable: a count the KERNEL writes (that is where
    # the result goes), a RAGGED one (a count per segment), or one the host does not know.
    import numpy

    class Counter( Aggregate ):
        out       : RealTensor[ "num" ]

        num       : Axis[ "nb_out" ]

        nb_out    : ShapeVar     # written by the kernel -> a buffer: it is the result
        nb_wanted : ShapeVar     # prescribed, only read   -> crosses by value

    noyau = FfiCode.per_item( code = """
    auto c = outputs.cnt( batch_index );
    
    static_assert( std::is_same_v< std::decay_t< decltype( c.nb_wanted.view ) >, ScalarValue<SI> >,
                   "a host-known, read-only count must cross by value" );
    static_assert( ! std::is_same_v< std::decay_t< decltype( c.nb_out.view ) >, ScalarValue<SI> >,
                   "a count the kernel writes needs a real buffer" );
    
    c.nb_out.set( c.nb_wanted );
    for ( SI n = 0; n < SI( c.nb_out ); ++n )
        c.out( num = n ) = 10 * n;
    """ )
    cnt = Counter( nb_wanted = 3 )
    loom.ffi_call(
        "test_call_scalar_count",
        noyau,
        cnt = loom.out( cnt, "nb_out", "out", capacities = { "nb_out": 8 } ),
    )

    assert cnt.nb_out.value == 3
    assert numpy.asarray( cnt.out.value ).tolist() == [ 0.0, 10.0, 20.0 ]


if test( "une_sortie_nue_est_semee" ):
    # UN TAMPON DE SORTIE PART SEME, y compris quand le tenseur est passe NU.
    #
    # `JaxFfi._render_call` demande un `cpp_seed_root` a chaque argument racine qui en a un, et
    # seuls les AGREGATS en avaient : une sortie tensorielle nue -- le cas le plus simple -- n'etait
    # jamais semee, alors que `LOOM_ZERO_OUTPUTS` promet le contraire.
    #
    # On le teste sous `poison` et pas sous le zero par defaut : un tampon fraichement alloue vaut
    # souvent zero par chance, donc le defaut ne distingue pas « seme » de « chanceux ». Le poison,
    # lui, ne peut venir que du semis -- et c'est aussi ce qui le rend utile : une ecriture oubliee
    # devient un NaN qui se propage, au lieu d'un zero credible qui passe les tests.
    import math, os
    import numpy

    ancien = os.environ.get( "LOOM_ZERO_OUTPUTS" )
    os.environ[ "LOOM_ZERO_OUTPUTS" ] = "poison"
    try:
        n = ShapeVar( 8 )
        ax = Axis( n )
        ax.name = "seed_n"

        out = RealTensor[ ax ]()
        loom.ffi_call(
            "test_seed_bare",
            FfiCode.per_item( code = "outputs.out( seed_n = 0 ) = 1;" ),
            out = loom.out( out ),
        )

        vals = numpy.asarray( out.raw ).reshape( -1 ).tolist()
        assert vals[ 0 ] == 1
        assert all( math.isnan( v ) for v in vals[ 1 : ] ), vals
    finally:
        if ancien is None:
            del os.environ[ "LOOM_ZERO_OUTPUTS" ]
        else:
            os.environ[ "LOOM_ZERO_OUTPUTS" ] = ancien


if test( "un_axe_de_batch_vivant_ne_renomme_pas_le_noyau" ):
    # LE NOM D'UN AXE DE BATCH NE DOIT PAS ATTEINDRE LA SOURCE.
    #
    # `CallArgsAnalysis` renomme les axes de batch d'un appel en `batch_0`, `batch_1`, ... dans
    # l'ordre ou ils s'y presentent. Sans ca, le nom venait de l'objet `Axis`, donc d'une piscine
    # d'indices empruntes a la VIE des objets : deux appels structurellement identiques rendaient
    # deux sources differentes des que leurs axes etaient vivants EN MEME TEMPS -- ce que fait
    # toute chaine d'appels que l'adjoint doit remonter. Mesure avant correction sur une chaine de
    # dix pas : 30 noyaux compiles au lieu de 3.
    #
    # D'ou la forme du test : on RETIENT le premier axe pendant le deuxieme appel. C'est
    # exactement la situation qui produisait un nom neuf.
    from loom.compilation import journal
    from loom.tensor import new_batch_axis

    class Sortie( Aggregate ):
        val : RealTensor


    noyau = FfiCode.per_item( code = "outputs.res.val( batch_index ) = 1;" )
    def un_appel():
        axe = new_batch_axis( 3, prefix = "essai" )
        res = Sortie( batch_axes = [ axe ] )
        loom.ffi_call(
            "test_axe_canonique",
            noyau,
            res = loom.out( res ),
        )
        return axe, res

    vivants = [ un_appel() ]                       # l'axe reste vivant : c'est le point
    avant = journal.stats()
    vivants.append( un_appel() )
    apres = journal.stats()

    assert apres[ "kernels" ] == avant[ "kernels" ], \
        f"le deuxieme appel a fabrique un noyau de plus ({ avant[ 'kernels' ] } -> { apres[ 'kernels' ] })"
    assert apres[ "reuses" ] > avant[ "reuses" ], "le deuxieme appel n'a pas resservi la cible du premier"
    # `.value` ET PAS `.raw` : `raw` est le tampon, dimensionne a la CAPACITE. L'alignement de
    # batch vaut 128 octets sur CUDA, donc un lot de 3 occupe SEIZE fentes en fp64 -- et cette
    # assertion comparait les 16 a une liste de 3. C'etait le dernier echec de l'arbre, et
    # c'etait ce piege-la.
    assert [ float( v ) for v in vivants[ 1 ][ 1 ].val.value ] == [ 1.0, 1.0, 1.0 ]


if test( "une_valeur_brute_entre_telle_quelle" ):
    # CE QU'ON N'ECRIT PLUS. Un tableau numpy, un tableau du framework, un flottant python
    # traversent SANS emballage : `Tensor.as_tensor` lit leur kind ( un fait ) et laisse la
    # taille au driver ( une politique ) -- exactement ce que `RealTensor( u )` faisait a la
    # main. Les axes sont deduits de la forme, donc anonymes, donc nommes par leur POSITION
    # dans le C++ : deux valeurs de meme rang tombent sur la meme grille.
    import numpy

    noyau = FfiCode( code = """
        namespace {
            struct Ajoute {
                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const TF d = args.inputs.decalage;
                    args.outputs.sortie( coords ) = args.inputs.entree( coords ) + d;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( Ajoute(), args.outputs.sortie.domain(), args, batch_axes );
            }
        }
    """ )

    entree = numpy.arange( 6.0 ).reshape( 2, 3 )    # ni tenseur loom, ni tableau du framework
    sortie = RealTensor.like( entree )              # `like` lit la forme d'une valeur brute

    loom.ffi_call(
        "test_valeur_brute",
        noyau,
        entree = entree,
        decalage = 10.0,
        sortie = loom.out( sortie ),
    )

    assert numpy.asarray( sortie.raw ).tolist() == [ [ 10, 11, 12 ], [ 13, 14, 15 ] ]

    # ... SAUF en sortie : le tenseur serait bati par l'appel, donc le resultat n'aurait nulle
    # part ou revenir. C'est dit, au lieu d'etre ecrit dans le vide.
    try:
        loom.ffi_call(
            "test_valeur_brute",
            noyau,
            entree = entree,
            decalage = 10.0,
            sortie = loom.out( numpy.zeros( ( 2, 3 ) ) ),
        )
        assert False, "une sortie brute aurait du etre refusee"
    except ValueError as e:
        assert "brute" in str( e ), e


if test( "le_role_se_dit_sur_la_valeur" ):
    # LE VOCABULAIRE D'ARGUMENTS DE `ffi_call`. Le role est porte par la valeur, pas par une
    # liste de chemins a cote -- donc il ne peut pas designer autre chose qu'elle, et les noms
    # que `driver.call` reservait sont rendus au noyau.
    import numpy
    import loom

    # `output_attributes` EST ICI UN ARGUMENT DU NOYAU, et il vit a `args.inputs.output_attributes`.
    # C'est le test : ce nom etait vole par `driver.call`, et l'ecrire y aurait declare une liste de
    # sorties vide. Sous les groupes le premier niveau ne contient que les groupes, donc plus rien
    # ne peut etre vole.
    ajoute = FfiCode( code = """
        namespace {
            struct Ajoute {
                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const TF d = args.inputs.output_attributes;
                    args.outputs.sortie( coords ) = args.inputs.entree( coords ) + d;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( Ajoute(), args.outputs.sortie.domain(), args, batch_axes );
            }
        }
    """ )

    entree = numpy.arange( 6.0 ).reshape( 2, 3 )
    attendu = [ [ 10, 11, 12 ], [ 13, 14, 15 ] ]

    sortie = RealTensor.like( entree )
    rendu = loom.ffi_call( "test_role_out", ajoute,
                           entree = entree,
                           output_attributes = 10.0,
                           sortie = loom.out( sortie ) )

    # `loom.out` ne rend rien : l'objet etait deja le notre, le resultat y est reecrit.
    assert rendu is None
    assert numpy.asarray( sortie.raw ).tolist() == attendu


if test( "un_argument_mutable_rend_sa_nouvelle_valeur" ):
    # UN NOM, DEUX TAMPONS. Les entrees et les sorties d'un appel sont disjointes, donc une mise
    # a jour en place est deux tampons plus un rebinding : c'est ce que `loom.mutable` ecrit.
    import numpy
    import loom

    ajoute = FfiCode( code = """
        namespace {
            struct Ajoute {
                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const TF d = args.inputs.decalage;
                    args.outputs.v( coords ) = args.inputs.v( coords ) + d;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( Ajoute(), args.outputs.v.domain(), args, batch_axes );
            }
        }
    """ )

    entree = numpy.arange( 6.0 ).reshape( 2, 3 )
    attendu = [ [ 10, 11, 12 ], [ 13, 14, 15 ] ]

    # CE QUI REVIENT EST DE LA MEME ESPECE QUE CE QU'ON A DONNE. Un tableau brut...
    brut = loom.ffi_call( "test_role_mut", ajoute, v = loom.mutable( entree ), decalage = 10.0 )
    assert not isinstance( brut, Tensor )
    assert numpy.asarray( brut ).tolist() == attendu

    # ... un tenseur loom.
    tenseur = loom.ffi_call( "test_role_mut", ajoute,
                             v = loom.mutable( RealTensor( entree ) ), decalage = 10.0 )
    assert isinstance( tenseur, Tensor )
    assert numpy.asarray( tenseur.raw ).tolist() == attendu

    # une chaine : c'est ce que `mutable` achete, et le resultat doit etre celui des pas repetes
    v = entree
    for _ in range( 3 ):
        v = loom.ffi_call( "test_role_mut", ajoute, v = loom.mutable( v ), decalage = 10.0 )
    assert numpy.asarray( v ).tolist() == [ [ 30, 31, 32 ], [ 33, 34, 35 ] ]

    # LES NOMS DERIVES ENTRENT DANS LE MEME NAMESPACE : une collision ferait lire au noyau le
    # mauvais tampon, en silence. Elle est refusee.
    try:
        loom.ffi_call( "test_role_mut", ajoute, v = loom.mutable( entree ),
                       v_input = entree, decalage = 10.0 )
        assert False, "la collision sur 'v_input' aurait du etre refusee"
    except ValueError as e:
        assert "v_input" in str( e ), e
