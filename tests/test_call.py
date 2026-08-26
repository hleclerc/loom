from loom import CtShapeVar, ShapeVar, Axis, Tensor, Aggregate, driver, FfiCode, RealTensor, IntTensor
from loom.testing import test

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

    # the body runs on a DEVICE. `queue` is the call's execution context (a `sdot::Queue`, chosen
    # by the device -- a typedef, since the memory space a pointer lives in is part of its type),
    # and `global_batch_indices` is what the kernel iterates over: by default a single item, the
    # empty multi-index (a `vmap` is what will give it axes).
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
    driver.call(
        FfiCode( name = "test_call_basic", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                cell.nb_vertices( batch_index ).set( 1 );
                cell.vertex_positions( batch_index, dim = 0, num_vertex = 0 ) = 1;
                cell.vertex_positions( batch_index, dim = 1, num_vertex = 0 ) = 2;
            },
            cell_io, cell
        );
        """ ),
        cell = cell,
        output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
        output_capacities = { "cell.nb_vertices": 8 },   # => `vertex_positions` is allocated 8x2
        # frame = driver.array( [ [ 0 ] ] )
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

    driver.call(
        FfiCode( name = "test_call_basic_res", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell, auto res ) {
                res( batch_index, num_vertex = 0 ) = cell.nb_vertices( batch_index );
            },
            cell_io, cell, OutList(), res
        );
        """ ),
        cell = cell,
        res = res,
        output_attributes = [ "res" ],
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

    driver.call(
        FfiCode( name = "test_partial_init", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                cell.nb_vertices( batch_index ).set( 1 );
                // a fresh output buffer is NOT guaranteed zero-initialized (see
                // `ProjectedSumOfDiracs::zero_position_grad`'s docstring for the general fact) --
                // write every element the assertion below reads, rather than relying on a
                // leftover-memory default.
                cell.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 1;
                cell.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 0;

                static_assert( DECAYED_TYPE_OF( cell.vertex_positions.is_valid() )::value == 1 );
                static_assert( DECAYED_TYPE_OF( cell.vertex_indices  .is_valid() )::value == 0 );
            },
            cell_io, cell
        );
        """ ),
        cell = cell,
        output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
        output_capacities = { "cell.nb_vertices": 8 },
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

    driver.call(
        FfiCode( name = "test_input_exceptions_init", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                cell.nb_vertices( batch_index ).set( 1 );
                cell.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 1;
                cell.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 2;
            },
            cell_io, cell
        );
        """ ),
        cell = cell,
        output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
        output_capacities = { "cell.nb_vertices": 8 },
    )

    driver.call(
        FfiCode( name = "test_input_exceptions_use", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                static_assert( DECAYED_TYPE_OF( cell.vertex_positions.is_valid() )::value == 0 );
            },
            cell_io, cell
        );
        """ ),
        cell = cell,
        input_exceptions = [ "cell.vertex_positions" ],
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

    driver.call(
        FfiCode( name = "two_instances", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto flat, auto volu ) {
                // an index applies to a whole aggregate just as well as to one of its members:
                // `f( batch_index ).nb_vertices` and `flat.nb_vertices( batch_index )` are the
                // same thing. Handy when every member takes the same index.
                auto f = flat( batch_index );
                f.nb_vertices.set( 1 );
                f.vertex_positions( num_vertex = 0, dim = 0 ) = 1;
                f.vertex_positions( num_vertex = 0, dim = 1 ) = 2;

                volu.nb_vertices( batch_index ).set( 1 );
                // see `partial_init`'s comment above: write every dim, not just the nonzero one.
                volu.vertex_positions( batch_index, num_vertex = 0, dim = 0 ) = 0;
                volu.vertex_positions( batch_index, num_vertex = 0, dim = 1 ) = 0;
                volu.vertex_positions( batch_index, num_vertex = 0, dim = 2 ) = 3;
            },
            flat_io, flat, volu_io, volu
        );
        """ ),
        flat = flat,
        volu = volu,
        output_attributes = [
            "flat.nb_vertices", "flat.vertex_positions",
            "volu.nb_vertices", "volu.vertex_positions",
        ],
        output_capacities = { "flat.nb_vertices": 8, "volu.nb_vertices": 4 },
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

    driver.call(
        FfiCode( name = "test_call_nested", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto pair ) {
                // indexing an aggregate indexes its members -- a nested one included, recursively.
                auto p = pair( batch_index );

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
            },
            pair_io, pair
        );
        """ ),
        pair = pair,
        output_attributes = [ "pair" ],   # a whole subtree can be named at once
        output_capacities = { "pair.left.nb_vertices": 8, "pair.right.nb_vertices": 4 },
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



    code = FfiCode( name = "test_call_vmap", fwd_code = """
    run_parallel(
        queue,
        global_batch_indices,
        []( auto batch_index, auto cell, auto scale ) {
            auto c = cell( batch_index );
            c.nb_vertices.set( 1 );
            c.vertex_positions( num_vertex = 0, dim = 0 ) = scale( batch_index, dim = 0 );
            c.vertex_positions( num_vertex = 0, dim = 1 ) = scale( batch_index, dim = 1 );
        },
        cell_io, cell, InpList(), scale
    );
    """ )

    def positions_of( raw_scale ):
        cell = Cell5( nb_dims = 2 )

        # a bare tensor input, borrowing `cell`'s `dim` axis: one scale per dimension.
        scale = RealTensor[ cell.dim ]()
        scale.set( raw_scale )

        driver.call(
            code,
            cell = cell,
            scale = scale,
            output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
            output_capacities = { "cell.nb_vertices": 4 },
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



    code = FfiCode( name = "test_call_overflow", fwd_code = """
    run_parallel(
        queue,
        global_batch_indices,
        []( auto batch_index, auto cell ) {
            auto c = cell( batch_index );

            // the count may not fit -- and then what one READS BACK is the capacity, never more,
            // which is what makes the loop below safe whatever happens.
            c.nb_vertices.set( c.nb_wanted );
            for ( SI n = 0; n < SI( c.nb_vertices ); ++n )
                c.vertex_positions( num_vertex = n, dim = 0 ) = n;
        },
        cell_io, cell
    );
    """ )

    def cell_of( nb_wanted, capacity ):
        cell = Cell6( nb_dims = 2, nb_wanted = nb_wanted )
        driver.call(
            code,
            cell = cell,
            output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
            output_capacities = { "cell.nb_vertices": capacity },
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
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ] == [ 0, 1, 2, 0 ]

    # 5 into a buffer of 1 -> `max( 5, 2 * 1 ) = 5`: this time it is the count that decides.
    cell = cell_of( 5, 1 )
    assert cell.nb_vertices.value == 5
    assert cell.vertex_positions.capacity == ( 5, 2 )
    assert [ row[ 0 ] for row in cell.vertex_positions.raw.tolist() ] == [ 0, 1, 2, 3, 4 ]


if test( "der" ):
    # a differentiable call: `fwd_code` computes the output, `bwd_code` the input gradients. The
    # backward is generated as an ORDINARY kernel call -- its inputs are the forward inputs and
    # outputs plus the output cotangents (`grad_for_out`), its output the input cotangent
    # (`grad_for_inp`). Jax reaches it through a `custom_vjp` rule.
    #
    # An INTEGER tensor is non-differentiable and non-perturbable: it never gets a `grad_for_`.
    # A symbolically-zero output cotangent lowers to a `ZeroTensor` (`grad_for_out.surely_null()`
    # is a compile-time true), and a non-perturbed input gradient to a `NoneTensor`
    # (`grad_for_inp.is_valid()` a compile-time false) -- either lets the body drop a term at
    # compile time rather than move or multiply a buffer of zeros.
    code = FfiCode( name = "test_call_der",
        fwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto out, auto inp ) {
                    out = 2 * inp + 100;
                },
                OutList{}, out,
                InpList{}, inp
            );
        """,
        bwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto grad_for_inp, auto grad_for_out ) {
                    if ( ! grad_for_out.surely_null() && grad_for_inp.is_valid() )
                        grad_for_inp = 2 * grad_for_out;
                },
                OutList{}, grad_for_inp,
                InpList{}, grad_for_out
            );
        """,
    )

    def fwd_of( x ):
        inp = RealTensor()
        inp.set( x )
        out = RealTensor()
        driver.call( code, output_attributes = [ "out" ], out = out, inp = inp )
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
    code = FfiCode( name = "test_call_der_sz",
        fwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto out_a, auto out_b, auto inp ) {
                    out_a = 2 * inp;
                    out_b = 3 * inp;
                },
                OutList{}, out_a, out_b,
                InpList{}, inp
            );
        """,
        bwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto grad_for_inp, auto grad_for_out_a, auto grad_for_out_b ) {
                    grad_for_inp = 2 * grad_for_out_a + 3 * grad_for_out_b;
                },
                OutList{}, grad_for_inp,
                InpList{}, grad_for_out_a, grad_for_out_b
            );
        """,
    )

    def only_a( x ):
        inp = RealTensor()
        inp.set( x )
        out_a = RealTensor()
        out_b = RealTensor()
        driver.call( code, output_attributes = [ "out_a", "out_b" ],
                     out_a = out_a, out_b = out_b, inp = inp )
        return out_a.raw   # `out_b` is never used: its cotangent is a symbolic zero

    # d( 2 * inp ) / d inp = 2 -- the `3 * grad_for_out_b` term drops (ZeroTensor)
    g = driver.grad( only_a )( driver.array( 5.0 ) )
    assert float( g ) == 2


if test( "der_non_perturbed" ):
    # two float inputs, but only one is a function of the differentiated variable: the other is a
    # constant, so Jax does not perturb it. Its gradient is never requested, so `grad_for_bias`
    # reaches the backward kernel as a `NoneTensor` -- `is_valid()` is a compile-time false, and
    # the body simply does not compute it (nor is a buffer allocated for it).
    code = FfiCode( name = "test_call_der_np",
        fwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto out, auto inp, auto bias ) {
                    out = inp + bias;
                },
                OutList{}, out,
                InpList{}, inp, bias
            );
        """,
        bwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto grad_for_inp, auto grad_for_bias, auto grad_for_out ) {
                    // the perturbation is a COMPILE-TIME fact here: `grad_for_inp` is a real
                    // gradient buffer, `grad_for_bias` a `NoneTensor` (bias is never perturbed).
                    static_assert( DECAYED_TYPE_OF( grad_for_inp .is_valid() )::value == 1 );
                    static_assert( DECAYED_TYPE_OF( grad_for_bias.is_valid() )::value == 0 );

                    // a `NoneTensor` has no `operator=`, so its write must be dropped at COMPILE
                    // time -- `if constexpr` on `is_valid()`, not a runtime `if`.
                    if constexpr ( DECAYED_TYPE_OF( grad_for_inp.is_valid() )::value )
                        grad_for_inp = grad_for_out;
                    if constexpr ( DECAYED_TYPE_OF( grad_for_bias.is_valid() )::value )
                        grad_for_bias = grad_for_out;
                },
                OutList{}, grad_for_inp, grad_for_bias,
                InpList{}, grad_for_out
            );
        """,
    )

    def loss( x ):
        inp = RealTensor()
        inp.set( x )
        bias = RealTensor()
        bias.set( driver.array( 100.0 ) )   # a constant: not a function of `x`, so non-perturbed
        out = RealTensor()
        driver.call( code, output_attributes = [ "out" ], out = out, inp = inp, bias = bias )
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

    code = FfiCode( name = "test_call_der_sv",
        fwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto out, auto vec ) {
                    out( n = 0 ) = 2 * vec( n = 0 );
                    out( n = 1 ) = 3 * vec( n = 1 );
                },
                OutList{}, out,
                InpList{}, vec
            );
        """,
        bwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto grad_for_vec, auto grad_for_out ) {
                    if ( grad_for_vec.is_valid() && ! grad_for_out.surely_null() ) {
                        grad_for_vec( n = 0 ) = 2 * grad_for_out( n = 0 );
                        grad_for_vec( n = 1 ) = 3 * grad_for_out( n = 1 );
                    }
                },
                OutList{}, grad_for_vec,
                InpList{}, grad_for_out
            );
        """,
    )

    def loss( x ):
        vec = RealTensor[ ax ]()
        vec.set( x )            # length-2 vector -> `n` is solved to 2 from the data
        out = RealTensor[ ax ]()
        driver.call( code, output_attributes = [ "out" ], out = out, vec = vec )
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


    code = FfiCode( name = "test_call_der_agg",
        fwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto out, auto cell ) {
                    out = 2 * cell.data( n = 0 ) + 3 * cell.data( n = 1 );
                },
                OutList{}, out,
                InpList{}, cell
            );
        """,
        bwd_code = """
            run_parallel( queue, global_batch_indices,
                []( auto batch_index, auto grad_for_out, auto grad_for_cell ) {
                    if ( ! grad_for_out.surely_null() && grad_for_cell.data.is_valid() ) {
                        grad_for_cell.data( n = 0 ) = 2 * grad_for_out;
                        grad_for_cell.data( n = 1 ) = 3 * grad_for_out;
                    }
                },
                InpList{}, grad_for_out,
                OutList{}, grad_for_cell
            );
        """,
    )

    def loss( x ):
        cell = Vec( nn = 2 )
        cell.data = x           # a float INPUT member
        out = RealTensor()          # a bare scalar output
        driver.call( code, output_attributes = [ "out" ], out = out, cell = cell )
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
    # `tensor_view` (logical extents + physical BYTE strides), and `Tensor.tensor`'s gather.
    from loom.tensor import new_batch_axis

    class Cell7( Aggregate ):
        scale            : RealTensor[ "dim" ]
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar


    code = FfiCode( name = "test_call_batch_align", fwd_code = """
    run_parallel(
        queue,
        global_batch_indices,
        []( auto batch_index, auto cell ) {
            auto c = cell( batch_index );
            c.nb_vertices.set( 1 );
            c.vertex_positions( num_vertex = 0, dim = 0 ) = c.scale( dim = 0 );
            c.vertex_positions( num_vertex = 0, dim = 1 ) = c.scale( dim = 1 );
        },
        cell_io, cell
    );
    """ )

    def run( alignment ):
        cell = Cell7( nb_dims = 2, batch_axes = [ new_batch_axis( 3 ) ] )
        cell.scale = driver.array( [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ] )
        driver.call(
            code,
            cell = cell,
            output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
            output_capacities = { "cell.nb_vertices": 4 },
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
    assert plain.tensor.tolist()  == want
    assert padded.tensor.tolist() == want


if test( "physical_axis_reorder" ):
    # PHASE 3: a physical-order REORDER. The input tensor is stored COLUMN-MAJOR (its two axes
    # permuted in memory), a layout with NON-MONOTONIC strides. The kernel indexes it by LOGICAL
    # name (`row=`, `col=`) and must recover the right values -- proving the 4-arg `tensor_view`
    # honours an arbitrary per-axis stride, so a hardware reorder is pure performance.
    import numpy
    from loom.tensor import PhysicalLayout
    from loom.tensor import ReferenceShape
    from loom import Axis, ShapeVar, Tensor

    code = FfiCode( name = "test_call_phys_reorder", fwd_code = """
    run_parallel( queue, global_batch_indices,
        []( auto batch_index, auto m, auto out ) {
            out( batch_index, row = 0, col = 0 ) = m( batch_index, row = 0, col = 0 );
            out( batch_index, row = 0, col = 1 ) = m( batch_index, row = 0, col = 1 );
            out( batch_index, row = 1, col = 0 ) = m( batch_index, row = 1, col = 0 );
            out( batch_index, row = 1, col = 1 ) = m( batch_index, row = 1, col = 1 );
        },
        InpList(), m, OutList(), out
    );
    """ )

    row = Axis( ShapeVar( 2 ), name = "row" )
    col = Axis( ShapeVar( 2 ), name = "col" )

    # logical m = [ [ 1, 2 ], [ 3, 4 ] ], but laid out column-major: phys strides [ row=1, col=2 ].
    L = PhysicalLayout.of( [ 2, 2 ], [ False, False ], phys_num = [ 1, 0 ] )
    assert L.buffer_shape == [ 2, 2 ] and L.strides == [ 1, 2 ] and not L.is_identity

    m = RealTensor[ row, col ]()
    m._raw   = driver.array( [ [ 1, 3 ], [ 2, 4 ] ] )   # the physical (column-major) buffer
    m._shape = ReferenceShape.from_dense_shape( [ 2, 2 ] )
    m._layout = L
    assert numpy.asarray( m.tensor ).tolist() == [ [ 1, 2 ], [ 3, 4 ] ]   # reads back logical

    out = RealTensor[ row, col ]()
    driver.call( code, m = m, out = out, output_attributes = [ "out" ] )

    # the kernel read the permuted input by name and copied it: the logical value is preserved.
    assert numpy.asarray( out.tensor ).tolist() == [ [ 1, 2 ], [ 3, 4 ] ]
