import sys
from errand import has_tag

# The pointer FFI is what numpy, torch and cupy run kernels through ( `drivers/PointerFfi.py` ): Jax has
# its own, inside the XLA program, where there is no address to take.
if not ( has_tag( "driver=numpy" ) or has_tag( "driver=torch" ) or has_tag( "driver=cupy" ) ):
    sys.exit( 0 )

import numpy
import loom
import loom
from loom import RealTensor
from loom.compilation.FfiCode import FfiCode
from errand import test
from loom.testing import need

# What crosses is an ADDRESS: a value the framework already holds is handed to the kernel where it
# sits, and only a value the kernel cannot read in place is conformed first.

SHIFT = FfiCode( code = """
    namespace {
        struct AddShift {
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.output_value( coords ) = args.inputs.input_value( coords ) + args.inputs.shift;
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( AddShift(), args.outputs.output_value.domain(), args, batch_axes );
        }
    }
""" )


def _address( x ):
    if isinstance( x, numpy.ndarray ):
        return x.ctypes.data
    return x.data.ptr if hasattr( x, "data" ) and hasattr( x.data, "ptr" ) else x.data_ptr()


def _host( x ):
    """What a value holds, as nested lists -- whichever framework, wherever it lives (a card included)."""
    return loom.to_numpy( x ).tolist()


def _spy():
    """The operands the engine builds from now on, as `( list, restore )`."""
    import importlib
    seen = []
    module = { "numpy": "NumpyFfi", "torch": "TorchFfi", "cupy": "CupyFfi" }[ str( loom.resolved_framework() ) ]
    adapter = importlib.import_module( f"loom.drivers.{ module }" )._adapter
    original = adapter.operand

    def operand( x, device, *more ):
        res = original( x, device, *more )
        seen.append( ( x, res ) )
        return res

    adapter.operand = operand
    return seen, lambda: setattr( adapter, "operand", original )


def _shift( value, name ):
    out = RealTensor.like( value )
    loom.ffi_call( name, SHIFT, input_value = value, shift = 10.0, output_value = loom.out( out ) )
    return out


if test( "a_dense_input_crosses_by_address" ):
    value = loom.array( numpy.arange( 6.0 ).reshape( 2, 3 ) )
    seen, restore = _spy()
    try:
        out = _shift( value, "test_pointer_dense" )
    finally:
        restore()

    assert _host( out.raw ) == [ [ 10, 11, 12 ], [ 13, 14, 15 ] ]
    ours = [ ( x, op ) for x, op in seen if x is value ]
    assert ours, "the input was not handed to the adapter as it is"
    for x, op in ours:
        assert op.address == _address( x ), "a dense input was copied before the kernel read it"


def _shift_in_place( view, name ):
    """`view + 10` through the kernel, and what the engine did with the input: `( values, operand )`."""
    seen, restore = _spy()
    try:
        out = _shift( view, name )
    finally:
        restore()
    ours = [ op for x, op in seen if x is view ]
    assert ours, "the input was not handed to the adapter as it is"
    return _host( out.raw ), ours[ -1 ]


if test( "a_transposed_input_is_read_where_it_is" ):
    # same memory, other order: the kernel reads the strides of the view -- no copy, the LOGICAL values
    base = loom.array( numpy.arange( 6.0 ).reshape( 3, 2 ) )
    view = base.T
    values, operand = _shift_in_place( view, "test_pointer_transposed" )
    assert values == [ [ 10, 12, 14 ], [ 11, 13, 15 ] ]
    assert operand.address == _address( view ), "a strided input was copied before the kernel read it"
    assert operand.strides_bytes == ( 8, 16 )


if test( "a_slice_is_read_where_it_is" ):
    # an offset (the first column is skipped) and a row stride wider than the row
    base = loom.array( numpy.arange( 12.0 ).reshape( 3, 4 ) )
    view = base[ :, 1: ]
    values, operand = _shift_in_place( view, "test_pointer_slice" )
    assert values == [ [ 11, 12, 13 ], [ 15, 16, 17 ], [ 19, 20, 21 ] ]
    assert operand.address == _address( view ) and operand.address != _address( base )


if test( "a_broadcast_is_read_with_a_null_stride" ):
    # `expand`/`broadcast_to`: a stride of 0 -- the memory of one row, read as three
    row = loom.array( [ 1.0, 2.0, 3.0 ] )
    view = row.expand( 3, 3 ) if hasattr( row, "expand" ) else numpy.broadcast_to( row, ( 3, 3 ) )
    values, operand = _shift_in_place( view, "test_pointer_broadcast" )
    assert values == [ [ 11, 12, 13 ] ] * 3
    assert operand.address == _address( view )
    assert operand.strides_bytes[ 0 ] == 0


if test( "strided_and_dense_inputs_do_not_share_a_kernel_by_accident" ):
    # the same call on a dense value and on a strided one: both right, whichever comes first
    dense = loom.array( numpy.arange( 6.0 ).reshape( 2, 3 ) )
    strided = loom.array( numpy.arange( 6.0 ).reshape( 3, 2 ) ).T
    for value in ( dense, strided, dense, strided ):
        values, _ = _shift_in_place( value, "test_pointer_mixed" )
        expected = ( numpy.asarray( _host( value ) ) + 10 ).tolist()
        assert values == expected, ( values, expected )


if test( "a_vmap_is_one_launch_of_a_batched_kernel" ):
    need( "vmap" )
    # `torch.func.vmap` maps the call over a new axis, and the KERNEL runs it: N items, one launch --
    # not N calls of the unbatched kernel (what a Python loop over the items would be).
    from loom.drivers import PointerFfi
    launches = []
    original = PointerFfi.run

    def counting( *args, **kwargs ):
        launches.append( 1 )
        return original( *args, **kwargs )

    def shifted( x ):
        return _shift( x, "test_pointer_vmap" ).raw

    PointerFfi.run = counting
    try:
        many = loom.vmap( shifted )( loom.array( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ] ) )
    finally:
        PointerFfi.run = original

    assert _host( many ) == [ [ 11, 12 ], [ 13, 14 ], [ 15, 16 ] ]
    assert len( launches ) == 1, f"{ len( launches ) } launches for 3 items"


if test( "a_gradient_goes_through_a_vmap" ):
    need( "grad" )
    need( "vmap" )
    # the batched kernel's backward is a batched kernel too: `grad( vmap( f ) )` is one autograd graph
    forward = FfiCode.per_item( code = """
            outputs.out( batch_index ) = 2 * inputs.inp( batch_index ) + 100;
        """ )
    backward = FfiCode.per_item( """
            if ( ! grad_of_outputs.out.surely_null && grad_of_inputs.inp.is_valid )
                grad_of_inputs.inp( batch_index ) = 2 * grad_of_outputs.out( batch_index );
        """ )

    def fwd_of( x ):
        inp = RealTensor()
        inp.set( x )
        out = RealTensor()
        loom.ffi_call( "test_pointer_vmap_der", forward, backward, out = loom.out( out ), inp = inp )
        return out.raw

    xs = loom.array( [ 1.0, 2.0, 3.0 ] )
    assert _host( loom.vmap( fwd_of )( xs ) ) == [ 102, 104, 106 ]
    g = loom.grad( lambda v: loom.vmap( fwd_of )( v ).sum() )( xs )
    assert _host( g ) == [ 2, 2, 2 ]


if test( "a_jitted_function_gives_the_values_of_the_eager_one" ):
    # `loom.jit` is `torch.compile` under Torch, the identity under numpy, XLA under Jax: whichever, same values.
    # Under Torch the kernel call is a break in the graph; the tensor code around it is compiled.
    forward = FfiCode.per_item( code = "outputs.out = 2 * inputs.inp + 100;" )

    def fwd_of( x ):
        inp = RealTensor()
        inp.set( x )
        out = RealTensor()
        loom.ffi_call( "test_pointer_jit", forward, out = loom.out( out ), inp = inp )
        return out.raw

    def f( x ):
        return fwd_of( x * 3 ) + 1

    x = loom.array( 17.0 )
    assert float( _host( f( x ) ) ) == 203
    assert float( _host( loom.jit( f )( x ) ) ) == 203
    assert float( _host( loom.jit( f )( loom.array( 1.0 ) ) ) ) == 107


if test( "a_gradient_goes_through_a_jit" ):
    need( "grad" )
    forward = FfiCode.per_item( code = "outputs.out = 2 * inputs.inp + 100;" )
    backward = FfiCode.per_item( """
            if ( ! grad_of_outputs.out.surely_null && grad_of_inputs.inp.is_valid )
                grad_of_inputs.inp = 2 * grad_of_outputs.out;
        """ )

    def loss( x ):
        inp = RealTensor()
        inp.set( x * 3 )
        out = RealTensor()
        loom.ffi_call( "test_pointer_jit_der", forward, backward, out = loom.out( out ), inp = inp )
        return out.raw * 1

    # d/dx ( 2 * 3x + 100 ) = 6
    assert float( _host( loom.grad( loom.jit( loss ) )( loom.array( 17.0 ) ) ) ) == 6
