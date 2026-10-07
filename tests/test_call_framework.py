import numpy
import loom
import loom
from loom import RealTensor
from loom.compilation.FfiCode import FfiCode
from errand import test

# A call runs where its buffers are: the default framework only decides what to BUILD when nothing
# was said, it does not drag an existing buffer into another framework.

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


def _frameworks():
    """`{ name: maker }` for the frameworks importable here, each making a buffer of its own kind."""
    res = {}
    try:
        import jax.numpy as jnp
        res[ "jax" ] = lambda a: jnp.asarray( a )
    except ImportError:
        pass
    try:
        import torch
        res[ "torch" ] = lambda a: torch.as_tensor( a )
    except ImportError:
        pass
    return res


def _shift( value, name ):
    out = RealTensor.like( value )
    loom.ffi_call( name, SHIFT, input_value = value, shift = 10.0, output_value = loom.out( out ) )
    return out


if test( "a_call_follows_the_framework_of_its_buffers" ):
    here = str( loom.resolved_framework() )
    for name, make in _frameworks().items():
        if name == here:
            continue
        value = make( numpy.arange( 6.0 ).reshape( 2, 3 ) )
        out = _shift( value, f"follows_{ name }" )
        assert type( out.raw ).__module__.split( "." )[ 0 ].replace( "jaxlib", "jax" ) == name, ( here, name, type( out.raw ) )
        assert numpy.asarray( out.value ).tolist() == ( numpy.arange( 6.0 ).reshape( 2, 3 ) + 10 ).tolist()
        assert str( loom.resolved_framework() ) == here          # the default is untouched afterwards

if test( "numpy_buffers_ask_for_nothing" ):
    value = numpy.arange( 6.0 ).reshape( 2, 3 )
    out = _shift( value, "follows_numpy" )
    assert numpy.asarray( out.value ).tolist() == ( value + 10 ).tolist()

if test( "an_operation_follows_its_operands" ):
    # tensors held by DIFFERENT frameworks meet in an operation: it runs on one of them, the other
    # crosses by DLPack, and the default framework only decides a tie
    here = str( loom.resolved_framework() )
    makers = _frameworks()
    if len( makers ) >= 2:
        data = numpy.arange( 6.0 ).reshape( 2, 3 )
        a = RealTensor( makers[ "jax" ]( data ) )
        b = RealTensor( makers[ "torch" ]( data ) )
        assert a.storage.framework == "jax" and b.storage.framework == "torch"
        expect = ( data * 2 ).tolist()
        for res in ( a + b, b + a, a * 1 + b, ( a + b ).sum( 0 ) * 0 + a + b ):
            assert res.storage.framework == ( here if here in makers else "jax" ), ( here, res.storage.framework )
        assert numpy.asarray( ( a + b ).value ).tolist() == expect
        assert numpy.asarray( ( b + a ).value ).tolist() == expect
        assert numpy.asarray( ( a @ b.T ).value ).tolist() == ( data @ data.T ).tolist()
        assert numpy.asarray( ( a > 1 ).where( a, b ).value ).tolist() == numpy.where( data > 1, data, data ).tolist()
        # a numpy operand and a tensor of a framework: the framework decides, whatever the default is
        assert ( a + RealTensor( data ) ).storage.framework == "jax"
        assert ( b + RealTensor( data ) ).storage.framework == "torch"
        # a tensor that traces decides
        import torch
        g = RealTensor( torch.as_tensor( data ).requires_grad_() )
        assert ( g + a ).storage.framework == "torch"

if test( "an_unaligned_buffer_is_copied_not_refused" ):
    makers = _frameworks()
    if "jax" in makers and "torch" in makers:
        import torch
        data = torch.zeros( 1001, dtype = torch.float64 )[ 1: ]          # not aligned the way XLA wants
        assert data.data_ptr() % 64 != 0
        a = RealTensor( makers[ "jax" ]( numpy.ones( 1000 ) ) )
        assert numpy.asarray( ( a + RealTensor( data ) ).value ).tolist() == [ 1.0 ] * 1000

SUM = FfiCode( code = """
    namespace {
        struct Add {
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.output_value( coords ) = args.inputs.a( coords ) + args.inputs.b( coords );
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( Add(), args.outputs.output_value.domain(), args, batch_axes );
        }
    }
""" )


def _sum( a, b, name ):
    out = RealTensor.like( a )
    loom.ffi_call( name, SUM, a = a, b = b, output_value = loom.out( out ) )
    return out


if test( "a_foreign_buffer_is_read_in_place" ):
    # a kernel call over buffers of two frameworks that trace nothing runs by ADDRESS: nothing is converted,
    # so an input is read where it sits -- unaligned, strided, whatever -- and the kernel decides what it needs
    makers = _frameworks()
    if "jax" in makers and "torch" in makers:
        import torch
        data = numpy.arange( 6.0 ).reshape( 2, 3 )
        j = makers[ "jax" ]( data )
        # the torch buffer is unaligned: XLA could not read it in place, the pointer path does
        unaligned = torch.zeros( 7, dtype = torch.float64 )[ 1: ].reshape( 2, 3 )
        unaligned.copy_( torch.as_tensor( data ) )
        assert unaligned.data_ptr() % 64 != 0
        out = _sum( RealTensor( j ), RealTensor( unaligned ), "foreign_unaligned" )
        assert out.storage.framework == "torch"          # jax gives way to what reads by address
        assert numpy.asarray( out.value ).tolist() == ( data * 2 ).tolist()
        # a transposed (strided) torch view crosses with its own strides
        v = torch.as_tensor( data ).T
        out = _sum( RealTensor( makers[ "jax" ]( data.T ) ), RealTensor( v ), "foreign_strided" )
        assert numpy.asarray( out.value ).tolist() == ( data.T * 2 ).tolist()

if test( "the_defaults_are_plain_globals" ):
    # `loom.default_xxx` decide what is built from nothing, and are read at each use: assigning is all it takes
    old = ( loom.default_framework, loom.default_device, loom.default_dtype.size, loom.default_itype.size )
    try:
        loom.default_dtype.size = 32
        assert numpy.asarray( RealTensor( [ 1.0, 2.0 ] ).value ).dtype == numpy.float32
        loom.default_dtype.size = 64
        assert numpy.asarray( RealTensor( [ 1.0, 2.0 ] ).value ).dtype == numpy.float64
        loom.default_itype.size = 32
        assert numpy.asarray( loom.IntTensor( [ 1, 2 ] ).value ).dtype == numpy.int32
        for name in _frameworks():
            loom.default_framework = name
            assert RealTensor( [ 1.0 ] ).storage.framework in ( name, "numpy" )     # a host value may stay numpy
            assert str( loom.resolved_framework() ) == name
        # a value that already exists keeps what it has, whatever the defaults say
        loom.default_dtype.size = 32
        assert RealTensor( numpy.zeros( 2 ) ).dtype.size == 64
    finally:
        loom.default_framework, loom.default_device = old[ :2 ]
        loom.default_dtype.size, loom.default_itype.size = old[ 2: ]

if test( "assigning_a_default_checks_it_and_keeps_the_concrete_object" ):
    from loom.drivers.Framework import Framework
    old = loom.default_framework
    try:
        loom.default_framework = "numpy"
        assert isinstance( loom.default_framework, Framework )           # the object, made once, not the string
        assert loom.default_framework is Framework.factory( "numpy" )
        assert loom.default_framework.operations is loom.resolved_framework().operations
        try:
            loom.default_framework = "tensorflwo"                         # a typo fails HERE
            assert False
        except ValueError:
            pass
        assert str( loom.default_framework ) == "numpy"                   # and the old value stands
        try:
            loom.default_device = "tpu"
            assert False
        except ValueError:
            pass
    finally:
        loom.default_framework = old
