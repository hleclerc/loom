import os
import sys
from errand import has_tag

# A call of a form already analysed is REPLAYED ( `drivers/CallPlan.py` ): the pointer drivers -- Torch, Numpy, Cupy.
if not ( has_tag( "driver=torch" ) or has_tag( "driver=numpy" ) or has_tag( "driver=cupy" ) ):
    sys.exit( 0 )

# every replay is analysed again and compared with its plan: what these tests check, beyond the values
os.environ[ "LOOM_VERIFY_CALL_PLAN" ] = "1"

import numpy
import loom
from loom import Axis, IntTensor, RealTensor, ShapeVar
from loom.compilation.FfiCode import FfiCode
from loom.drivers import CallPlan
from errand import test

SHIFT = FfiCode( code = """
    namespace {
        struct AddShift {
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.res( coords ) = args.inputs.x( coords ) + args.inputs.shift( 0 );
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( AddShift(), args.outputs.res.domain(), args, batch_axes );
        }
    }
""" )


def _framework():
    return str( loom.resolved_framework() )


def _reals( values ):
    """`values` as an array of the framework, float64, on the call's device."""
    return loom.array( numpy.asarray( values, dtype = numpy.float64 ) )


def _ints( values ):
    """`values` as an int64 array of the framework ( host: the bodies that read it are host code )."""
    values = numpy.asarray( values, dtype = numpy.int64 )
    if _framework() == "torch":
        import torch
        return torch.as_tensor( values )
    return values


def _dense( x ):
    if _framework() == "torch":
        return x.contiguous()
    return __import__( type( x ).__module__.split( "." )[ 0 ] ).ascontiguousarray( x )


def _shift( x, shift, name = "test_call_plan_shift", **kwargs ):
    res = RealTensor.like( x )
    loom.ffi_call( name, SHIFT, x = x, shift = shift, res = loom.out( res ), has_dynamic_capacity = False, **kwargs )
    return res.value


def _plans_of( name ):
    return [ p for k, p in CallPlan._plans.items() if k[ 0 ] == name ]


def _arange( *shape ):
    return _reals( numpy.arange( numpy.prod( shape ) ).reshape( shape ) )


if test( "a_call_of_a_known_form_is_replayed" ):
    shift = _reals( [ 10.0 ] )
    first = _shift( _arange( 2, 3 ), shift )
    assert first.tolist() == [ [ 10, 11, 12 ], [ 13, 14, 15 ] ]
    plans = _plans_of( "test_call_plan_shift" )
    assert len( plans ) == 1 and plans[ 0 ], "the call was not kept"
    # other values, same form: the plan, and what it writes goes to the tensor of THIS call
    second = _shift( -_arange( 2, 3 ), _reals( [ 1.0 ] ) )
    assert second.tolist() == [ [ 1, 0, -1 ], [ -2, -3, -4 ] ]
    assert first.tolist() == [ [ 10, 11, 12 ], [ 13, 14, 15 ] ], "a replay wrote into the outputs of another call"
    assert len( _plans_of( "test_call_plan_shift" ) ) == 1


if test( "another_shape_or_layout_is_another_form" ):
    shift = _reals( [ 1.0 ] )
    name = "test_call_plan_forms"
    assert _shift( _arange( 2, 3 ), shift, name ).tolist() == [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
    assert _shift( _arange( 3, 2 ), shift, name ).tolist() == [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
    # a transposed view: the kernel reads its strides, and a dense input of the same shape is another form
    for _ in range( 2 ):
        assert _shift( _arange( 2, 3 ).T, shift, name ).tolist() == [ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ]
        assert _shift( _dense( _arange( 2, 3 ).T ), shift, name ).tolist() == [ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ]
    assert len( _plans_of( name ) ) == 3      # [ 2, 3 ], [ 3, 2 ] ( the dense copy of the view too ), the view


if test( "a_value_to_convert_is_analysed_every_time" ):
    # a python float becomes a tensor of the call's own: not the value given, so no plan
    name = "test_call_plan_scalar"
    for _ in range( 2 ):
        res = RealTensor[ Axis( ShapeVar( 3 ), name = "res_0" ), dict( size = 64 ) ]()
        loom.ffi_call( name, FfiCode.inline( "args.outputs.res.fill_with( queue, 2.0 );" ), x = 1.0, res = loom.out( res ),
                       has_dynamic_capacity = False )
        assert res.value.tolist() == [ 2, 2, 2 ]
    assert not any( _plans_of( name ) )


if test( "a_failure_is_reported_on_a_replay" ):
    from loom.testing import need
    need( "cpu" )       # the body is host code: it records into the error buffer where it runs
    from loom.drivers.CallArg_Errors import KernelFailure
    name = "test_call_plan_failure"
    code = FfiCode.inline( "if ( SI( args.inputs.wanted( 0 ) ) ) args.errors.record( SI( args.inputs.wanted( 0 ) ), 7, 42 );\n"
                           "    args.outputs.res.fill_with( queue, 1.0 );" )

    def run( wanted ):
        res = RealTensor[ Axis( ShapeVar( 4 ), name = "num_fail" ), dict( size = 64 ) ]()
        loom.ffi_call( name, code, wanted = _ints( [ wanted ] ), res = loom.out( res ),
                       failures = { 7: "the seed {value} is past the limit" } )
        return res.value

    assert run( 0 ).tolist() == [ 1 ] * 4
    assert run( 0 ).tolist() == [ 1 ] * 4
    assert len( _plans_of( name ) ) == 1 and _plans_of( name )[ 0 ]
    try:
        run( 2 )
        raise AssertionError( "a failure went through a replay" )
    except KernelFailure as e:
        assert str( e ) == f"{ name }: the seed 42 is past the limit", str( e )


if test( "without_a_dynamic_capacity_the_error_buffer_is_not_read" ):
    # `has_dynamic_capacity = False`, no failure: nothing to act on, so no read back ( on a card, a synchronization per call )
    from loom.tensor import host
    reads, original = [], host.to_host
    host.to_host = lambda x: ( reads.append( x ), original( x ) )[ 1 ]
    try:
        shift = _reals( [ 1.0 ] )
        for _ in range( 3 ):
            _shift( _arange( 2, 2 ), shift, "test_call_plan_no_read" )
    finally:
        host.to_host = original
    assert not reads, f"{ len( reads ) } reads of the error buffer"


if test( "a_call_with_a_backward_is_replayed_where_no_backward_runs" ):
    # under Torch the call is an autograd function, which the analysis makes: not replayed. Numpy and cupy run the forward
    # alone, so the backward changes nothing to the call
    name = "test_call_plan_backward"
    fwd = FfiCode.inline( "args.outputs.res.fill_with( queue, 1.0 );" )
    bwd = FfiCode.inline( "" )
    for _ in range( 2 ):
        res = RealTensor[ Axis( ShapeVar( 2 ), name = "res_0" ), dict( size = 64 ) ]()
        loom.ffi_call( name, fwd, bwd, x = _arange( 2 ), res = loom.out( res ), has_dynamic_capacity = False )
        assert res.value.tolist() == [ 1, 1 ]
    plans = _plans_of( name )
    assert len( plans ) == 1 and bool( plans[ 0 ] ) == ( _framework() != "torch" ), plans
