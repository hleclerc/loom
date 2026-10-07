"""The RENDER CACHE (`drivers/render_key.py`): a call that repeats an earlier one is not rendered
again -- and what makes that safe is what these tests pin down.

  * a repeated call hits; a call that differs in what the SOURCE spells (a shape, the environment)
    misses -- never a hit with the wrong text;
  * what crosses as an FFI ATTRIBUTE (a bare `int`) is NOT in the key, so a new value still hits, and the
    result follows the value: the attributes are recomputed from the current call, never remembered;
  * a call the key cannot describe (a field that is not a plain fact) is rendered as before;
  * `LOOM_VERIFY_RENDER_CACHE=1` renders anyway on every hit and fails on any difference: it is on for
    every call below, and it is what to run over the whole suite after touching the lowering.
"""
import os

import numpy

import loom
from loom.testing import host
from loom import Axis, IntTensor, ShapeVar
from loom.compilation.FfiCode import FfiCode
from loom.drivers import render_key
from errand import test


_CODE = """
    namespace {
        struct Setter {
            SI k;
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.res( coords ) = k * ( coords[ num ] + 1 );
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( Setter{ args.inputs.k }, args.outputs.res.domain(), args, batch_axes );
        }
    }
"""


def _call( k, n = 3, code = None ):
    num = Axis( ShapeVar( n ), name = "num" )
    res = IntTensor[ num ]()
    loom.ffi_call( "test_render_cache_scale", code or FfiCode( code = _CODE ), k = k, res = loom.out( res ) )
    return host( res.value ).reshape( -1 ).tolist()


class _Env:
    """Sets environment variables for the duration of a block."""
    def __init__( self, **values ):
        self.values = values

    def __enter__( self ):
        self.saved = { k: os.environ.get( k ) for k in self.values }
        os.environ.update( self.values )

    def __exit__( self, *exc ):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop( k, None )
            else:
                os.environ[ k ] = v


def _counts():
    return render_key.stats[ "hit" ], render_key.stats[ "miss" ], sum( render_key.stats[ "uncacheable" ].values() )


if test( "a_repeated_call_is_not_rendered_again" ):
    with _Env( LOOM_VERIFY_RENDER_CACHE = "1" ):
        assert _call( 3 ) == [ 3, 6, 9 ]
        hit, miss, _ = _counts()
        assert _call( 3 ) == [ 3, 6, 9 ]
        assert _counts()[ :2 ] == ( hit + 1, miss )


if test( "an_attribute_value_is_not_part_of_the_key_and_the_result_follows_it" ):
    with _Env( LOOM_VERIFY_RENDER_CACHE = "1" ):
        _call( 3 )
        hit, miss, _ = _counts()
        assert _call( 4 ) == [ 4, 8, 12 ]
        assert _call( 5 ) == [ 5, 10, 15 ]
        assert _counts()[ :2 ] == ( hit + 2, miss )


if test( "another_shape_is_another_key" ):
    with _Env( LOOM_VERIFY_RENDER_CACHE = "1" ):
        _call( 3 )
        hit, miss, _ = _counts()
        assert _call( 3, n = 5 ) == [ 3, 6, 9, 12, 15 ]
        assert _counts()[ :2 ] == ( hit, miss + 1 )


if test( "the_environment_is_part_of_the_key" ):
    with _Env( LOOM_VERIFY_RENDER_CACHE = "1" ):
        _call( 3 )
        hit, miss, _ = _counts()
        with _Env( LOOM_ZERO_OUTPUTS = "poison" ):
            assert _call( 3 ) == [ 3, 6, 9 ]
        assert _counts()[ :2 ] == ( hit, miss + 1 )


if test( "a_call_the_key_cannot_describe_is_rendered_as_before" ):
    with _Env( LOOM_VERIFY_RENDER_CACHE = "1" ):
        code = FfiCode( code = _CODE )
        code.opaque = object()              # not a plain fact: the key must refuse
        hit, miss, unc = _counts()
        assert _call( 3, code = code ) == [ 3, 6, 9 ]
        assert _call( 3, code = code ) == [ 3, 6, 9 ]
        assert _counts() == ( hit, miss, unc + 2 )


if test( "the_cache_can_be_switched_off" ):
    with _Env( LOOM_RENDER_CACHE = "0" ):
        before = _counts()
        assert _call( 3 ) == [ 3, 6, 9 ]
        assert _counts() == before
