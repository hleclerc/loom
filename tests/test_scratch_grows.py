"""A `loom.scratch` is the handler's: taken from the call's allocator, and GROWN IN THE CALL when one of its counts overflows.

The scratch of a call is its own business -- nothing reads it once the call returns. So the generated handler takes its buffers
from the allocator ( `kernels/ScratchBuffer.h` ) rather than Python allocating them as FFI results, and when the body says that a
count did not fit ( `nb_words.set( wanted )` past the capacity ), the handler replaces the buffers by larger ones and runs the body
again -- in the same call. Python is not involved: this holds under a `jit`, where a call cannot be run again from Python.
"""
import loom
from loom import Axis, ShapeVar, RealTensor, IntTensor
from loom.compilation.FfiCode import FfiCode
from loom.util import Aggregate
from errand import test
from loom.testing import need

import numpy


class Work( Aggregate ):
    """a scratch: `nb_words` words, a count the body may ask more of"""
    words    : RealTensor[ "num_word" ]
    num_word : Axis[ "nb_words" ]
    nb_words : ShapeVar


# the body needs `n` words, which only it knows ( an input ): it asks for them when the buffer is too small, and returns -- the
# handler grows the buffer and runs it again
_CODE = """
    const SI n = SI( inputs.n( 0 ) );
    if ( scratch.work.words.shape( 0 ) < n ) {
        scratch.work.nb_words.set( n );
        return;
    }
    for ( SI i = 0; i < n; ++i )
        scratch.work.words( i ) = double( i + 1 );
    double s = 0;
    for ( SI i = 0; i < n; ++i )
        s += scratch.work.words( i );
    outputs.total( 0 ) = s;
"""


def _sum_through_scratch( n, capacity ):
    total = RealTensor[ Axis( ShapeVar( 1 ), name = "num_output" ) ]()
    loom.ffi_call(
        "test_scratch_grows",
        FfiCode.per_item( code = _CODE ),
        n = IntTensor[ Axis( ShapeVar( 1 ), name = "num_n" ) ]( numpy.array( [ n ], dtype = numpy.int32 ) ),
        work = loom.scratch( Work(), capacities = { "nb_words": capacity } ),
        total = loom.out( total ),
        has_dynamic_capacity = False,
    )
    return total


if test( "a_scratch_that_overflows_grows_in_the_call" ):
    need( "cpu" )
    for n, capacity in ( ( 10, 64 ), ( 1000, 4 ), ( 5, 1 ) ):
        total = _sum_through_scratch( n, capacity )
        assert float( numpy.asarray( total.raw )[ 0 ] ) == n * ( n + 1 ) / 2, ( n, capacity, total.raw )


if test( "a_scratch_grows_under_jit" ):
    need( "cpu" )
    import jax
    # the same, compiled: Python cannot run the call again here, the handler does
    def f( x ):
        total = _sum_through_scratch( 777, 3 )
        return total.raw[ 0 ] + 0 * x
    res = jax.jit( f )( 1.0 )
    assert float( res ) == 777 * 778 / 2, res
