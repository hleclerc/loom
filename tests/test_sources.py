"""MULTI-FILE kernels: `FfiCode.per_item( sources = [ ( "x.cpp", { "DEF": ... } ) ] )`.

A domain source is compiled once per (source, defines, compiler) into a `.o` that
all the kernels naming it link against -- the code you do not want to reinstantiate in each
generated unit (a density, a dimension: macros choose). This test checks that the
defines do make DISTINCT units, and that the graph (ninja) reuses them.

IT IS COMPILED AS HOST CODE, and that is what this test must exercise. A separate unit is not
device code: calling it from an `HD` body would require CUDA's separate device compilation
(`-rdc=true` on both sides, then `-dlink`), which loom does not do -- and which nobody asks of it.
The only real user of this facility, `sdot/sdotplan/Linear.cpp`, calls it from the
HANDLER, which is host code.

This test used to do it from a `per_item` body, hence from the device: it compiled on CPU and
failed on CUDA ("calling a __host__ function from a __host__ __device__ function"), which
was not a defect of loom but of the test. It is now written on the real usage: `scaled` is called
in `kernel`, and the functor carries the result away by CAPTURE.
"""
from pathlib import Path
import numpy

import loom
from loom import driver
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, IntTensor, ShapeVar
from errand import test

HERE = Path( __file__ ).resolve().parent / "cpp_sources"


_CODE = """
    namespace {
        /// what the HOST computed, carried by value to the device
        struct Setter {
            SI v[ 3 ];
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.res( coords ) = v[ coords[ num ] ];
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            // `scaled` lives in a SEPARATELY compiled unit ( `sources = ...` ), as HOST code.
            // `kernel` IS host code: this is where we call it, exactly as
            // `OtPlan` calls `sdotplan::solve`.
            queue.run_parallel( Setter{ { sdot::scaled( 1 ), sdot::scaled( 2 ), sdot::scaled( 3 ) } },
                                args.outputs.res.domain(), args, batch_axes );
        }
    }
"""


def _scaled( scale ):
    num = Axis( ShapeVar( 3 ), name = "num" )
    res = IntTensor[ num ]()
    loom.ffi_call(
        f"test_sources_scaled_{ scale }",
        FfiCode( code = _CODE,
            includes = [ str( HERE / "scaled.h" ) ],
            sources = [ ( str( HERE / "scaled.cpp" ), { "SCALE": str( scale ) } ) ] ),
        res = loom.out( res ),
    )
    return numpy.asarray( res.value ).reshape( -1 ).tolist()


if test( "a_source_compiled_with_a_define_is_linked_in" ):
    assert _scaled( 2 ) == [ 2, 4, 6 ]


if test( "another_define_is_another_unit" ):
    assert _scaled( 3 ) == [ 3, 6, 9 ]
    assert _scaled( 2 ) == [ 2, 4, 6 ]
