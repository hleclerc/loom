import sys
from errand import has_tag

# This file is only for the numpy driver. errand imports every candidate file to find the entries,
# so without this exit the `setdefault` below would pin the framework to numpy for the whole run.
if not has_tag( "driver=numpy" ):
    sys.exit( 0 )

import os
os.environ.setdefault( "LOOM_FRAMEWORK", "numpy" )

import numpy
import loom
import loom
from loom import CtShapeVar, ShapeVar, Axis, Aggregate, RealTensor
from loom.compilation.FfiCode import FfiCode
from errand import test

# The numpy driver: the BEFORE driver, for a machine that has neither jax nor torch. The kernels
# are the same as under jax, compiled against `loom/support/numpy_ffi` ( see `drivers/NumpyFfi.py` )
# and launched on the CPU queue; there is neither tracing nor differentiation.

if test( "driver" ):
    assert loom.resolved_framework() == "numpy"
    assert loom.resolved_device().is_cpu
    assert loom.resolved_dtype().cpp_name == "FP64"

    a = loom.array( [ 1, 2, 3 ] )
    assert isinstance( a, numpy.ndarray ) and a.dtype == numpy.float64
    assert not loom.is_traced( a )

    # no tape: whatever needs one says so, instead of returning a wrong value
    for verb in ( loom.vmap, loom.grad ):
        try:
            verb( lambda x: x )
            assert False, "expected NotImplementedError"
        except NotImplementedError:
            pass
    assert loom.jit( lambda x: x + 1 )( 1 ) == 2


if test( "call" ):
    class Cell1N( Aggregate ):
        vertex_positions : RealTensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

    cell = Cell1N( nb_dims = 2 )

    loom.ffi_call(
        "test_numpy_driver_call",
        FfiCode.per_item( code = """
        outputs.cell.nb_vertices( batch_index ).set( 1 );
        outputs.cell.vertex_positions( batch_index, dim = 0, num_vertex = 0 ) = 1;
        outputs.cell.vertex_positions( batch_index, dim = 1, num_vertex = 0 ) = 2;
        """ ),
        cell = loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ), capacities = { "nb_vertices": 8 } ),
    )

    assert cell.nb_vertices.value == 1
    assert cell.vertex_positions.shape == [ 1, 2 ]
    assert cell.vertex_positions.capacity == ( 8, 2 )
    assert cell.vertex_positions.raw.tolist()[ 0 ] == [ 1, 2 ]
    assert isinstance( cell.vertex_positions.raw, numpy.ndarray )
