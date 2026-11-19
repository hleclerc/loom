"""Run by test_cache_journal.py in a process of its own ( LOOM_CACHE_DIR set by the test ): one kernel, compiled."""
import os, sys
os.environ.setdefault( "LOOM_FRAMEWORK", "numpy" )
import loom
from loom import CtShapeVar, ShapeVar, Axis, Aggregate, RealTensor
from loom.compilation.FfiCode import FfiCode


class CacheJournalCell( Aggregate ):
    vertex_positions : RealTensor[ "num_vertex", "dim" ]
    num_vertex       : Axis[ "nb_vertices" ]
    dim              : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_dims          : CtShapeVar


cell = CacheJournalCell( nb_dims = 2 )
value = sys.argv[ 1 ] if len( sys.argv ) > 1 else "1"
loom.ffi_call(
    "test_cache_journal",
    FfiCode.per_item( code = f"""
    outputs.cell.nb_vertices( batch_index ).set( 1 );
    outputs.cell.vertex_positions( batch_index, dim = 0, num_vertex = 0 ) = { value };
    outputs.cell.vertex_positions( batch_index, dim = 1, num_vertex = 0 ) = 2;
    """ ),
    cell = loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ), capacities = { "nb_vertices": 8 } ),
)
assert cell.vertex_positions.raw.tolist()[ 0 ] == [ float( value ), 2 ]
