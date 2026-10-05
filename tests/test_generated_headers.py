"""Generated headers (`compilation/generated_headers.py`) must not race between calls.

An aggregate's generated header is NAMED after its type only (`sdot/generated/aggregates/<T>.h`),
while its CONTENT depends on the call (which members, which io policy). Several processes sharing
one build directory (errand runs several jobs on one remote tree) used to overwrite each other's
version between the write and the compile -- and the per-process `_written` cache then believed
the file still held its own content, and skipped rewriting it: a sporadic `ninja failed`, with
`no instance of constructor ...`.

Each compiled kernel now reads the generated headers it rendered from its OWN include overlay
(inside its content-hashed directory, first on the `-I` path); the shared tree is only written for
editors and hand-written helpers. These tests play the other process by hand: they clobber the
shared tree between two calls, which is exactly what the race did.
"""
import subprocess
from pathlib import Path

import loom
from loom import ShapeVar, Axis, Aggregate, RealTensor
from loom.compilation.FfiCode import FfiCode
from loom.compilation.build import kernels_root, ninja_path
from loom.compilation.generated_headers import include_root
from errand import test


def _variant( with_y ):
    """Two DIFFERENT aggregates under the SAME C++ name: their generated headers collide."""
    if with_y:
        class RaceAgg( Aggregate ):
            x   : RealTensor[ "num" ]
            y   : RealTensor[ "num" ]
            num : Axis[ "n" ]
            n   : ShapeVar
    else:
        class RaceAgg( Aggregate ):
            x   : RealTensor[ "num" ]
            num : Axis[ "n" ]
            n   : ShapeVar
    return RaceAgg


def _call( name, with_y, value ):
    agg = _variant( with_y )()
    code = ( "outputs.agg.n( batch_index ).set( 1 );\n"
             f"outputs.agg.x( batch_index, num = 0 ) = { value };\n" )
    writes = ( "n", "x" )
    if with_y:
        code += f"outputs.agg.y( batch_index, num = 0 ) = { value + 1 };\n"
        writes += ( "y", )
    loom.ffi_call( name, FfiCode.per_item( code = code ),
                   agg = loom.out( agg, writes = writes, capacities = { "n": 2 } ) )
    assert agg.n.value == 1
    assert float( agg.x.raw[ 0 ] ) == value
    if with_y:
        assert float( agg.y.raw[ 0 ] ) == value + 1


def _clobber_shared():
    """What another process sharing the build directory does: its own variant lands in the shared
    tree (here, something that cannot compile at all, so the test cannot pass by luck)."""
    for p in ( include_root() / "sdot" / "generated" / "aggregates" ).glob( "RaceAgg*.h" ):
        p.write_text( "#error clobbered by another process\n" )


if test( "race" ):
    # this process renders and compiles variant A ...
    _call( "test_generated_headers_race_a", False, 3 )
    _call( "test_generated_headers_race_b", True, 5 )
    # ... another one overwrites the shared headers between our write and our compile (or simply
    # between two of our calls: our `_written` cache still believes the files hold OUR content) ...
    _clobber_shared()
    # ... and a NEW kernel of ours, naming the same aggregate, must still compile and run.
    _call( "test_generated_headers_race_c", False, 7 )
    _clobber_shared()
    _call( "test_generated_headers_race_d", True, 9 )
    # already-compiled kernels too: a clobbered shared header must not make them rebuild (or fail)
    _clobber_shared()
    _call( "test_generated_headers_race_a", False, 3 )


if test( "depfiles" ):
    # the compiled kernel must depend on ITS overlay, never on the shared tree: otherwise any write
    # there (another variant, another process) rebuilds it -- churn, at best.
    _call( "test_generated_headers_deps", True, 11 )
    shared = str( include_root() )
    # (a directory left by a failed compilation has no object, hence no deps: not ours to judge)
    dirs = [ d for d in kernels_root().iterdir()
             if "test_generated_headers_deps_" in d.name and any( d.glob( "*.o" ) ) ]
    assert dirs, "no kernel directory found"
    for d in dirs:
        deps = subprocess.run( [ ninja_path(), "-C", str( d ), "-t", "deps" ],
                               stdout = subprocess.PIPE, text = True, check = True ).stdout
        assert "RaceAgg" in deps, deps
        bad = [ l.strip() for l in deps.splitlines() if l.strip().startswith( shared ) ]
        assert not bad, f"{ d.name } depends on the shared tree: { bad }"
