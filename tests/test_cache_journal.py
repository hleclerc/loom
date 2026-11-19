"""What loom says of its build cache ( `compilation/cache_journal.py`, errand's format `errand-cache/1` ): a kernel built
is an entry, with the headers and the `.py` it depends on and the shared libraries it needs; a kernel served is a use."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from errand import test

WORKER = Path( __file__ ).with_name( "cache_journal_worker.py" ).resolve()


def run( cache, *args, **env ):
    e = { k: v for k, v in os.environ.items() if k not in ( "LOOM_BUILD_DIR", "ERRAND_CACHES" ) }
    e.update( LOOM_CACHE_DIR = str( cache ), LOOM_FRAMEWORK = "numpy", **env )
    r = subprocess.run( [ sys.executable, str( WORKER ), *args ], env = e, capture_output = True, text = True )
    assert r.returncode == 0, r.stderr[ -3000: ]


def lines( cache ):
    out = []
    for j in sorted( ( cache / "errand-cache-journal" ).glob( "*.jsonl" ) ):
        out += [ json.loads( l ) for l in j.read_text().splitlines() ]
    return out


if test( "a kernel built is an entry, with its headers, its .py and what it needs; served again, only a use" ):
    cache = Path( tempfile.mkdtemp( prefix = "loom-journal-" ) ).resolve()
    run( cache )
    got = lines( cache )
    kernels = [ l for l in got if l[ "op" ] == "create" and not l[ "shared" ] ]
    shared = { l[ "entry" ] for l in got if l[ "op" ] == "create" and l[ "shared" ] }
    assert len( kernels ) == 1, got
    k = kernels[ 0 ]
    assert k[ "entry" ].startswith( "build/kernels/" ) and k[ "product" ] == [ k[ "entry" ] ] and ( cache / k[ "entry" ] ).is_dir()
    assert str( WORKER ) in k[ "deps" ], k[ "deps" ].keys()
    assert any( d.endswith( ".h" ) for d in k[ "deps" ] ), k[ "deps" ].keys()
    assert k[ "needs" ] and set( k[ "needs" ] ) <= shared, ( k[ "needs" ], shared )
    assert k[ "size" ] > 0 and k[ "cost_s" ] > 0 and k[ "view" ] == ""
    assert not any( str( WORKER ) in l[ "deps" ] for l in got if l[ "op" ] == "create" and l[ "shared" ] ), got
    assert [ l[ "entries" ] for l in got if l[ "op" ] == "use" ] == [ [ k[ "entry" ] ] ], got
    assert ( cache / "CACHEDIR.TAG" ).is_file()
    run( cache )
    again = lines( cache )[ len( got ): ]
    assert [ l[ "op" ] for l in again ] == [ "use" ], again


if test( "without errand, loom compacts its journals itself past a limit; under errand it leaves them" ):
    cache = Path( tempfile.mkdtemp( prefix = "loom-journal-" ) ).resolve()
    run( cache )
    jd = cache / "errand-cache-journal"
    for i in range( 70 ):
        ( jd / f"other-{ i }-0.jsonl" ).write_text( json.dumps( { "op": "use", "entries": [ "x" ], "at": "2026-01-01T00:00:00Z" } ) + "\n" )
    run( cache, ERRAND_CACHES = str( cache.parent ) )
    assert len( list( jd.glob( "*.jsonl" ) ) ) > 64
    run( cache )
    m = json.loads( ( cache / "errand-cache.json" ).read_text() )
    assert m[ "format" ] == "errand-cache/1" and any( e.startswith( "build/kernels/" ) for e in m[ "entries" ] ), m
    m = json.loads( ( cache / "errand-cache.json" ).read_text() )
    assert all( ( jd / n ).stat().st_size == read for n, ( _, read ) in m[ "journals" ].items() if ( jd / n ).exists() ), m[ "journals" ]
    before = ( cache / "errand-cache.json" ).read_text()
    run( cache )                                         # merged already: nothing to compact for one more journal
    assert json.loads( ( cache / "errand-cache.json" ).read_text() )[ "journals" ] == json.loads( before )[ "journals" ]


if test( "each entry says the view errand gave the run: where its deps are seen from ( errand's ADR 0022 )" ):
    cache = Path( tempfile.mkdtemp( prefix = "loom-journal-" ) ).resolve()
    run( cache, ERRAND_CACHES_VIEW = "apptainer:img.sif" )
    creates = [ l for l in lines( cache ) if l[ "op" ] == "create" ]
    assert creates and { l[ "view" ] for l in creates } == { "apptainer:img.sif" }, creates
    run( cache )                                         # compacted by loom itself: the view kept in the manifest
    for i in range( 70 ):
        ( cache / "errand-cache-journal" / f"other-{ i }-0.jsonl" ).write_text( json.dumps( { "op": "use", "entries": [], "at": "2026-01-01T00:00:00Z" } ) + "\n" )
    run( cache )
    m = json.loads( ( cache / "errand-cache.json" ).read_text() )
    assert { e.get( "view" ) for e in m[ "entries" ].values() } == { "apptainer:img.sif" }, m
