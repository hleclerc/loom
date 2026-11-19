"""What loom says of its build cache, for whoever keeps it tidy -- errand ( its ADR 0020 ), or loom itself.

The format is errand's ( `errand-cache/1` ), written here without importing errand: each process appends lines to its
OWN journal, `<cache>/errand-cache-journal/<host>-<pid>-<start>.jsonl`, and never touches anything else. A compactor
( errand's agent side under errand, loom itself otherwise, past `SELF_COMPACT` journals ) merges them into
`<cache>/errand-cache.json`; errand then removes what no longer serves. What loom says:

* an ENTRY per kernel, `build/kernels/<name>`, written when ninja actually built it ( new, or rebuilt in place after a
  header changed: the entry is written again, with what it depends on now ). Its `deps`: the closure of headers ninja
  knows, and the `.py` that built its `FfiCode` ( `FfiCode.origin`, and what `depends_on =` adds ) -- a kernel whose code
  changed in Python has another name, and the old one is only found dead through that file. Its `needs`: the shared
  libraries it links.
* a SHARED entry per domain unit and runtime ( `units/…`, `runtime/…`, with their objects ), which goes once no kernel
  left needs it.
* the USES: the kernels this process loaded, once, when it ends.
* the VIEW of each entry: where its `deps` are seen from, as errand says it ( `$ERRAND_CACHES_VIEW`, its ADR 0022: empty on
  a machine as it is, `apptainer:<image>` in a container ), so that a purge judges only what it sees as the build saw it.

Nothing is written when the build directory is not under the cache root ( an explicit `LOOM_BUILD_DIR` ): nobody keeps
it. The objects of a kernel are not declared as intermediates: the fast path would rebuild a kernel whose `.o` is gone.
"""
import atexit
import contextlib
import contextvars
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

FORMAT   = "errand-cache/1"
JOURNALS = "errand-cache-journal"
MANIFEST = "errand-cache.json"
LOCK     = "errand-cache.lock"
SELF_COMPACT = 64            # journals past which loom compacts by itself, when errand does not

_declared = contextvars.ContextVar( "loom_kernel_python", default = () )
_used = set()
_hashes = {}                 # ( path, size, mtime_ns ) -> sha256: a header is hashed once per process
_root = [ False ]            # the cache root, once known ( None: no journal )
_started = time.time()


def _now( t = None ) -> str:
    return datetime.fromtimestamp( time.time() if t is None else t, timezone.utc ).strftime( "%Y-%m-%dT%H:%M:%SZ" )


def root():
    """The cache this process journals into: `cache_root()`, when the build directory is under it."""
    if _root[ 0 ] is False:
        from . import build_dir, cache_root
        try:
            r, b = cache_root().resolve(), build_dir().resolve()
            _root[ 0 ] = r if b.is_relative_to( r ) else None
        except ( OSError, RuntimeError ):
            _root[ 0 ] = None
        if _root[ 0 ] is not None:
            atexit.register( _at_exit )
    return _root[ 0 ]


@contextlib.contextmanager
def declaring( files ):
    """The Python files the kernels compiled inside the block depend on ( `Kernels.python_files` )."""
    token = _declared.set( tuple( files or () ) )
    try:
        yield
    finally:
        _declared.reset( token )


def _rel( path ):
    r = root()
    try:
        return Path( path ).resolve().relative_to( r ).as_posix()
    except ValueError:
        return None


def _dep( path ):
    try:
        st = os.stat( path )
    except OSError:
        return None
    k = ( str( path ), st.st_size, st.st_mtime_ns )
    if k not in _hashes:
        h = hashlib.sha256()
        try:
            with open( path, "rb" ) as f:
                for chunk in iter( lambda: f.read( 1 << 20 ), b"" ):
                    h.update( chunk )
        except OSError:
            return None
        _hashes[ k ] = h.hexdigest()
    return [ _hashes[ k ], st.st_size, st.st_mtime_ns ]


def _size( path ) -> int:
    p = Path( path )
    if p.is_dir():
        return sum( f.stat().st_size for f in p.rglob( "*" ) if f.is_file() and not f.is_symlink() )
    return p.stat().st_size if p.exists() else 0


def _journal() -> Path:
    return root() / JOURNALS / f"{ socket.gethostname() }-{ os.getpid() }-{ int( _started ) }.jsonl"


def _write( line: dict ):
    """One line, appended to this process's own journal: open, write, close -- a compactor may have taken the file
    meanwhile ( a rename ), and the line then starts a new one."""
    path = _journal()
    try:
        path.parent.mkdir( parents = True, exist_ok = True )
        with open( path, "a" ) as f:
            f.write( json.dumps( line ) + "\n" )
    except OSError:
        pass


def record_create( entry, products, needs = (), deps = (), cost_s = 0.0, shared = False, size_of = None ):
    """`entry` ( a path under the cache ) was built: what it is made of, needs, depends on, cost. Paths are absolute;
    `size_of`: where its bytes are now, when it is not published yet ( a kernel's staging directory )."""
    if root() is None:
        return
    key = _rel( entry )
    if key is None:
        return
    products = [ p for p in ( _rel( p ) for p in products ) if p is not None ]
    # the Python a kernel is generated from is its own: a shared library ( the runtime, a domain unit ) is the same whichever
    # kernel had it built
    deps = { str( d ): rec for d in { *deps, *( () if shared else _declared.get() ) } for rec in [ _dep( d ) ] if rec is not None }
    _write( { "op": "create", "entry": key, "product": products, "intermediate": [],
              "needs": [ n for n in ( _rel( n ) for n in needs ) if n is not None ], "shared": bool( shared ),
              "size": _size( size_of ) if size_of is not None else sum( _size( root() / p ) for p in products ),
              "cost_s": round( float( cost_s ), 3 ),
              "deps": deps, "view": os.environ.get( "ERRAND_CACHES_VIEW", "" ), "at": _now() } )


def record_use( entry ):
    """`entry` served this process; said once, when it ends."""
    if root() is not None:
        _used.add( entry )


def _at_exit():
    if _used:
        keys = sorted( k for k in ( _rel( e ) for e in _used ) if k is not None )
        if keys:
            _write( { "op": "use", "entries": keys, "at": _now() } )
    if not os.environ.get( "ERRAND_CACHES" ):
        with contextlib.suppress( Exception ):
            compact_if_many()


def _read_journal( path, start ):
    """The whole lines of a journal from byte `start` -> ( lines, where the next read starts ): a line still being
    written waits for the next read."""
    with open( path, "rb" ) as f:
        f.seek( start )
        data = f.read()
    end = data.rfind( b"\n" ) + 1
    out = []
    for raw in data[ : end ].splitlines():
        with contextlib.suppress( ValueError ):
            out.append( json.loads( raw ) )
    return out, start + end


def compact_if_many( limit = SELF_COMPACT ):
    """Without errand nobody compacts: past `limit` journals with lines not merged yet, loom merges them itself ( the
    format's rules, no purge ). As errand does ( its ADR 0022 ): each journal read from where the last compaction
    stopped ( `journals` in the manifest ), never renamed under a writer; removed once read and a day old."""
    r = root()
    jd = r / JOURNALS
    try:
        names = sorted( n for n in os.listdir( jd ) if n.endswith( ".jsonl" ) or ".jsonl.taken-" in n )
    except OSError:
        return
    if len( names ) <= limit:
        return
    def manifest():
        try:
            m = json.loads( ( r / MANIFEST ).read_text() )
            assert m.get( "format" ) == FORMAT
            return m
        except ( OSError, ValueError, AssertionError ):
            return { "format": FORMAT, "hash": "sha256", "purged": None, "entries": {} }
    def unread( m ):
        state, n = m.get( "journals", {} ), 0
        for name in names:
            with contextlib.suppress( OSError ):
                st = ( jd / name ).stat()
                ino, read = state.get( name, [ None, 0 ] )
                n += st.st_size > ( read if ino == st.st_ino else 0 )
        return n
    if unread( manifest() ) <= limit:
        return
    lock = r / LOCK
    try:
        lock.mkdir()
    except OSError:
        return                                   # another one is at it
    try:
        m = manifest()
        state, lines, done, now = m.setdefault( "journals", {} ), [], [], time.time()
        for n in names:
            p = jd / n
            try:
                if n.endswith( ".jsonl" ):
                    st = p.stat()
                    ino, start = state.get( n, [ None, 0 ] )
                    if ino != st.st_ino or start > st.st_size:
                        start = 0
                    if st.st_size > start:
                        got, start = _read_journal( p, start )
                        lines += got
                    state[ n ] = [ st.st_ino, start ]
                    if start >= st.st_size and now - st.st_mtime > 86400:
                        done.append( p )
                else:                            # taken by an older compactor that died
                    lines += _read_journal( p, 0 )[ 0 ]
                    done.append( p )
            except OSError:
                continue
        entries = m.setdefault( "entries", {} )
        for l in sorted( lines, key = lambda l: l.get( "at", "" ) ):
            at = l.get( "at", "" )
            if l.get( "op" ) == "create" and l.get( "entry" ):
                old = entries.get( l[ "entry" ], {} )
                if old.get( "created", "" ) > at:
                    continue                     # a later build of it was merged already
                entries[ l[ "entry" ] ] = { **{ k: l[ k ] for k in ( "product", "intermediate", "needs", "shared", "size",
                                                                      "cost_s", "deps", "view" ) if k in l },
                                            "created": at, "used": max( at, old.get( "used", "" ) ) }
            elif l.get( "op" ) == "use":
                for k in l.get( "entries", [] ):
                    if k in entries:
                        entries[ k ][ "used" ] = max( at, entries[ k ].get( "used", "" ) )
        present = set( names ) - { p.name for p in done }
        for n in list( state ):
            if n not in present:
                del state[ n ]
        part = r / f".{ MANIFEST }.{ os.getpid() }.part"
        part.write_text( json.dumps( m, indent = 1, sort_keys = True ) )
        os.replace( part, r / MANIFEST )
        for t in done:
            with contextlib.suppress( OSError ):
                t.unlink()
    finally:
        with contextlib.suppress( OSError ):
            for f in lock.iterdir():
                f.unlink()
            lock.rmdir()
