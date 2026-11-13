"""EXTERNAL C++ libraries, header-only, that the build chain fetches by itself.

A package built on loom DECLARES one (`register_external`): a name, a pinned archive and its
SHA-256, never a branch. A kernel that includes it ASKS for it by name:

    FfiCode.inline( "...", sources = [ "my/solver.cpp" ], externals = [ "eigen" ] )

and only then is the archive downloaded, ONCE, into the user cache (`cache_root() / "ext"`), and
its directory put on the `-I` of that kernel and of the units it links -- of nothing else. A kernel
that does not ask never triggers a download and never sees the library.

The request is written into the kernel's source (`externals_marker`, read back by `externals_of`):
it is part of the kernel's name, of its catalogue key and of the render cache with no extra key,
and the compile step reads it from the very text it compiles.

THE CHECK IS MADE ONCE PER PROCESS (`include_dir_of`): the directory found, or the failure. Without
network (`LOOM_EXTERNALS=0`, or a download that fails), the external is missing and we say so ONCE:
the source that expects it guards itself with `__has_include` and makes do with what it has.
`LOOM_EXT_DIR` puts the cache elsewhere (a build machine, a shared mount).
"""

from pathlib import Path
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from .. import env


class External:
    def __init__( self, name, version, url, sha256, include = "" ):
        self.name, self.version, self.url, self.sha256, self.include = name, str( version ), url, sha256, include

    @property
    def root( self ) -> Path:
        return ext_root() / f"{ self.name }-{ self.version }"

    @property
    def include_dir( self ) -> Path:
        return self.root / self.include if self.include else self.root


_externals = {}
_resolved = {}          # ( name, ext_root ) -> the include directory, or None: the check, made once per process


def register_external( name, version, url, sha256, include = "" ):
    """Declares a library: `url` of an archive (`.tar.gz` / `.zip`) whose top level is stripped,
    `sha256` of the archive, `include` the subdirectory to put on the include path (the root by
    default). Declaring is free: nothing is fetched until a kernel asks for `name`. Declaring the
    same one twice has no effect."""
    if name not in _externals:
        _externals[ name ] = External( name, version, url, sha256, include )


def registered_externals() -> dict:
    return dict( _externals )


def ext_root() -> Path:
    from . import cache_root
    override = env.var( "EXT_DIR" )
    return Path( override ).expanduser() if override else cache_root() / "ext"


def _fetch( ext: External ):
    """The archive, verified, unpacked into `ext.root` -- atomically: a sibling directory then a
    rename, so that a second process never sees a half-done unpacking."""
    ext_root().mkdir( parents = True, exist_ok = True )
    with tempfile.TemporaryDirectory( dir = ext_root(), prefix = f".{ ext.name }-" ) as tmp:
        tmp = Path( tmp )
        archive = tmp / "archive"
        with urllib.request.urlopen( ext.url, timeout = 120 ) as r, open( archive, "wb" ) as f:
            shutil.copyfileobj( r, f )
        digest = hashlib.sha256( archive.read_bytes() ).hexdigest()
        if digest != ext.sha256:
            raise RuntimeError( f"{ ext.name } { ext.version } : unexpected SHA-256 ({ digest }, expected { ext.sha256 })" )
        out = tmp / "out"
        out.mkdir()
        if zipfile.is_zipfile( archive ):
            with zipfile.ZipFile( archive ) as z:
                z.extractall( out )
        else:
            with tarfile.open( archive ) as t:
                t.extractall( out, filter = "data" ) if hasattr( tarfile, "data_filter" ) else t.extractall( out )
        entries = [ p for p in out.iterdir() ]
        top = entries[ 0 ] if len( entries ) == 1 and entries[ 0 ].is_dir() else out
        if ext.root.exists():
            return
        try:
            os.rename( top, ext.root )
        except OSError:
            if not ext.root.exists():                  # not another process: a real error
                shutil.move( str( top ), str( ext.root ) )


def _ensure( ext: External ):
    """`ext.include_dir` if it is there (already, or after download), None otherwise."""
    if ext.include_dir.is_dir():
        return ext.include_dir
    if not env.flag( "EXTERNALS", True ):
        return None
    try:
        _fetch( ext )
    except Exception as e:                                # network, disk, checksum: we say so, and carry on without
        print( f"loom: { ext.name } { ext.version } not available ({ e }) -- the kernels that ask for it will make do without",
               file = sys.stderr )
        return None
    return ext.include_dir if ext.include_dir.is_dir() else None


def include_dir_of( name ):
    """The include directory of the external `name`, downloaded at the first request; None if it
    cannot be had. Asked once per process (and per `LOOM_EXT_DIR`): a failed download is not
    retried at every kernel."""
    ext = _externals.get( name )
    if ext is None:
        raise ValueError( f"loom: a kernel asks for the external `{ name }`, which nobody declared "
                          f"(`loom.compilation.register_external`); declared: { sorted( _externals ) or 'none' }" )
    key = ( name, str( ext_root() ) )
    if key not in _resolved:
        _resolved[ key ] = _ensure( ext )
    return _resolved[ key ]


def external_include_dirs( names ) -> list:
    """The `-I` of the externals `names` that can be had (downloaded if needed)."""
    return [ d for d in ( include_dir_of( n ) for n in names ) if d is not None ]


# ---- the request, written into the kernel's source ------------------------------------------------

_MARKER = "// loom externals:"


def externals_marker( names ) -> str:
    """The first line of a kernel's source that asks for `names` (empty when none: the source of a
    kernel that asks for nothing does not change). Each with its version, so a new pin is a new
    kernel."""
    names = sorted( set( names ) )
    if not names:
        return ""
    for n in names:
        if n not in _externals:                              # a typo fails at render, not at link
            include_dir_of( n )                              # raises, with the list of what is declared
    return _MARKER + " " + " ".join( f"{ n }@{ _externals[ n ].version }" for n in names ) + "\n"


def externals_of( source: str ) -> tuple:
    """The names a kernel source asks for (`externals_marker`)."""
    if not source.startswith( _MARKER ):
        return ()
    line = source[ len( _MARKER ) : source.find( "\n" ) ]
    return tuple( item.split( "@", 1 )[ 0 ] for item in line.split() )
