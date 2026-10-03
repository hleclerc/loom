"""EXTERNAL C++ libraries, header-only, that the build chain fetches by itself: a package built on
loom declares one (`register_external`), and its archive is downloaded ONCE into the user cache
(`cache_root() / "ext"`) at the first kernel that compiles -- then its directory goes into the
`-I` of every compilation (`include_dirs`).

This is what makes a `#include <Eigen/SparseCholesky>` or `<amgcl/...>` valid on any machine,
with no system package or manual step: sdot depends on it for the linear solvers of its
transport (`sdot/sdotplan/Linear.cpp`), and a kernel catalogue is built with these same pinned
versions -- an archive and its SHA-256, not a branch.

Without network (`LOOM_EXTERNALS=0`, or a download that fails), the external is missing and we say
so ONCE: the source that expects it guards itself with `__has_include` and makes do with what it
has. `LOOM_EXT_DIR` puts the cache elsewhere (a build machine, a shared mount).
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
        self._reported = False

    @property
    def root( self ) -> Path:
        return ext_root() / f"{ self.name }-{ self.version }"

    @property
    def include_dir( self ) -> Path:
        return self.root / self.include if self.include else self.root


_externals = {}


def register_external( name, version, url, sha256, include = "" ):
    """Declares a library: `url` of an archive (`.tar.gz` / `.zip`) whose top level is stripped,
    `sha256` of the archive, `include` the subdirectory to put on the include path (the root by
    default). Declaring the same one twice has no effect."""
    if name not in _externals:
        _externals[ name ] = External( name, version, url, sha256, include )


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


def ensure( ext: External ) -> bool:
    """True if `ext.include_dir` is there (already, or after download)."""
    if ext.include_dir.is_dir():
        return True
    if not env.flag( "EXTERNALS", True ):
        return False
    try:
        _fetch( ext )
    except Exception as e:                                # network, disk, checksum: we say so, and carry on without
        if not ext._reported:
            ext._reported = True
            print( f"loom: { ext.name } { ext.version } not available ({ e }) -- the kernels that expect it will make do without",
                   file = sys.stderr )
        return False
    return ext.include_dir.is_dir()


def external_include_dirs() -> list:
    """The `-I` of the declared externals that are there (downloaded if needed)."""
    return [ ext.include_dir for ext in _externals.values() if ensure( ext ) ]
