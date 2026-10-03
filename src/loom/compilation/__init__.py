from pathlib import Path
import tempfile
import hashlib
import getpass
import sys
import os
from .. import env


def _src_root():
    """Root used to anchor the default (in-tree) build directory."""
    return Path( __file__ ).resolve().parents[ 4 ]


def _dev_repo_root():
    """Repo root, but only when this really looks like a dev checkout (loom/include/loom
    exists) -- None when running from an installed wheel."""
    candidate = Path( __file__ ).resolve().parents[ 4 ]
    return candidate if ( candidate / "loom" / "include" / "loom" ).is_dir() else None


def loom_include_root() -> Path:
    """The `-I` root so `#include <loom/support/...>` resolves: `<repo>/loom/include` from a dev
    checkout (edit C++ without rebuilding the wheel), the `loom/_include` tree the wheel ships next
    to the package otherwise (pure `__file__`-relative, no repo assumption)."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        return dev_root / "loom" / "include"
    packaged = Path( __file__ ).resolve().parents[ 1 ] / "_include"
    if ( packaged / "loom" / "support" ).is_dir():
        return packaged
    raise RuntimeError( "loom: cannot locate the C++ header tree (neither a dev checkout's "
                        "loom/include nor the packaged loom/_include were found) -- broken install?" )


# The C++ roots of the packages built ON loom (sdot's `sdot/include`, holding `sdot/` and
# `asimd/`): each registers its own at import (`register_include_root`), from a checkout or a
# wheel alike -- loom does not know who uses it. Generated headers live elsewhere, on their own
# root (`generated_headers.include_root()`).
_registered_include_roots = []


def register_include_root( path ):
    path = Path( path ).resolve()
    if path not in _registered_include_roots:
        _registered_include_roots.append( path )


def include_roots() -> list:
    """The `-I` roots of the HAND-WRITTEN C++ (loom's, then the registered ones), in order."""
    return [ loom_include_root(), *_registered_include_roots ]


def cpp_include_root():
    """The first registered root -- historically sdot's `sdot/include`. Prefer `include_roots()`."""
    if _registered_include_roots:
        return _registered_include_roots[ 0 ]
    return loom_include_root()


def _is_writable_dir( path: Path ):
    """Return True if `path` can be created and written into.

    We don't trust a mere `os.access(..., os.W_OK)` check: it lies on some
    platforms (notably Windows, and read-only network/overlay mounts), so we
    actually try to create the directory and write a probe file into it.
    """
    try:
        path.mkdir( parents = True, exist_ok = True )
        probe = path / f".write_probe_{ os.getpid() }"
        probe.write_text( "ok" )
        probe.unlink()
        return True
    except OSError:
        return False


def _fallback_build_dir( root: Path ):
    """A stable, per-user, per-checkout build directory under the temp dir.

    Used when the in-tree `build` directory is read-only (e.g. sources shipped
    in a read-only location). The path is deterministic for a given checkout so
    that incremental builds keep reusing the same .o / .a artifacts instead of
    rebuilding from scratch every time.

    `tempfile.gettempdir()` is honoured (TMPDIR/TEMP/TMP, then a sane default),
    so this works on macOS, Linux and Windows.
    """
    # Short hash of the checkout path: keeps distinct checkouts apart while
    # staying deterministic across runs.
    digest = hashlib.sha1( str( root ).encode( "utf-8" ) ).hexdigest()[ :12 ]

    # Including the user name avoids permission clashes in a world-shared
    # temp dir (typical on Linux: /tmp shared between users).
    try:
        user = getpass.getuser()
    except Exception:
        user = "anon"
    user = "".join( c if c.isalnum() else "_" for c in user )

    return Path( tempfile.gettempdir() ) / f"loom-build-{ user }-{ digest }"


def build_dir():
    """Directory where compilation artifacts (.o, .a, shared libs) are stored.

    Resolution order:
      1. `LOOM_BUILD_DIR` if set (explicit override).
      2. Dev checkout: `<repo>/build` when writable, else a stable per-user directory
         under the system temp dir (used when the checkout is read-only).
      3. Installed wheel (no dev checkout): the per-user cache root (`cache_root`) -- never
         inside the venv/site-packages.

    The chosen directory is created if needed and returned as a `Path`.

    The answer is CACHED per `LOOM_BUILD_DIR` value: it is asked on every generated header of every
    call (`generated_headers.shared_header`), and probing writability each time -- a `mkdir`, a
    probe file, an `unlink` -- was measured at a third of the per-call overhead of a small kernel.
    """
    override = env.var( "BUILD_DIR" )
    cached = _build_dir_cache.get( override )
    if cached is not None:
        return cached
    path = _build_dir_cache[ override ] = _resolve_build_dir( override )
    return path


_build_dir_cache = {}


def _resolve_build_dir( override ):
    if override:
        path = Path( override ).expanduser()
        path.mkdir( parents = True, exist_ok = True )
        return path

    dev_root = _dev_repo_root()
    if dev_root is not None:
        default = dev_root / "build"
        if _is_writable_dir( default ):
            return default

        fallback = _fallback_build_dir( dev_root )
        fallback.mkdir( parents = True, exist_ok = True )
        return fallback

    # installed wheel: no meaningful in-tree default -- the per-user cache root.
    installed_default = cache_root() / "build"
    installed_default.mkdir( parents = True, exist_ok = True )
    return installed_default


def cache_root() -> Path:
    """The first writable user cache directory: `LOOM_CACHE_DIR` if set, otherwise the platform
    convention (`~/.cache/loom`, `~/Library/Caches/loom`, `%LOCALAPPDATA%/loom/cache`), then
    `/tmp` as a last resort."""
    candidates = []
    override = env.var( "CACHE_DIR" )
    if override:
        candidates.append( Path( override ).expanduser() )
    if sys.platform == "darwin":
        candidates.append( Path.home() / "Library" / "Caches" / "loom" )
    elif os.name == "nt":
        base = os.getenv( "LOCALAPPDATA" ) or str( Path.home() / "AppData" / "Local" )
        candidates.append( Path( base ) / "loom" / "cache" )
    else:
        xdg = os.getenv( "XDG_CACHE_HOME" )
        candidates.append( ( Path( xdg ) if xdg else Path.home() / ".cache" ) / "loom" )
    if os.name != "nt":
        uid = os.getuid() if hasattr( os, "getuid" ) else "shared"
        candidates.append( Path( "/tmp" ) / f"loom-cache-{ uid }" )
    for p in candidates:
        if _is_writable_dir( p ):
            return p
    raise RuntimeError( "loom : could not find a writable cache directory. Set LOOM_CACHE_DIR to an explicit writable path." )


def include_dirs() -> list:
    """All the `-I` of a compilation: the C++ sources (loom, then the registered packages), the
    generated headers (under the build directory), and the declared external libraries
    (`externals.py`, downloaded if needed)."""
    from .generated_headers import include_root
    from .externals import external_include_dirs
    return [ *include_roots(), include_root(), *external_include_dirs() ]


from .externals import register_external as register_external


def make_library( lib_name, src_paths, device, *, extra_flags = None, sources = (), work_dir = None ):
    """Compile & link `src_paths` into a shared library with the compiler of `device`, through
    the build graph (see `build.py`).

    `work_dir`: the directory PRIVATE to this library. What belongs to nobody else -- the
    generated source and its object -- lives there, and goes away with it when it is erased; the
    domain units (`sources`) stay in the common graph, since they have dependents. Without
    `work_dir`, everything goes into the common graph (the catalogue, the C++ tests).

    Emits a relocatable shared object meant to be `dlopen`ed at runtime (e.g. to expose an XLA
    FFI handler symbol to Jax), linked against the runtime library (`libloom_runtime`: the
    process's thread pool). The output file name is taken verbatim, so callers are expected to
    make it unique -- typically a content hash of the sources + the compiler's `build_signature`
    (see `JaxFfi`).

    `sources`: extra units, `( path, defines )`, compiled once per (source, defines, compiler)
    and shared among all the kernels that ask for them -- the domain code one does not want to
    reinstantiate in every kernel (a density, a dimension: macros choose, the `.o` is made once).

    Rebuilds are ninja's business: a target is remade iff one of ITS inputs (the exact header
    closure, from the compiler's depfile) changed. `SDOT_FORCE_BUILD=1` remakes it regardless.
    Returns the path to the built library.
    """
    from .build import Build
    with Build( device, work_dir = work_dir ) as b:
        # the generated source has no dependent: its object is PRIVATE to this library
        objects = [ b.object( p, extra_flags = extra_flags or [], shared = work_dir is None )
                    for p in src_paths ]
        objects += [ b.object( p, defines ) for p, defines in sources ]
        lib = b.shared_library( ( work_dir or build_dir() ) / lib_name, objects, [ b.runtime_library() ] )
        b.run( [ lib ] )
    return lib


def make_executable( exe_name, src_paths, device, *, extra_flags = None, sources = () ):
    """Compile & link `src_paths` into an executable (the C++ tests). Same graph, same runtime
    library, as `make_library`."""
    from .build import Build
    with Build( device ) as b:
        objects = [ b.object( p, extra_flags = extra_flags or [] ) for p in src_paths ]
        objects += [ b.object( p, defines ) for p, defines in sources ]
        exe = b.executable( build_dir() / exe_name, objects, [ b.runtime_library() ] )
        b.run( [ exe ] )
    return exe
