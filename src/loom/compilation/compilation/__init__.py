from ..devices.Device import Device
from pathlib import Path
import subprocess
import tempfile
import hashlib
import getpass
import shutil
import sys
import os


def _src_root():
    """Root used to anchor the default (in-tree) build directory."""
    return Path( __file__ ).resolve().parents[ 4 ]


def _dev_repo_root():
    """Repo root, but only when this really looks like a dev checkout (a sibling
    src/cpp/sdot exists) -- None when running from an installed wheel."""
    candidate = Path( __file__ ).resolve().parents[ 4 ]
    return candidate if ( candidate / "src" / "cpp" / "sdot" ).is_dir() else None


def cpp_include_root():
    """The `-I` root so `#include <sdot/Cell.h>` resolves, from a dev checkout OR an
    installed wheel.

    Dev: the sibling `src/cpp` tree (edit C++ without rebuilding the wheel). Installed: the
    `sdot/_cpp` tree the wheel ships next to the package (pure `__file__`-relative, no repo
    assumption). The generated headers live elsewhere, on their own `-I` root -- see
    generated_headers.include_root().
    """
    dev_root = _dev_repo_root()
    if dev_root is not None:
        return dev_root / "src" / "cpp"

    packaged = Path( __file__ ).resolve().parents[ 1 ] / "_cpp"
    if ( packaged / "sdot" / "Cell.h" ).is_file():
        return packaged

    raise RuntimeError(
        "sdot: cannot locate the C++ header tree (neither a dev checkout's src/cpp nor "
        "the packaged sdot/_cpp were found) -- broken install?"
    )


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

    return Path( tempfile.gettempdir() ) / f"sdot-build-{ user }-{ digest }"


def build_dir():
    """Directory where compilation artifacts (.o, .a, shared libs) are stored.

    Resolution order:
      1. `SDOT_BUILD_DIR` if set (explicit override).
      2. Dev checkout: `<repo>/build` when writable, else a stable per-user directory
         under the system temp dir (used when the checkout is read-only).
      3. Installed wheel (no dev checkout): the per-user cache root, the same convention
         as the AdaptiveCpp toolchain -- never inside the venv/site-packages.

    The chosen directory is created if needed and returned as a `Path`.
    """
    override = os.getenv( "SDOT_BUILD_DIR" )
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

    # installed wheel: no meaningful in-tree default -- reuse the same per-user cache
    # convention as the AdaptiveCpp toolchain itself. (local import avoids a circular
    # import between this module and adaptive_cpp.)
    from .adaptive_cpp import cache_root
    installed_default = cache_root() / "build"
    installed_default.mkdir( parents = True, exist_ok = True )
    return installed_default


def additional_include_dirs():
    return []


def make_executable( exe_name: str, src_paths: list, device: Device, requires = None ):
    """Build a standalone executable from *src_paths* using the shared compilation/xmake.lua.

    Counterpart of make_dylib_from_files for the C++/CUDA tests: produces a binary
    (SDOT_XMAKE_KIND=binary), links Catch2 instead of nanobind, and — for CUDA — routes the
    sources through nvcc via a generated .cu shim (nvcc is selected by the .cu extension,
    exactly like the bindings). Returns the path to the built executable.
    """
    project_root = Path( __file__ ).absolute().parents[ 4 ]
    src_paths = [ Path( p ) for p in src_paths ]
    requires = list( requires or [ "catch2" ] )

    # CUDA: wrap the .cpp sources in a .cu shim so nvcc compiles them (defines __CUDACC__)
    if device.is_cuda_gpu:
        raise NotImplementedError
        # shim = compilation_directories.src_dir( exe_name ) / f"{ exe_name }.cu"
        # shim.write_text( "".join( f'#include "{ p }"\n' for p in src_paths ) )
        # sources = [ shim ]
    else:
        sources = src_paths

    extended_path = os.pathsep.join( p for p in [
        str( Path( sys.executable ).parent ),
        str( Path.home() / ".local" / "bin" ),  # default xmake.io install
        "/opt/homebrew/bin",                    # homebrew Apple Silicon
        "/usr/local/bin",                       # homebrew Intel
        os.environ.get( "PATH", "" ),
    ] if p )

    xmake_bin = shutil.which( "xmake", path = extended_path )
    if xmake_bin is None:
        raise RuntimeError( "xmake introuvable (brew install xmake ou https://xmake.io)" )

    output_dir = build_dir() # / "tests"
    output_dir.mkdir( parents = True, exist_ok = True )

    print( output_dir )

    env = {
        **os.environ,
        **( { "XMAKE_ROOT": "y" } if hasattr( os, "getuid" ) and os.getuid() == 0 else {} ),
        "SDOT_XMAKE_KIND"      : "binary",
        "SDOT_XMAKE_TARGET"    : exe_name,
        "SDOT_XMAKE_OUTPUT_DIR": str( output_dir ),
        "SDOT_XMAKE_NEEDS_CUDA": str( int( device.is_cuda_gpu ) ),
        "SDOT_XMAKE_REQUIRES"  : ",".join( requires ),
        "SDOT_XMAKE_INCLUDES"  : str.join( ",", map( str, [
                                      project_root / "src" / "cpp"
                                  ] + additional_include_dirs() ) ),
        "SDOT_XMAKE_CXXFLAGS"  : "-fno-strict-aliasing",
        "SDOT_XMAKE_SOURCES"   : ",".join( map( str, sources ) ),
        "SDOT_XMAKE_DEFINES"   : "",
        "PATH"                 : extended_path,
    }

    sdot_dir = Path( __file__ ).parents[ 4 ] / "scripts"  # holds xmake.lua
    mode = os.environ.get( "SDOT_XMAKE_MODE", "release" )

    def run( cmd ):
        if subprocess.run( cmd, cwd = output_dir, env = env ).returncode:
            raise RuntimeError( f"xmake failed: { ' '.join( map( str, cmd ) ) }" )

    run( [ xmake_bin, "f", "-P", str( sdot_dir ), "-y", "--require=yes", "-m", mode ] )
    run( [ xmake_bin, "-P", str( sdot_dir ), "-v" ] )

    return output_dir / exe_name
