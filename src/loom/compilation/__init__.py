from pathlib import Path
import subprocess
import tempfile
import hashlib
import getpass
import sys
import os


def _src_root():
    """Root used to anchor the default (in-tree) build directory."""
    return Path( __file__ ).resolve().parents[ 4 ]


def _dev_repo_root():
    """Repo root, but only when this really looks like a dev checkout (loom/include/loom
    exists) -- None when running from an installed wheel."""
    candidate = Path( __file__ ).resolve().parents[ 4 ]
    return candidate if ( candidate / "loom" / "include" / "loom" ).is_dir() else None


def cpp_include_root():
    """The `-I` root so `#include <sdot/Cell.h>` resolves, from a dev checkout OR an
    installed wheel.

    Dev: `<repo>/sdot/include` (edit C++ without rebuilding the wheel). Installed: the
    `sdot/_include` tree the wheel ships next to the package (pure `__file__`-relative,
    no repo assumption). The generated headers live elsewhere, on their own `-I` root --
    see generated_headers.include_root().
    """
    dev_root = _dev_repo_root()
    if dev_root is not None:
        return dev_root / "sdot" / "include"

    packaged = Path( __file__ ).resolve().parents[ 1 ] / "_include"
    if ( packaged / "sdot" / "Cell.h" ).is_file():
        return packaged

    raise RuntimeError(
        "sdot: cannot locate the C++ header tree (neither a dev checkout's sdot/include nor "
        "the packaged sdot/_include were found) -- broken install?"
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
      3. Installed wheel (no dev checkout): the per-user cache root (`cache_root`) -- never
         inside the venv/site-packages.

    The chosen directory is created if needed and returned as a `Path`.

    The answer is CACHED per `SDOT_BUILD_DIR` value: it is asked on every generated header of every
    call (`generated_headers.shared_header`), and probing writability each time -- a `mkdir`, a
    probe file, an `unlink` -- was measured at a third of the per-call overhead of a small kernel.
    """
    override = os.getenv( "SDOT_BUILD_DIR" )
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


def additional_include_dirs():
    """Extra `-I` roots needed alongside cpp_include_root() — e.g. the loom support headers."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        return [ str( dev_root / "loom" / "include" ) ]
    return []


def cache_root() -> Path:
    """Le premier répertoire de cache utilisateur inscriptible : `SDOT_CACHE_DIR` s'il est mis,
    sinon la convention de la plateforme (`~/.cache/sdot`, `~/Library/Caches/sdot`,
    `%LOCALAPPDATA%/sdot/cache`), puis `/tmp` en dernier recours."""
    candidates = []
    override = os.getenv( "SDOT_CACHE_DIR" )
    if override:
        candidates.append( Path( override ).expanduser() )
    if sys.platform == "darwin":
        candidates.append( Path.home() / "Library" / "Caches" / "sdot" )
    elif os.name == "nt":
        base = os.getenv( "LOCALAPPDATA" ) or str( Path.home() / "AppData" / "Local" )
        candidates.append( Path( base ) / "sdot" / "cache" )
    else:
        xdg = os.getenv( "XDG_CACHE_HOME" )
        candidates.append( ( Path( xdg ) if xdg else Path.home() / ".cache" ) / "sdot" )
    if os.name != "nt":
        uid = os.getuid() if hasattr( os, "getuid" ) else "shared"
        candidates.append( Path( "/tmp" ) / f"sdot-cache-{ uid }" )
    for p in candidates:
        if _is_writable_dir( p ):
            return p
    raise RuntimeError( "sdot: could not find a writable cache directory. Set SDOT_CACHE_DIR to an explicit writable path." )


def include_dirs() -> list:
    """Tous les `-I` d'une compilation : les sources C++ de sdot, ceux de loom, les en-têtes
    générés (sous le répertoire de build)."""
    from .generated_headers import include_root
    return [ cpp_include_root(), *additional_include_dirs(), include_root() ]


def force_build_level() -> int:
    """`SDOT_FORCE_BUILD` as a level, not a bool -- see `make_library`.

    0 : unset / "0" / "false" / "no" / "off" -- plain disk cache, reuse the library as-is.
    1 : "1" (or any other truthy value) -- hash-checked rebuild: compare the hand-written C++
        sources against the stamp left next to the library, rebuild only on a mismatch. The
        default for test runs (`cli/main.py`), since it is what makes "the header changed"
        distinguishable from "nothing changed" without paying for a full rebuild every time.
    2 : "2" -- unconditional rebuild, ignoring both the library and the stamp. For when the hash
        itself is under suspicion, or the toolchain/flags changed in a way the hash can't see.
    """
    v = os.getenv( "SDOT_FORCE_BUILD" )
    if v is None:
        return 0
    v = v.strip().lower()
    if v in ( "", "0", "false", "no", "off" ):
        return 0
    return 2 if v == "2" else 1


_source_hash_cache = {}


def cpp_sources_hash() -> str:
    """SHA-256 over every hand-written `.h`/`.hpp`/`.cxx`/`.cpp` under the C++ include roots.

    `make_library`'s disk cache is keyed on the *generated* .cpp text alone (see its docstring):
    a kernel's fwd/bwd body is in there, but the hand-written headers/sources it `#include`s
    (`sdot/include`, `loom/include`) are not, so editing one of those does not change the key.
    This hash covers exactly that gap. Walking + reading the whole include tree costs real time,
    so it is computed once per (include roots) and memoized -- fine since nothing under those
    roots changes while a test process is running.

    (L'étape « couches » de la refonte remplace ce hachage global par les depfiles du
    compilateur : la fermeture d'en-têtes exacte de chaque unité, et rien d'autre.)
    """
    roots = tuple( sorted( { str( cpp_include_root() ), *( str( d ) for d in additional_include_dirs() ) } ) )
    cached = _source_hash_cache.get( roots )
    if cached is not None:
        return cached

    files = []
    for root in roots:
        root_path = Path( root )
        if root_path.is_dir():
            files += ( p for p in root_path.rglob( "*" ) if p.suffix in ( ".h", ".hpp", ".cxx", ".cpp" ) )

    h = hashlib.sha256()
    for p in sorted( files ):
        h.update( str( p ).encode() )
        h.update( p.read_bytes() )

    digest = h.hexdigest()
    _source_hash_cache[ roots ] = digest
    return digest


def _run( cmd ):
    print( "[cxx] $ " + " ".join( map( str, cmd ) ), flush = True )
    r = subprocess.run( list( map( str, cmd ) ) )
    if r.returncode:
        raise RuntimeError( f"command failed ({ r.returncode }): { ' '.join( map( str, cmd ) ) }" )


def _build( out_name, src_paths, device, extra_flags, kind ):
    """Le cache disque commun à `make_library` / `make_executable`, puis la commande du
    compilateur du device."""
    out_dir = build_dir()
    out_dir.mkdir( parents = True, exist_ok = True )
    out = out_dir / out_name
    stamp = out.with_name( out.name + ".srchash" )

    level = force_build_level()
    if out.exists() and level != 2:
        if level == 0:
            return out
        # level == 1: reuse only if the hand-written sources haven't moved since this exact
        # library was built (an older one, or one built at level 0/2, has no stamp -> rebuild).
        if stamp.is_file() and stamp.read_text().strip() == cpp_sources_hash():
            return out

    _run( device.compiler.command( src_paths, out, include_dirs(), extra_flags, kind ) )
    if level == 1:
        stamp.write_text( cpp_sources_hash() )
    elif stamp.exists():
        stamp.unlink()
    return out


def make_library( lib_name, src_paths, device, *, extra_flags = None ):
    """Compile & link `src_paths` into a shared library with the compiler of `device`.

    Emits a relocatable shared object (`-shared -fPIC`) meant to be `dlopen`ed at runtime (e.g.
    to expose an XLA FFI handler symbol to Jax). The output file name is taken verbatim, so
    callers are expected to make it unique -- typically a content hash of the sources + the
    compiler's `build_signature` (see `JaxFfi`).

    Disk cache: if the target already exists it is returned as-is, unless `SDOT_FORCE_BUILD`
    says otherwise (the dev/test override) -- see `force_build_level`. Since the file name is a
    hash of the *generated* source, a changed kernel body naturally produces a new name and a
    rebuild on its own; `SDOT_FORCE_BUILD` only matters for the hand-written headers/sources
    that name can't see (level 1: rebuild only if their content hash moved since the last build
    of *this* name; level 2: always). Returns the path to the built library.
    """
    return _build( lib_name, src_paths, device, extra_flags, "shared" )


def make_executable( exe_name, src_paths, device, *, extra_flags = None ):
    """Compile & link `src_paths` into an executable (the C++ tests). Same cache as
    `make_library`."""
    return _build( exe_name, src_paths, device, extra_flags, "binary" )
