"""A store for SHARED, generated C++ headers -- the mechanism only, nothing about what they hold.

A call would otherwise spell its boilerplate (an axis' `DEFINE_AXIS`, an aggregate's `struct`)
straight into its one-off `.cpp` -- a content-hashed file, unreadable and useless to an editor.
Yet that boilerplate is call-INDEPENDENT, so it belongs in a shared place, written ONCE and then
`#include`d by both the generated `.cpp` and any hand-written helper (which then gets declared
symbols to autocomplete, and compiles on its own).

WHERE, not what: WHAT a header contains is the business of the thing it describes -- an axis knows
its own `DEFINE_AXIS`, an aggregate its own `struct`. This module knows only the store. It lives
under the BUILD directory, never the sources: those may be read-only (a `sudo pip install`), and
writing into them is a bad habit regardless. `build_dir()` already picks a writable location (with
a per-user temp fallback), and `include_root()` is added to the compile `-I` path so the headers
resolve as `sdot/generated/...`. Write-if-changed keeps a deterministic content from churning
mtimes (hence rebuilds).

THE SHARED TREE IS NOT WHAT A KERNEL COMPILES AGAINST. A header is named after its type only
(`aggregates/<T>.h`), but its content depends on the CALL (which members an aggregate has, which
io policy) -- so two calls, or two processes sharing one build directory (errand runs several jobs
on one remote tree), overwrite each other's version between the write and the compile: a sporadic
`no instance of constructor`. Hence the overlay: `collecting_headers()` gathers what a call
rendered, and the compiler writes exactly that into the kernel's own content-hashed directory
(`write_overlay`), first on its `-I` path. The kernel's name hashes those contents, so an overlay
never changes once written -- no race, and depfiles that point at it never churn. The shared tree
is still written (atomically, best effort), for editors and for whatever a call did not render.
"""
from contextlib import contextmanager
from pathlib import Path
import threading
import os

from . import build_dir, include_roots


def manual_header( rel_path: str ) -> str | None:
    """The HAND-WRITTEN header at `rel_path` (`sdot/Cell.h`) if the user provides one on the C++
    source path, else `None`.

    This is the switch between an aggregate's two C++ modes: a manual header lets it carry methods
    written by hand (the user's `struct` drops in the generated macros and adds its own code);
    without one, the struct is generated WHOLE and needs no C++ at all. Only the sources are
    consulted -- a generated header (under the build tree) is never a manual override of itself."""
    return rel_path if any( ( root / rel_path ).is_file() for root in include_roots() ) else None


def include_root():
    """The `-I` root the generated headers live under -- kept apart from the build artifacts
    (`.cpp`, `.so`) so it is a clean include tree. Added to the compile flags by the driver."""
    root = build_dir() / "include"
    root.mkdir( parents = True, exist_ok = True )
    return root


def shared_header( rel_path: str, content: str ) -> str:
    """Declare that this call needs `content` at `rel_path` (e.g. `sdot/generated/axes/num_vertex.h`),
    and return `rel_path` -- the string to `#include`.

    The header is recorded by every active `collecting_headers()` (the compiled kernel gets it in
    its own overlay, see the module docstring), and mirrored into the SHARED tree under
    `include_root()`, for editors and hand-written helpers -- atomically, and only when the bytes
    would differ.

    What this process already mirrored is remembered: a call renders its headers EVERY time it
    runs (not only when it compiles), and re-reading each one from disk to find it unchanged was
    ~30 file reads per call -- more than the kernel itself, on a small problem. The cache may lie
    (another process may have written there since): that is harmless, nothing compiles against
    the shared tree's copy of a header the call rendered."""
    for headers in _collectors():
        headers[ rel_path ] = content
    if _written.get( rel_path ) == content:
        return rel_path
    try:
        write_if_changed( include_root() / rel_path, content )
    except OSError:
        pass  # best effort: the overlay is what compiles
    _written[ rel_path ] = content
    return rel_path


@contextmanager
def collecting_headers():
    """`with collecting_headers() as headers:` -- `headers` ( `{ rel_path: content }` ) receives
    every `shared_header` rendered inside the block, in this thread."""
    headers = {}
    stack = _collectors()
    stack.append( headers )
    try:
        yield headers
    finally:
        stack.pop()


def write_overlay( root: Path, headers: dict ) -> Path:
    """Write `headers` ( `{ rel_path: content }` ) under `root` -- the PRIVATE include tree of one
    kernel, to put first on its `-I` path. Atomic and write-if-changed: two processes compiling
    the same kernel write the same bytes, and an unchanged file keeps its mtime."""
    for rel_path, content in headers.items():
        write_if_changed( root / rel_path, content )
    return root


def headers_key( headers ) -> str:
    """What `headers` add to a kernel's cache key: their contents, in a canonical order -- the
    source only NAMES them, and the same name may hold another struct. Empty for no header, so a
    header-free kernel keeps its name."""
    if not headers:
        return ""
    return "|" + "|".join( f"{ k }\0{ v }" for k, v in sorted( headers.items() ) )


def write_if_changed( path: Path, content: str ):
    """`path` holds `content` afterwards. Written through a temporary file and a rename, so a
    concurrent reader (a compiler in another process) never sees a half-written header, and not
    written at all when it already holds these bytes (a new mtime would look like a change)."""
    try:
        if path.read_text() == content:
            return
    except OSError:
        pass
    path.parent.mkdir( parents = True, exist_ok = True )
    tmp = path.with_name( f".{ path.name }.{ os.getpid() }.{ threading.get_ident() }.tmp" )
    try:
        tmp.write_text( content )
        os.replace( tmp, path )
    finally:
        tmp.unlink( missing_ok = True )


def _collectors() -> list:
    stack = getattr( _local, "collectors", None )
    if stack is None:
        stack = _local.collectors = []
    return stack


_local = threading.local()
_written = {}
