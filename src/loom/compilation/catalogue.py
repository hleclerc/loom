"""The CATALOGUE: precompiled kernels, shipped in a wheel, so that a standard use has nothing to compile.

It is a DIRECTORY, and the directory is the index:

    <root>/<tag>/<key>.so                 one library per kernel
    <root>/<tag>/libloom_runtime_*.so     the runtime they share (found through `$ORIGIN` / `@loader_path`)
    <root>/<tag>/index.json               what is in it, for humans and for `report` -- never read to look up
    <root>/<tag>/xla-ffi.json             the XLA FFI API version its Jax kernels were compiled against

A TAG is `<kind>-<target>` (`cpu-x86-64-v3`, `cuda-sm_80`): the architecture the kernels of that directory were
compiled for (`Device.catalogue_tag`). A KEY is the hash of the kernel's source and of its domain sources -- nothing of
the machine, of the compiler or of the architecture, so it is the same for everyone. To serve a kernel, a machine
looks in ONE tag: the architecture its device behaves as for that kernel (`Device.current_catalogue_tag`) -- a `stat`,
no index to load, and only the libraries that are used get loaded. There is NO fall back on a poorer architecture: its
build is another code (other results, other timings), and which architecture a kernel runs as is a decision that
belongs to `choices.py`, taken from measurements. A kernel the tag does not hold is compiled (`auto`) or missing
(`catalogue`).

= Producing it: run the program, on every hardware

There is nothing to record and nothing to compile elsewhere. The program that exercises the kernels (for `sdot`,
`python -m sdot.catalogue`) is a Python script, and the loops on the hardware belong to it:

    from loom import catalogue, hardware
    with catalogue.collect( "my_package/_catalogue" ):
        for run in hardware.runs():              # each device present, as each architecture it can behave as
            with hardware.behave_as( run ):
                exercise_the_package()           # whatever triggers the kernels
    print( catalogue.report( "my_package/_catalogue" ) )

While a catalogue is being collected, every kernel the program triggers is compiled (the catalogue is not consulted)
and its library is copied under the tag of the architecture the device behaves as. The hardware that decides things in
Python that loom does not manage (`device.nb_threads`, `group_size`) is seen as it would be on each architecture, because
the device really behaves as it (`devices/hardware.py`); a kernel whose source depends on it simply gets another key.

The directory can be fed from several machines (a CPU one, a GPU one): the tags are disjoint, merge the directories.
It is the wheel that ships it, registered at import time by `register_catalogue`.

= The XLA FFI API version

A Jax kernel declares the XLA FFI API version of the jaxlib headers it was compiled with, and a jaxlib refuses a handler
of a NEWER version than its own ( an older one is fine ). `xla-ffi.json` records it per tag, when the catalogue is
collected; a jaxlib older than that does not take the Jax kernels of the tag: they are compiled, as with no catalogue.
Collect with a jaxlib old enough for the users ( the version does not change the kernels' code, only this check ). The
Pointer kernels ( Torch, Cupy, Numpy ) do not depend on jaxlib.

`LOOM_KERNELS`: `auto` (catalogue, otherwise compile -- the default), `catalogue` (never compile: an absence is an
error that names the kernel), `compile` (always compile, ignore the catalogue -- which is what a development checkout
does anyway, it has none).
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import ctypes
import json
import shutil
import subprocess
import sys
import os

from ..util.encode_base_62 import encode_base_62
from .. import env

_roots = []           # catalogue directories, in registration order
_loaded_libs = {}     # path -> CDLL
_collecting = None    # the directory being fed, or None
_published = {}       # tag -> { key: { "name", "signature", "owners" } }, of this process
_published_ffi = {}   # tag -> the XLA FFI API version of the Jax kernels published under it, by this process
_ffi_of_tag = {}      # tag directory -> the XLA FFI API version it records ( None: none recorded )
_owner = None         # who is asking for the kernels being compiled right now: a kernel's choice (`choices.directory_name`), or None


def policy() -> str:
    v = env.var( "KERNELS", "auto" ).strip().lower()
    if v not in ( "auto", "catalogue", "compile" ):
        raise ValueError( f"LOOM_KERNELS={ v !r}: expected auto, catalogue or compile" )
    return v


def library_suffix() -> str:
    from .Compiler import shared_suffix
    return shared_suffix()


def key( source: str, sources, flavor: str = "" ) -> str:
    """A kernel's key in a catalogue: source + domain sources (paths as given, relative to the C++
    roots). The architecture is NOT in it: it is the tag (the directory) that carries it. In a canonical
    form, so that the same kernel has the same key when it is published and when it is looked up."""
    canon = json.dumps( [ [ str( p ), [ [ str( k ), str( v ) ] for k, v in d ] ] for p, d in sources ], sort_keys = True )
    # `flavor`: the ABI the kernel is compiled for when it is not Jax's (`"pointer"`: Torch, Cupy, Numpy). The text of the source
    # can be the same; the binary is not. Nothing is added for Jax's, which keeps its keys.
    h = hashlib.sha256( f"{ source }|{ canon }{ '|' + flavor if flavor else '' }".encode() ).hexdigest()
    return encode_base_62( h )[ :24 ]


# ── serving ─────────────────────────────────────────────────────────────────

def register_catalogue( root ):
    """Add a catalogue directory (`<root>/<tag>/<key>.so`). Called by the package that ships one (`sdot/__init__.py`);
    a missing directory is not an error -- a checkout has none."""
    root = Path( root )
    if root.is_dir() and root not in _roots:
        _roots.append( root )


def registered_tags() -> list:
    return sorted( { d.name for root in _roots for d in root.iterdir() if d.is_dir() } )


def lookup( source: str, sources, device, flavor: str = "" ):
    """`( library, handler )` of the precompiled kernel that serves this source on this device, or None: the file of the
    tag of the architecture the device behaves as, and no other."""
    if _collecting is not None or policy() == "compile" or not _roots:
        return None
    tag = device.current_catalogue_tag()
    if tag is None:
        return None
    from ..drivers.JaxFfi import _HANDLER_SYMBOL
    name = key( source, sources, flavor ) + library_suffix()
    for root in _roots:
        path = root / tag / name
        if path.is_file() and ( flavor or _takes_jax_kernels_of( root / tag ) ):
            lib = _loaded_libs.get( path )
            if lib is None:
                from .Compiler import load_library
                lib = _loaded_libs[ path ] = load_library( path )
            return lib, getattr( lib, _HANDLER_SYMBOL )
    return None


def serves_jax_kernels( tag: str ) -> bool:
    """Whether a registered catalogue serves this jaxlib the Jax kernels of `tag` ( see `_takes_jax_kernels_of` )."""
    return any( _takes_jax_kernels_of( root / tag ) for root in _roots if ( root / tag ).is_dir() )


def _takes_jax_kernels_of( tag_dir: Path ) -> bool:
    """Whether this jaxlib accepts the Jax kernels of `tag_dir`: compiled against an XLA FFI API of the same major and of
    a minor not newer than its own. A tag that records no version ( collected before it was recorded ) is not trusted."""
    if tag_dir not in _ffi_of_tag:
        path = tag_dir / "xla-ffi.json"
        _ffi_of_tag[ tag_dir ] = tuple( json.loads( path.read_text() )[ "api_version" ] ) if path.is_file() else None
    built = _ffi_of_tag[ tag_dir ]
    if built is None:
        return False
    from ..drivers.JaxFfi import ffi_api_version
    here = ffi_api_version()
    return built[ 0 ] == here[ 0 ] and built[ 1 ] <= here[ 1 ]


# ── producing ───────────────────────────────────────────────────────────────

@contextmanager
def collect( out ):
    """Feed the catalogue directory `out` with every kernel the program compiles inside the block, under the tag of the
    architecture the device behaves as. The catalogue is not consulted meanwhile: what is served is what gets compiled.
    `index.json` of each tag is written when the block ends."""
    global _collecting
    if _collecting is not None:
        raise RuntimeError( "loom: a catalogue is already being collected" )
    _collecting = Path( out ).resolve()
    _collecting.mkdir( parents = True, exist_ok = True )
    _published.clear()
    _published_ffi.clear()
    try:
        yield _collecting
    finally:
        try:
            for tag, entries in _published.items():
                _write_index( _collecting / tag, entries )
            for tag, version in _published_ffi.items():
                _write_ffi_version( _collecting / tag, version )
        finally:
            _collecting = None


def collecting():
    return _collecting


@contextmanager
def owned_by( owner: dict | None ):
    """Inside the block, the kernels that get published are asked for by `owner` -- `{ "name", "extra_keys" }` of a
    `loom.kernel_parameters_for`, whose choice of architecture says which tags it can ever need (see `prune`). Kernels
    asked for any other way have no owner, and are kept under every tag."""
    global _owner
    previous, _owner = _owner, owner
    try:
        yield
    finally:
        _owner = previous


def _write_index( tag_dir: Path, entries: dict ):
    path = tag_dir / "index.json"
    known = json.loads( path.read_text() ) if path.exists() else {}
    for k, meta in entries.items():
        old = known.get( k, {} ).get( "owners", [] )
        meta[ "owners" ] = old + [ o for o in meta.get( "owners", [] ) if o not in old ]
        known[ k ] = meta
    tmp = path.with_suffix( ".json.tmp" )
    tmp.write_text( json.dumps( known, indent = 1, sort_keys = True ) )
    tmp.replace( path )


def _write_ffi_version( tag_dir: Path, version: tuple ):
    """Record the XLA FFI API version of the Jax kernels of `tag_dir`. A directory fed by several jaxlib ( merged from
    several machines ) records the newest: the one a jaxlib must reach to take all of them."""
    path = tag_dir / "xla-ffi.json"
    if path.is_file():
        version = max( version, tuple( json.loads( path.read_text() )[ "api_version" ] ) )
    path.write_text( json.dumps( { "api_version": list( version ) } ) + "\n" )
    _ffi_of_tag.pop( tag_dir, None )


def publish( library, source: str, sources, device, code_name = None, signature = None, flavor: str = "" ):
    """Copy the library of a kernel that was just compiled into the catalogue being collected, under the tag of the
    architecture the device behaves as. Does nothing outside `collect`. Returns the path it wrote (None if none)."""
    if _collecting is None:
        return None
    target = device.catalogue_target()
    if target is None:
        return None
    from .build import Build

    tag = device.catalogue_tag( target )
    k = key( source, sources, flavor )
    if not flavor:
        from ..drivers.JaxFfi import ffi_api_version
        _published_ffi[ tag ] = max( _published_ffi.get( tag, ( 0, 0 ) ), ffi_api_version() )
    if k in _published.get( tag, {} ):
        _own( _published[ tag ][ k ] )
        return _collecting / tag / ( k + library_suffix() )
    tag_dir = _collecting / tag
    tag_dir.mkdir( parents = True, exist_ok = True )

    # the kernel is linked against the runtime library and the domain units, and finds them next to itself
    from . import dependencies
    needs = dependencies.get( Path( library ).name )
    if needs is None:
        with Build( device ) as b:
            needs = [ Path( b.runtime_library() ) ]
    for dep in needs:
        _copy( Path( dep ), tag_dir / Path( dep ).name )
    dest = tag_dir / ( k + library_suffix() )
    _copy( Path( library ), dest )

    _published.setdefault( tag, {} )[ k ] = _own( { "name": code_name, "signature": None if signature is None else str( signature ),
                                                    "libraries": sorted( Path( d ).name for d in needs ) } )
    return dest


def _own( meta: dict ) -> dict:
    owners = meta.setdefault( "owners", [] )
    if _owner not in owners:
        owners.append( _owner )
    return meta


def _copy( src: Path, dest: Path ):
    """Atomic, and skipped when the file is already there and not older than its source (several runs may publish the same
    kernel: the same source compiled for the same tag). The copy is stripped of the symbols the loader does not need: the
    mangled names of templates are about half of a kernel's file on macOS, a fifth to a quarter on Linux."""
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    tmp = dest.with_name( f".{ dest.name }.{ os.getpid() }.tmp" )
    shutil.copy2( src, tmp )
    _strip( tmp )
    tmp.replace( dest )


def _strip( path: Path ):
    """Remove the local symbols from `path` (the exported ones, the handler, stay). Best effort: a missing or failing
    `strip` leaves the file as it was."""
    tool = shutil.which( "strip" )
    if tool is None:
        return
    flag = "-x" if sys.platform == "darwin" else "--strip-unneeded"
    try:
        subprocess.run( [ tool, flag, str( path ) ], check = True, capture_output = True )
    except ( OSError, subprocess.SubprocessError ):
        pass


# ── reading what was produced ───────────────────────────────────────────────

def index( root ) -> dict:
    """`{ tag: { key: { "name", "signature" } } }`, from the `index.json` of each tag."""
    out = {}
    for d in sorted( Path( root ).iterdir() ) if Path( root ).is_dir() else []:
        if ( d / "index.json" ).is_file():
            out[ d.name ] = json.loads( ( d / "index.json" ).read_text() )
    return out


def hardware_dependent( root ) -> dict:
    """`{ name: { key: [ tags ] } }`: the kernels that not every run of their family produced. Everything the
    program does is the same on every run, except what depends on the hardware (in loom, or in the Python above it that
    asks the device) -- those sources give other keys, present under some tags only."""
    by_kind = {}
    for tag, entries in index( root ).items():
        by_kind.setdefault( tag.split( "-", 1 )[ 0 ], {} )[ tag ] = entries
    found = {}
    for kind, tags in by_kind.items():
        for tag, entries in tags.items():
            for k, meta in entries.items():
                where = [ t for t, e in tags.items() if k in e ]
                if len( where ) != len( tags ):
                    found.setdefault( meta.get( "name" ), {} )[ k ] = where
    return found


def report( root ) -> str:
    """A text summary of a collected catalogue: what each tag holds, and the kernels that depend on the hardware."""
    lines = []
    for tag, entries in index( root ).items():
        size = sum( f.stat().st_size for f in ( Path( root ) / tag ).iterdir() if f.is_file() )
        lines.append( f"  { tag:20s} { len( entries ):4d} kernel(s), { size / 1e6:.1f} MB" )
    for name, keys in sorted( hardware_dependent( root ).items(), key = str ):
        lines.append( f"  `{ name }` depends on the hardware:" )
        for k, tags in keys.items():
            lines.append( f"      { k }: { ', '.join( tags ) }" )
    return "\n".join( [ f"catalogue { root }:", *lines ] )


# ── keeping only what is useful ─────────────────────────────────────────────

def prune( root, choices_roots = None ) -> list:
    """Remove from the collected catalogue `root` the kernels that no machine will ever ask for. A kernel asked for through
    `loom.kernel_parameters_for` runs, on a machine of architecture R, as the architecture its choice says (`choices.read`,
    R itself when nothing is recorded): under a tag T it is needed only if some R of the family resolves to T. A kernel that
    anyone asked for any other way (no owner) is needed everywhere. Returns the `( tag, key )` removed."""
    from .. import choices
    from ..devices.Device import Device
    roots = [ Path( r ) for r in ( choices_roots if choices_roots is not None else choices._roots ) ]
    removed = []
    for tag, entries in index( root ).items():
        kind = tag.split( "-", 1 )[ 0 ]
        targets = Device.factory( "cuda:0" if kind == "cuda" else "cpu" ).catalogue_targets()
        this = tag.split( "-", 1 )[ 1 ]
        keep = {}
        for k, meta in entries.items():
            owners = meta.get( "owners", [ None ] )
            needed = None in owners or not owners or any(
                this in { ( choices.read( o[ "name" ], o[ "extra_keys" ], R, roots ) or ( R, ) )[ 0 ] for R in targets } for o in owners )
            if needed:
                keep[ k ] = meta
            else:
                ( Path( root ) / tag / ( k + library_suffix() ) ).unlink( missing_ok = True )
                removed.append( ( tag, k ) )
        if len( keep ) != len( entries ):
            ( Path( root ) / tag / "index.json" ).write_text( json.dumps( keep, indent = 1, sort_keys = True ) )
            used = { lib for meta in keep.values() for lib in meta.get( "libraries", [] ) }
            for f in ( Path( root ) / tag ).glob( "libloom_*" ):
                if f.name not in used and keep:
                    f.unlink()
    return removed


def build( package: str, usage ):
    """Build the catalogue of `package` -- all of it, from a script of a few lines:

        # my_package/catalogue.py
        import loom
        def usage():                       # what a standard use does: whatever triggers the kernels
            ...
            for _ in loom.bench_loop( "my_kernel", weight = n ):      # and, where it matters, what to time
                kp.ffi_call( ... )
        loom.catalogue.build( "my_package", usage )

    `usage` is run on every piece of hardware present, as every architecture it can behave as; the kernels it compiles are
    collected into `<package>/precompiled-kernels/`. The loops of `loom.bench_loop` are timed on every run, which decides
    which architecture each kernel runs as (`<package>/choices/`); what no machine will ask for is then removed (`prune`).
    Nothing to configure: the places are the package's, the hardware is what is present."""
    from importlib.util import find_spec
    from ..devices import hardware
    from ..benchmark import Measurements
    from ..drivers.Framework import UnsupportedDevice
    from .. import choices
    base = Path( find_spec( package ).submodule_search_locations[ 0 ] )
    out, chosen = base / "precompiled-kernels", base / "choices"
    runs = hardware.runs()
    if not runs:
        raise SystemExit( f"loom: no hardware to build the catalogue of { package } on" )
    from ..devices.Device import Device
    for kind in sorted( { r.device.split( ":" )[ 0 ] for r in runs } ):
        missing = hardware.uncovered( runs, kind, Device.factory( kind ).catalogue_targets() )
        if missing:
            print( f"{ kind }: no hardware here can run { ', '.join( missing ) } -- no build for them" )
    with Measurements( chosen ) as measures, collect( out ):
        for run in runs:
            print( f"== { run.device } as { run.target }", flush = True )
            try:
                with hardware.behave_as( run ):
                    usage()
            except UnsupportedDevice as e:                  # numpy has no card, cupy no CPU: this framework does not visit it
                print( f"   skipped: { e }", flush = True )
    for name, _, device, target, use, scores in measures.write():
        if target == Device.factory( device ).real_target():
            print( f"{ name } on { device }: { target } -> { use }   ( mean time relative to the best: { ', '.join( f'{ t } { s:.3f}' for t, s in scores.items() ) } )" )
    choices.register_choices( chosen )
    removed = prune( out, [ chosen ] )
    print( f"pruned { len( removed ) } kernel(s) that no machine will ask for" )
    print( report( out ) )
