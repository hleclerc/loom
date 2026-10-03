"""The CATALOGUE: precompiled kernels, shipped in a wheel, so that a standard use has
nothing to compile.

Three steps:

  1. RECORD (`SDOT_CATALOGUE_RECORD=dir`): on a development machine, run whatever
     triggers the desired compilations (the tests, `python -m sdot.catalogue`); every generated
     source that goes through `compile_and_register` is dropped into `dir` with what is needed to
     recompile it elsewhere (the device, the domain sources, the generated headers). A GPU is
     only needed here, to EXECUTE the calls -- not to compile.

  2. COMPILE (`scripts/build_catalogue.py compile`): for a variant (`x86-64-v3`, or a list
     of CUDA architectures), each recorded source becomes an object, all are linked into ONE
     library (`libsdot_kernels.so`, with the runtime inside) and `catalogue.json` says which
     entry point serves which key. This is what continuous integration does, per platform.

  3. LOOK UP (at execution time): a kernel's key is the hash of its source, of its domain
     sources and of the catalogue's TAG (`cpu-x86-64-v3`, `cuda`) -- nothing of the machine or of the
     compiler, so the same key for everyone. At import, the package registers its
     catalogue directory (`register_catalogue`); at call time, `lookup` tries the tags
     that the machine can load, from richest to poorest, and returns the entry point.
     Nothing to walk, nothing to validate: the wheel embeds headers and catalogue built together.

`SDOT_KERNELS`: `auto` (catalogue, otherwise compile -- the default), `catalogue` (never compile:
an absence is an error that names the kernel), `compile` (always compile, ignore the
catalogue -- which is what a development checkout does anyway, it has no catalogue).
"""
from pathlib import Path
import hashlib
import ctypes
import json
import os

from ..util.encode_base_62 import encode_base_62
from .. import env

# tags -> directory (containing `catalogue.json` and the library), in registration order
_catalogues = {}
_loaded_libs = {}
_entries = {}  # tag -> dict( key -> entry point name )


def policy() -> str:
    v = env.var( "KERNELS", "auto" ).strip().lower()
    if v not in ( "auto", "catalogue", "compile" ):
        raise ValueError( f"LOOM_KERNELS={ v !r}: expected auto, catalogue or compile" )
    return v


def key( source: str, sources, tag: str ) -> str:
    """A kernel's key in the catalogue `tag`: source + domain sources (paths as
    given, relative to the C++ roots) + tag. In a canonical form, so that the same
    kernel has the same key at recording, at compilation and at call time."""
    canon = json.dumps( [ [ str( p ), [ [ str( k ), str( v ) ] for k, v in d ] ] for p, d in sources ], sort_keys = True )
    h = hashlib.sha256( f"{ source }|{ canon }|{ tag }".encode() ).hexdigest()
    return encode_base_62( h )[ :24 ]


def entry_symbol( k: str ) -> str:
    return f"sdot_ffi_{ k }"


def register_catalogue( root ):
    """`root/<tag>/catalogue.json` + `root/<tag>/libsdot_kernels.<so|dylib>` for each tag
    present. Called by the package that ships the catalogue (`sdot/__init__.py`)."""
    root = Path( root )
    if not root.is_dir():
        return
    for d in sorted( root.iterdir() ):
        if ( d / "catalogue.json" ).is_file():
            _catalogues[ d.name ] = d


def registered_tags() -> list:
    return list( _catalogues )


def _entries_of( tag ):
    if tag not in _entries:
        _entries[ tag ] = json.loads( ( _catalogues[ tag ] / "catalogue.json" ).read_text() ).get( "kernels", {} )
    return _entries[ tag ]


def _library_of( tag ):
    if tag not in _loaded_libs:
        d = _catalogues[ tag ]
        libs = [ p for p in d.iterdir() if p.name.startswith( "libsdot_kernels" ) ]
        if not libs:
            raise RuntimeError( f"sdot: catalogue `{ tag }` has no library in { d }" )
        _loaded_libs[ tag ] = ctypes.CDLL( str( libs[ 0 ] ) )
    return _loaded_libs[ tag ]


def lookup( source: str, sources, device ):
    """`( library, entry point )` of the precompiled kernel that serves this source on this device, or
    None. Tags are tried from richest to poorest (`device.catalogue_tags`)."""
    if policy() == "compile" or not _catalogues:
        return None
    for tag in device.catalogue_tags():
        if tag not in _catalogues:
            continue
        entries = _entries_of( tag )
        k = key( source, sources, tag )
        if k in entries:
            lib = _library_of( tag )
            return lib, getattr( lib, entries[ k ] )
    return None


# ── recording ───────────────────────────────────────────────────────────────

def record( source: str, sources, device ):
    """Drops this source into `SDOT_CATALOGUE_RECORD` (if set), with what is needed to
    recompile it elsewhere: `<h>.<cpp|cu>`, `<h>.json` (device, domain sources), and the
    generated headers of the current build (copied whole, they are small and deterministic)."""
    root = env.var( "CATALOGUE_RECORD" )
    if not root:
        return
    from . import build_dir
    from .generated_headers import include_root
    import shutil

    root = Path( root ) / device.catalogue_kind()
    root.mkdir( parents = True, exist_ok = True )
    h = encode_base_62( hashlib.sha256( f"{ source }|{ list( sources ) }".encode() ).hexdigest() )[ :24 ]
    suffix = ".cu" if device.catalogue_kind() == "cuda" else ".cpp"
    src = root / f"{ h }{ suffix }"
    if not src.exists():
        src.write_text( source )
        ( root / f"{ h }.json" ).write_text( json.dumps( { "sources": [ list( s ) for s in sources ] }, indent = 1 ) )
    # the generated headers: the same tree, merged (write-if-changed, as originally)
    gen_src = include_root()
    gen_dst = Path( env.var( "CATALOGUE_RECORD" ) ) / "include"
    for p in gen_src.rglob( "*.h" ):
        q = gen_dst / p.relative_to( gen_src )
        if not ( q.exists() and q.read_bytes() == p.read_bytes() ):
            q.parent.mkdir( parents = True, exist_ok = True )
            shutil.copyfile( p, q )


# ── compiling a catalogue ───────────────────────────────────────────────────

def build( record_root, out_root, device, tag: str ):
    """Compiles everything recorded for this kind of device (`record_root/<cpu|cuda>`) with
    `device`'s compiler (its variant / its architectures), links the whole -- runtime included --
    into `out_root/<tag>/libsdot_kernels.so`, and writes `catalogue.json`. Returns the number of kernels."""
    from .build import Build, runtime_sources
    from ..drivers.JaxFfi import ffi_include_dir, _resolve_source
    from . import build_dir

    record_root = Path( record_root ).resolve()  # ninja runs from the build directory
    kind_root = record_root / device.catalogue_kind()
    out = Path( out_root ).resolve() / tag
    out.mkdir( parents = True, exist_ok = True )
    generated = record_root / "include"

    kernels = {}
    with Build( device ) as b:
        extra = [ "-isystem", ffi_include_dir(), "-I", str( generated ) ]
        objects = [ b.object( s ) for s in runtime_sources() ]
        for src in sorted( kind_root.glob( "*.c*" ) ):
            meta = json.loads( src.with_suffix( ".json" ).read_text() )
            sources = [ ( p, [ tuple( kv ) for kv in d ] ) for p, d in meta[ "sources" ] ]
            k = key( src.read_text(), sources, tag )
            objects.append( b.object( src, { "SDOT_FFI_ENTRY": entry_symbol( k ) }, extra_flags = extra ) )
            objects += [ b.object( _resolve_source( p ), dict( d ) ) for p, d in sources ]
            kernels[ k ] = entry_symbol( k )
        lib = b.shared_library( out / device.compiler.library_file_name( "sdot_kernels" ), list( dict.fromkeys( objects ) ) )
        b.run( [ lib ] )

    ( out / "catalogue.json" ).write_text( json.dumps( {
        "tag": tag, "kind": device.catalogue_kind(), "signature": device.compiler.build_signature, "kernels": kernels,
    }, indent = 1 ) )
    return len( kernels )
