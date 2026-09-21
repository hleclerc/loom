"""Le CATALOGUE : des noyaux précompilés, livrés dans un wheel, pour qu'un usage standard n'ait
rien à compiler.

Trois temps :

  1. ENREGISTRER (`SDOT_CATALOGUE_RECORD=dir`) : sur une machine de développement, on lance ce
     qui provoque les compilations voulues (les tests, `python -m sdot.catalogue`) ; chaque source
     généré qui passe par `compile_and_register` est déposé dans `dir` avec ce qu'il faut pour le
     recompiler ailleurs (le device, les sources de domaine, les en-têtes générés). Un GPU n'est
     nécessaire qu'ici, pour EXÉCUTER les appels -- pas pour compiler.

  2. COMPILER (`scripts/build_catalogue.py compile`) : pour une variante (`x86-64-v3`, ou une liste
     d'architectures CUDA), chaque source enregistré devient un objet, tous sont liés en UNE
     bibliothèque (`libsdot_kernels.so`, avec le runtime dedans) et `catalogue.json` dit quel
     point d'entrée sert quelle clé. C'est ce que fait l'intégration continue, par plateforme.

  3. CHERCHER (à l'exécution) : la clé d'un noyau est le hachage de son source, de ses sources de
     domaine et de l'ÉTIQUETTE du catalogue (`cpu-x86-64-v3`, `cuda`) -- rien de la machine ni du
     compilateur, donc la même clé chez tout le monde. À l'import, le paquet enregistre son
     répertoire de catalogues (`register_catalogue`) ; à l'appel, `lookup` essaie les étiquettes
     que la machine peut charger, de la plus riche à la plus pauvre, et rend le point d'entrée.
     Rien à parcourir, rien à valider : le wheel embarque en-têtes et catalogue bâtis ensemble.

`SDOT_KERNELS` : `auto` (catalogue, sinon compiler -- le défaut), `catalogue` (jamais compiler :
une absence est une erreur qui nomme le noyau), `atelier` (toujours compiler, ignorer le
catalogue -- ce que fait un checkout de développement de toute façon, il n'a pas de catalogue).
"""
from pathlib import Path
import hashlib
import ctypes
import json
import os

from ..util.encode_base_62 import encode_base_62

# tags -> répertoire (contenant `catalogue.json` et la bibliothèque), dans l'ordre d'enregistrement
_catalogues = {}
_loaded_libs = {}
_entries = {}  # tag -> dict( clé -> nom du point d'entrée )


def policy() -> str:
    v = ( os.getenv( "SDOT_KERNELS" ) or "auto" ).strip().lower()
    if v not in ( "auto", "catalogue", "atelier" ):
        raise ValueError( f"SDOT_KERNELS={ v !r} : attendu auto, catalogue ou atelier" )
    return v


def key( source: str, sources, tag: str ) -> str:
    """La clé d'un noyau dans le catalogue `tag` : source + sources de domaine (chemins tels que
    donnés, relatifs aux racines C++) + étiquette. Sous une forme canonique, pour que le même
    noyau ait la même clé à l'enregistrement, à la compilation et à l'appel."""
    canon = json.dumps( [ [ str( p ), [ [ str( k ), str( v ) ] for k, v in d ] ] for p, d in sources ], sort_keys = True )
    h = hashlib.sha256( f"{ source }|{ canon }|{ tag }".encode() ).hexdigest()
    return encode_base_62( h )[ :24 ]


def entry_symbol( k: str ) -> str:
    return f"sdot_ffi_{ k }"


def register_catalogue( root ):
    """`root/<tag>/catalogue.json` + `root/<tag>/libsdot_kernels.<so|dylib>` pour chaque étiquette
    présente. Appelé par le paquet qui livre le catalogue (`sdot/__init__.py`)."""
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
            raise RuntimeError( f"sdot: catalogue `{ tag }` sans bibliothèque dans { d }" )
        _loaded_libs[ tag ] = ctypes.CDLL( str( libs[ 0 ] ) )
    return _loaded_libs[ tag ]


def lookup( source: str, sources, device ):
    """`( bibliothèque, point d'entrée )` du noyau précompilé qui sert ce source sur ce device, ou
    None. Les étiquettes sont essayées de la plus riche à la plus pauvre (`device.catalogue_tags`)."""
    if policy() == "atelier" or not _catalogues:
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


# ── enregistrement ───────────────────────────────────────────────────────────

def record( source: str, sources, device ):
    """Dépose ce source dans `SDOT_CATALOGUE_RECORD` (si mis), avec ce qu'il faut pour le
    recompiler ailleurs : `<h>.<cpp|cu>`, `<h>.json` (device, sources de domaine), et les en-têtes
    générés du build courant (copiés en entier, ils sont petits et déterministes)."""
    root = os.getenv( "SDOT_CATALOGUE_RECORD" )
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
    # les en-têtes générés : la même arborescence, fusionnée (write-if-changed, comme à l'origine)
    gen_src = include_root()
    gen_dst = Path( os.getenv( "SDOT_CATALOGUE_RECORD" ) ) / "include"
    for p in gen_src.rglob( "*.h" ):
        q = gen_dst / p.relative_to( gen_src )
        if not ( q.exists() and q.read_bytes() == p.read_bytes() ):
            q.parent.mkdir( parents = True, exist_ok = True )
            shutil.copyfile( p, q )


# ── compilation d'un catalogue ───────────────────────────────────────────────

def build( record_root, out_root, device, tag: str ):
    """Compile tout ce qui est enregistré pour ce genre de device (`record_root/<cpu|cuda>`) avec
    le compilateur de `device` (sa variante / ses architectures), lie le tout -- runtime compris --
    en `out_root/<tag>/libsdot_kernels.so`, et écrit `catalogue.json`. Rend le nombre de noyaux."""
    from .build import Build, runtime_sources
    from ..drivers.JaxFfi import ffi_include_dir, _resolve_source
    from . import build_dir

    record_root = Path( record_root ).resolve()  # ninja tourne depuis le répertoire de build
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
