"""Les bibliothèques C++ EXTERNES, en-têtes seuls, que la chaîne de compilation va chercher
elle-même : un paquet bâti sur loom en déclare une (`register_external`), et son archive est
téléchargée UNE FOIS dans le cache utilisateur (`cache_root() / "ext"`) au premier noyau qui
compile -- puis son répertoire entre dans les `-I` de toute compilation (`include_dirs`).

C'est ce qui rend un `#include <Eigen/SparseCholesky>` ou `<amgcl/...>` vrai sur toute machine,
sans paquet système ni étape à la main : sdot en dépend pour les solveurs linéaires de son
transport (`sdot/otplan/Lineaire.cpp`), et un catalogue de noyaux se bâtit avec ces mêmes
versions, épinglées -- une archive et son SHA-256, pas une branche.

Sans réseau (`SDOT_EXTERNALS=0`, ou un téléchargement qui échoue), l'externe manque et on le dit
UNE fois : la source qui l'attend se garde par `__has_include` et fait avec ce qu'elle a.
`SDOT_EXT_DIR` place le cache ailleurs (une machine de build, un montage partagé).
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
    """Déclare une bibliothèque : `url` d'une archive (`.tar.gz` / `.zip`) dont le premier niveau
    est retiré, `sha256` de l'archive, `include` le sous-répertoire à mettre sur le chemin
    d'inclusion (la racine par défaut). Déclarer deux fois la même est sans effet."""
    if name not in _externals:
        _externals[ name ] = External( name, version, url, sha256, include )


def ext_root() -> Path:
    from . import cache_root
    override = os.getenv( "SDOT_EXT_DIR" )
    return Path( override ).expanduser() if override else cache_root() / "ext"


def _fetch( ext: External ):
    """L'archive, vérifiée, dépliée dans `ext.root` -- atomiquement : un répertoire voisin puis un
    renommage, pour qu'un second processus ne voie jamais un dépliage à moitié fait."""
    ext_root().mkdir( parents = True, exist_ok = True )
    with tempfile.TemporaryDirectory( dir = ext_root(), prefix = f".{ ext.name }-" ) as tmp:
        tmp = Path( tmp )
        archive = tmp / "archive"
        with urllib.request.urlopen( ext.url, timeout = 120 ) as r, open( archive, "wb" ) as f:
            shutil.copyfileobj( r, f )
        digest = hashlib.sha256( archive.read_bytes() ).hexdigest()
        if digest != ext.sha256:
            raise RuntimeError( f"{ ext.name } { ext.version } : SHA-256 inattendu ({ digest }, attendu { ext.sha256 })" )
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
            if not ext.root.exists():                  # pas un autre processus : une vraie erreur
                shutil.move( str( top ), str( ext.root ) )


def ensure( ext: External ) -> bool:
    """Vrai si `ext.include_dir` est là (déjà, ou après téléchargement)."""
    if ext.include_dir.is_dir():
        return True
    if os.getenv( "SDOT_EXTERNALS", "1" ) in ( "0", "no", "off" ):
        return False
    try:
        _fetch( ext )
    except Exception as e:                                # réseau, disque, somme : on le dit, et on continue sans
        if not ext._reported:
            ext._reported = True
            print( f"loom: { ext.name } { ext.version } non disponible ({ e }) -- les noyaux qui l'attendent feront sans",
                   file = sys.stderr )
        return False
    return ext.include_dir.is_dir()


def external_include_dirs() -> list:
    """Les `-I` des externes déclarés qui sont là (téléchargés au besoin)."""
    return [ ext.include_dir for ext in _externals.values() if ensure( ext ) ]
