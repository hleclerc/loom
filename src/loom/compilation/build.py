"""Le graphe de compilation : des unités (`.cpp` + defines -> `.o`), des bibliothèques et des
exécutables qui les lient, et ninja pour ne refaire que ce qui a changé.

Pourquoi ninja et pas un hachage maison : un noyau n'est pas un fichier mais une FERMETURE
d'en-têtes, et seul le compilateur la connaît exactement (`-MMD`). ninja lit ces depfiles, compare
les dates, et rebâtit une unité si et seulement si l'un de SES en-têtes a bougé -- là où le hachage
global de tout l'arbre (`cpp_sources_hash`, avant) rebâtissait chaque noyau pour un commentaire
touché dans un en-tête qu'il n'incluait pas. Et un graphe est ce qu'il faut pour les unités
partagées : la bibliothèque runtime (la file de threads, une par processus), demain les sources
de domaine compilées une fois par configuration et liées dans plusieurs noyaux.

Le graphe VIT SUR DISQUE (`build/ninja/manifest.json`) : chaque processus y ajoute ce qu'il demande,
réécrit `build/build.ninja` en entier et lance `ninja` sur ses cibles. Les rebuilds d'un processus
à l'autre restent donc exacts (ninja garde ses `.ninja_deps` / `.ninja_log` dans `build/`). Un
verrou de fichier sérialise les processus : deux ninja concurrents dans un même répertoire se
marcheraient dessus.

Les règles viennent du compilateur du device (`Compiler.ninja_rules`) : c'est lui qui sait
compiler un `.cpp` (et demain un `.cu`). Ce module ne connaît que des chemins et des commandes.
"""
from pathlib import Path
import subprocess
import hashlib
import shutil
import json
import sys
import os

from . import build_dir, include_dirs, _dev_repo_root
from ..util.encode_base_62 import encode_base_62


def ninja_path() -> str:
    p = os.getenv( "SDOT_NINJA" ) or shutil.which( "ninja" )
    if p is None:
        # the `ninja` pip package ships the binary next to the interpreter's scripts
        candidate = Path( sys.executable ).parent / "ninja"
        if candidate.is_file():
            p = str( candidate )
    if p is None:
        raise RuntimeError( "sdot: `ninja` introuvable (pip install ninja, ou SDOT_NINJA=/chemin/ninja)" )
    return p


def _short_hash( *parts ) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update( str( p ).encode() )
        h.update( b"\0" )
    return encode_base_62( h.hexdigest() )[ :10 ]


def _ninja_escape( s: str ) -> str:
    return str( s ).replace( "$", "$$" ).replace( " ", "$ " ).replace( ":", "$:" )


class Manifest:
    """Le graphe sur disque. `rules` : nom -> { command, depfile?, deps? } ; `edges` : sortie ->
    { rule, inputs, implicit, vars }. Chargé et réécrit sous le verrou."""

    def __init__( self, root: Path ):
        self.root = root
        self.path = root / "ninja" / "manifest.json"
        self.rules = {}
        self.edges = {}
        if self.path.is_file():
            data = json.loads( self.path.read_text() )
            self.rules = data.get( "rules", {} )
            self.edges = data.get( "edges", {} )

    def add_rule( self, name: str, command: str, depfile: bool, description: str ):
        rule = { "command": command, "description": description }
        if depfile:
            rule[ "depfile" ] = "$out.d"
            rule[ "deps" ] = "gcc"
        self.rules[ name ] = rule

    def add_edge( self, out: Path, rule: str, inputs: list, implicit: list = (), **variables ):
        self.edges[ str( out ) ] = { "rule": rule, "inputs": [ str( i ) for i in inputs ],
                                     "implicit": [ str( i ) for i in implicit ],
                                     "vars": { k: str( v ) for k, v in variables.items() if v } }

    def save( self ):
        self.path.parent.mkdir( parents = True, exist_ok = True )
        tmp = self.path.with_suffix( ".json.tmp" )
        tmp.write_text( json.dumps( { "rules": self.rules, "edges": self.edges }, indent = 1 ) )
        tmp.replace( self.path )

    def write_ninja( self ) -> Path:
        lines = [ "# généré par loom.compilation.build -- ne pas éditer, voir ninja/manifest.json", "ninja_required_version = 1.3", "" ]
        for name, rule in self.rules.items():
            lines.append( f"rule { name }" )
            lines.append( f"  command = { rule[ 'command' ] }" )
            lines.append( f"  description = { rule.get( 'description', '$out' ) }" )
            if "depfile" in rule:
                lines.append( f"  depfile = { rule[ 'depfile' ] }" )
                lines.append( f"  deps = { rule[ 'deps' ] }" )
            lines.append( "" )
        for out, edge in self.edges.items():
            ins = " ".join( _ninja_escape( i ) for i in edge[ "inputs" ] )
            imp = " ".join( _ninja_escape( i ) for i in edge[ "implicit" ] )
            lines.append( f"build { _ninja_escape( out ) }: { edge[ 'rule' ] } { ins }" + ( f" | { imp }" if imp else "" ) )
            for k, v in edge[ "vars" ].items():
                lines.append( f"  { k } = { v }" )
        path = self.root / "build.ninja"
        content = "\n".join( lines ) + "\n"
        if not ( path.exists() and path.read_text() == content ):
            path.write_text( content )
        return path


class _Lock:
    """Un verrou de fichier autour du graphe et de ninja (POSIX ; no-op ailleurs)."""

    def __init__( self, root: Path ):
        self.path = root / "ninja" / "lock"
        self.fd = None

    def __enter__( self ):
        self.path.parent.mkdir( parents = True, exist_ok = True )
        self.fd = os.open( self.path, os.O_RDWR | os.O_CREAT, 0o644 )
        try:
            import fcntl
            fcntl.flock( self.fd, fcntl.LOCK_EX )
        except ImportError:
            pass
        return self

    def __exit__( self, *exc ):
        try:
            import fcntl
            fcntl.flock( self.fd, fcntl.LOCK_UN )
        except ImportError:
            pass
        os.close( self.fd )


def force_build() -> bool:
    """`SDOT_FORCE_BUILD` : rebâtir les cibles demandées même si ninja les croit à jour -- pour
    quand la chaîne a changé d'une façon que les depfiles ne voient pas (un flag, un outil)."""
    v = ( os.getenv( "SDOT_FORCE_BUILD" ) or "" ).strip().lower()
    return v not in ( "", "0", "false", "no", "off" )


class Build:
    """Une session de construction : on y déclare des unités, des bibliothèques, des exécutables,
    puis `run( targets )` fait le nécessaire. Tout ce qui est déclaré s'ajoute au manifeste du
    répertoire de build."""

    def __init__( self, device ):
        self.device = device
        self.compiler = device.compiler
        self.root = build_dir()
        self.sig = _short_hash( self.compiler.build_signature )
        self._lock = _Lock( self.root )
        self._lock.__enter__()
        self.manifest = Manifest( self.root )
        for name, ( command, depfile, description ) in self.compiler.ninja_rules().items():
            self.manifest.add_rule( f"{ name }_{ self.sig }", command, depfile, description )

    def close( self ):
        self._lock.__exit__( None, None, None )

    def __enter__( self ):
        return self

    def __exit__( self, *exc ):
        self.close()

    # ── declarations ─────────────────────────────────────────────────────────
    def _rule( self, name ) -> str:
        return f"{ name }_{ self.sig }"

    def object( self, src: Path, defines: dict = None, extra_flags: list = () ) -> Path:
        """L'objet d'une source compilée avec ces `defines` -- une unité par (source, defines,
        compilateur). Le même `.o` sert à tous les noyaux qui le demandent."""
        src = Path( src ).resolve()
        defines = dict( defines or {} )
        extra_flags = list( extra_flags )
        obj = self.root / "obj" / f"{ src.stem }_{ _short_hash( self.sig, src, sorted( defines.items() ), extra_flags ) }.o"
        self.manifest.add_edge(
            obj, self._rule( self.compiler.rule_for( src ) ), [ src ],
            includes = " ".join( f"-I { _ninja_escape( d ) }" for d in include_dirs() ),
            defines  = " ".join( f"-D{ k }={ v }" if v is not None else f"-D{ k }" for k, v in defines.items() ),
            extra    = " ".join( extra_flags ),
        )
        return obj

    def shared_library( self, out: Path, objects: list, libraries: list = () ) -> Path:
        """`out` = un `.so` lié des `objects` et des `libraries` (des `.so` du graphe : le
        répertoire de chacune entre dans le rpath)."""
        out = Path( out )
        self.manifest.add_edge( out, self._rule( "link_shared" ), objects, implicit = libraries,
                                libs = self.compiler.link_libraries( libraries ),
                                soname = self.compiler.soname_flags( out ) )
        return out

    def executable( self, out: Path, objects: list, libraries: list = () ) -> Path:
        out = Path( out )
        self.manifest.add_edge( out, self._rule( "link_executable" ), objects, implicit = libraries,
                                libs = self.compiler.link_libraries( libraries ) )
        return out

    def runtime_library( self ) -> Path:
        """`libloom_runtime` pour ce compilateur : la file de threads du processus, et tout ce
        que les noyaux partagent sans avoir à le recompiler. Une par signature de compilateur."""
        objects = [ self.object( src ) for src in runtime_sources() ]
        return self.shared_library( self.root / "runtime" / self.compiler.library_file_name( f"loom_runtime_{ self.sig }" ), objects )

    # ── run ──────────────────────────────────────────────────────────────────
    def run( self, targets: list ):
        """Bâtit `targets` (et ce dont elles dépendent), et rien d'autre."""
        self.manifest.save()
        ninja_file = self.manifest.write_ninja()
        targets = [ str( t ) for t in targets ]
        if force_build():
            for t in targets:
                Path( t ).unlink( missing_ok = True )
                for i in self.manifest.edges.get( t, {} ).get( "inputs", [] ):
                    if i.endswith( ".o" ):
                        Path( i ).unlink( missing_ok = True )
        cmd = [ ninja_path(), "-C", str( self.root ), "-f", str( ninja_file ), *targets ]
        r = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
        out = r.stdout or ""
        # ninja's own line for a no-op build is noise; a real compilation is worth seeing
        if "no work to do" not in out:
            print( out, end = "", flush = True )
        if r.returncode:
            raise RuntimeError( f"ninja failed ({ r.returncode }) on { targets }" )


def runtime_sources() -> list:
    """Les sources de `libloom_runtime` (`loom/cpp/runtime/*.cpp`), depuis un checkout ou un wheel."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        root = dev_root / "loom" / "cpp" / "runtime"
    else:
        root = Path( __file__ ).resolve().parents[ 1 ] / "_cpp" / "runtime"
    return sorted( root.glob( "*.cpp" ) )
