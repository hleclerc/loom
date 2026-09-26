"""Le graphe de compilation : des unités (`.cpp` + defines -> `.o`), des bibliothèques et des
exécutables qui les lient, et ninja pour ne refaire que ce qui a changé.

Pourquoi ninja et pas un hachage maison : un noyau n'est pas un fichier mais une FERMETURE
d'en-têtes, et seul le compilateur la connaît exactement (`-MMD`). ninja lit ces depfiles, compare
les dates, et rebâtit une unité si et seulement si l'un de SES en-têtes a bougé -- là où le hachage
global de tout l'arbre (`cpp_sources_hash`, avant) rebâtissait chaque noyau pour un commentaire
touché dans un en-tête qu'il n'incluait pas. Mesuré : 88 ms de graphe par noyau contre ~8 s de
compilation, soit 1 % -- un bon marché pour la seule chose difficile.

= DEUX GRAPHES, parce qu'il y a deux sortes de cibles

Le critère n'est pas « commun / spécifique » mais AVOIR DES DÉPENDANTS OU NON.

  * Le graphe COMMUN (`build/`) porte ce qui en a : la bibliothèque runtime (la file de threads,
    une par signature de compilateur) et les UNITÉS DE DOMAINE -- une source compilée une fois par
    (source, defines, compilateur) et liée par tous les noyaux qui la nomment. Il est permanent, il
    a un verrou, et il ne grandit plus avec le nombre de noyaux.
  * Le graphe PROPRE d'un noyau (`build/noyaux/<cible>/`) porte sa source engendrée, son objet et
    sa bibliothèque : un noyau, zéro dépendant, et un nom qui est déjà un hachage de son contenu.
    Deux arêtes, son propre `build.ninja`, son propre verrou, son propre journal de dépendances.

Ce que ça change, et c'est le point : OUBLIER UN NOYAU DEVIENT UNE SUPPRESSION. `rm -rf` de son
répertoire, sans mutation de graphe, sans verrou, sans arête orpheline possible -- là où un
manifeste unique ne faisait que croître (5104 arêtes, 3506 bibliothèques, 2,7 Go mesurés avant
cette partition). Et le verrou se décompose : il ne sérialise plus que la phase « le runtime est-il
à jour ? », de sorte que deux processus qui compilent deux noyaux DIFFÉRENTS ne s'attendent plus.
Deux processus sur le MÊME noyau s'attendent encore, ce qui est le seul cas où il le faut.

Les deux phases sont séquentielles et dans cet ordre : le commun d'abord (il peut rafraîchir le
runtime), le noyau ensuite, qui voit alors une entrée plus récente et relie. Les artefacts communs
entrent dans le graphe du noyau comme de simples FICHIERS D'ENTRÉE, sans règle -- c'est ce qui
empêche deux répertoires de noyaux de décider indépendamment de rebâtir le runtime et de se battre
sur le même fichier de sortie. (C'est aussi pourquoi ce n'est pas un `subninja`.)

Les commandes viennent du compilateur du device (`Compiler.commands`) : c'est lui qui sait compiler
un `.cpp` ou un `.cu`, et il les donne sous forme de GABARITS D'ARGV à trous nommés (`{in}`,
`{out}`, `{depfile}`, ...). Ce module en fait la SYNTAXE ninja (`_en_regle_ninja`) ; le même gabarit
s'exécuterait directement ou s'écrirait dans un `compile_commands.json`. Les flags appartiennent au
compilateur, la syntaxe à qui rend la commande -- et aucune des deux couches ne connaît l'autre.
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
from .. import env


def ninja_path() -> str:
    p = env.var( "NINJA" ) or shutil.which( "ninja" )
    if p is None:
        # the `ninja` pip package ships the binary next to the interpreter's scripts
        candidate = Path( sys.executable ).parent / "ninja"
        if candidate.is_file():
            p = str( candidate )
    if p is None:
        raise RuntimeError( "loom : `ninja` introuvable (pip install ninja, ou LOOM_NINJA=/chemin/ninja)" )
    return p


def kernels_root() -> Path:
    """Où vit un répertoire par noyau -- chacun s'efface d'un bloc (voir la docstring du module)."""
    return build_dir() / "noyaux"


def _short_hash( *parts ) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update( str( p ).encode() )
        h.update( b"\0" )
    return encode_base_62( h.hexdigest() )[ :10 ]


# les trous d'un gabarit d'argv (`Compiler.commands`), rendus en variables ninja. `{depfile}` est
# `$out.d` : ninja veut le depfile à côté de la sortie, et le déclare dans la règle.
_TROUS_NINJA = {
    "{in}": "$in", "{out}": "$out", "{depfile}": "$out.d",
    "{includes}": "$includes", "{defines}": "$defines", "{extra}": "$extra",
    "{libs}": "$libs", "{soname}": "$soname",
}


def _en_regle_ninja( argv ) -> str:
    """Un gabarit d'argv en ligne de commande ninja. Les arguments qui ne sont pas des trous
    passent tels quels."""
    return " ".join( _TROUS_NINJA.get( a, a ) for a in argv )


def _ninja_escape( s: str ) -> str:
    return str( s ).replace( "$", "$$" ).replace( " ", "$ " ).replace( ":", "$:" )


class Manifest:
    """Le graphe sur disque. `rules` : nom -> { command, depfile?, deps? } ; `edges` : sortie ->
    { rule, inputs, implicit, vars }. Chargé et réécrit sous le verrou."""

    def __init__( self, root: Path ):
        self.root = Path( root )
        self.path = self.root / "ninja" / "manifest.json"
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

    def prune( self, garder = () ) -> int:
        """Retire les arêtes dont la SORTIE n'existe plus, `garder` exceptée (ce que ce build vient
        de déclarer et s'apprête justement à bâtir).

        Toujours sûr : si quelqu'un a encore besoin d'une arête retirée, le processus qui la déclare
        la remettra. C'est ce qui fait qu'effacer un répertoire de noyau -- ou un artefact à la
        main -- NETTOIE le graphe au lieu de le laisser enfler, et c'est ce qui manquait quand tout
        vivait dans un manifeste unique."""
        garder = { str( g ) for g in garder }
        vivantes = { out: e for out, e in self.edges.items()
                     if out in garder or Path( out ).exists() }
        retirees = len( self.edges ) - len( vivantes )
        self.edges = vivantes
        return retirees

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
    """Un verrou de fichier autour d'un graphe et de son ninja (POSIX ; no-op ailleurs).

    Un par graphe : celui du commun sérialise « le runtime est-il à jour ? », celui d'un noyau
    sérialise la compilation de CE noyau. Deux noyaux différents ne se croisent plus."""

    def __init__( self, root: Path ):
        self.path = Path( root ) / "ninja" / "lock"
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
    """`LOOM_FORCE_BUILD` : rebâtir les cibles demandées même si ninja les croit à jour -- pour
    quand la chaîne a changé d'une façon que les depfiles ne voient pas (un flag, un outil)."""
    return env.flag( "FORCE_BUILD" )


class _Graphe:
    """Un graphe dans un répertoire : ce que CE build y déclare, plus ce qu'il faut pour le fondre
    dans le manifeste qui s'y trouve déjà et lancer ninja dessus.

    Les déclarations restent ici et ne sont pas écrites tout de suite : le manifeste sur disque
    n'est lu et réécrit que dans `run`, sous le verrou -- donc les arêtes qu'un autre processus
    aurait ajoutées entre-temps ne sont pas perdues (avant, un verrou pris pour toute la durée du
    build rendait la question sans objet, au prix d'une sérialisation totale)."""

    def __init__( self, root, sig: str, commands: dict ):
        self.root = Path( root )
        self.sig = sig
        self.rules = { f"{ name }_{ sig }": ( _en_regle_ninja( argv ), depfile, description )
                       for name, ( argv, depfile, description ) in commands.items() }
        self.edges = {}

    def add_edge( self, out: Path, rule: str, inputs: list, implicit: list = (), **variables ):
        self.edges[ str( out ) ] = dict( rule = rule, inputs = list( inputs ),
                                         implicit = list( implicit ), variables = variables )

    def run( self, targets: list ):
        """Fond nos déclarations dans le manifeste du répertoire, puis bâtit `targets`."""
        targets = [ str( t ) for t in targets ]
        if not targets:
            return
        with _Lock( self.root ):
            manifest = Manifest( self.root )
            for name, ( command, depfile, description ) in self.rules.items():
                manifest.add_rule( name, command, depfile, description )
            for out, e in self.edges.items():
                manifest.add_edge( out, e[ "rule" ], e[ "inputs" ], e[ "implicit" ], **e[ "variables" ] )
            manifest.prune( garder = self.edges )
            manifest.save()
            ninja_file = manifest.write_ninja()

            if force_build():
                for t in targets:
                    Path( t ).unlink( missing_ok = True )
                    for i in manifest.edges.get( t, {} ).get( "inputs", [] ):
                        if i.endswith( ".o" ):
                            Path( i ).unlink( missing_ok = True )

            # `LOOM_BUILD_JOBS` : le parallélisme (défaut : celui de ninja, tous les coeurs) -- un
            # catalogue CUDA compile cent unités d'un gigaoctet chacune, on ne les veut pas toutes
            # en même temps sur une machine partagée
            jobs = env.var( "BUILD_JOBS" )
            cmd = [ ninja_path(), "-C", str( self.root ), "-f", str( ninja_file ),
                    *( [ "-j", jobs ] if jobs else [] ), *targets ]
            r = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
            out = r.stdout or ""
            # ninja's own line for a no-op build is noise; a real compilation is worth seeing
            if "no work to do" not in out:
                print( out, end = "", flush = True )
            if r.returncode:
                raise RuntimeError( f"ninja failed ({ r.returncode }) on { targets }" )


class Build:
    """Une session de construction : on y déclare des unités, des bibliothèques, des exécutables,
    puis `run( targets )` fait le nécessaire.

    `work_dir` est le répertoire PROPRE de ce qu'on bâtit -- celui d'un noyau, effaçable d'un bloc.
    Sans lui, tout va dans le graphe commun : c'est ce que font le catalogue (un seul gros lien dans
    un répertoire jetable) et les tests C++."""

    def __init__( self, device, work_dir = None ):
        self.device = device
        self.compiler = device.compiler
        self.root = build_dir()
        self.sig = _short_hash( self.compiler.build_signature )
        commands = self.compiler.commands()
        self.commun = _Graphe( self.root, self.sig, commands )
        self.propre = _Graphe( work_dir, self.sig, commands ) if work_dir is not None else self.commun

    def close( self ):
        pass

    def __enter__( self ):
        return self

    def __exit__( self, *exc ):
        self.close()

    # ── declarations ─────────────────────────────────────────────────────────
    def _rule( self, name ) -> str:
        return f"{ name }_{ self.sig }"

    def object( self, src: Path, defines: dict = None, extra_flags: list = (), partage = True ) -> Path:
        """L'objet d'une source compilée avec ces `defines` -- une unité par (source, defines,
        compilateur).

        `partage` (le défaut) : l'unité a des dépendants, le même `.o` sert à tous les noyaux qui la
        demandent, donc elle vit dans le graphe COMMUN. `partage = False` : l'objet n'est à personne
        d'autre (la source engendrée d'un noyau), il vit dans le répertoire propre et part avec lui."""
        src = Path( src ).resolve()
        defines = dict( defines or {} )
        extra_flags = list( extra_flags )
        graphe = self.commun if partage else self.propre
        nom = f"{ src.stem }_{ _short_hash( self.sig, src, sorted( defines.items() ), extra_flags ) }.o"
        obj = ( graphe.root / "obj" / nom ) if graphe is self.commun else ( graphe.root / nom )
        graphe.add_edge(
            obj, self._rule( self.compiler.rule_for( src ) ), [ src ],
            includes = " ".join( f"-I { _ninja_escape( d ) }" for d in include_dirs() ),
            defines  = " ".join( f"-D{ k }={ v }" if v is not None else f"-D{ k }" for k, v in defines.items() ),
            extra    = " ".join( extra_flags ),
        )
        return obj

    def shared_library( self, out: Path, objects: list, libraries: list = (), partage = False ) -> Path:
        """`out` = un `.so` lié des `objects` et des `libraries` (des `.so` du graphe : le
        répertoire de chacune entre dans le rpath).

        `partage = True` pour une bibliothèque qui a des dépendants (le runtime) : elle va dans le
        graphe commun."""
        out = Path( out )
        graphe = self.commun if partage else self.propre
        graphe.add_edge( out, self._rule( "link_shared" ), objects, implicit = libraries,
                         libs = self.compiler.link_libraries( libraries ),
                         soname = self.compiler.soname_flags( out ) )
        return out

    def executable( self, out: Path, objects: list, libraries: list = () ) -> Path:
        out = Path( out )
        self.propre.add_edge( out, self._rule( "link_executable" ), objects, implicit = libraries,
                              libs = self.compiler.link_libraries( libraries ) )
        return out

    def runtime_library( self ) -> Path:
        """`libloom_runtime` pour ce compilateur : la file de threads du processus, et tout ce que
        les noyaux partagent sans avoir à le recompiler. Une par signature de compilateur, dans le
        graphe commun -- c'est l'exemple même d'une cible à dépendants."""
        objects = [ self.object( src ) for src in runtime_sources() ]
        return self.shared_library(
            self.commun.root / "runtime" / self.compiler.library_file_name( f"loom_runtime_{ self.sig }" ),
            objects, partage = True )

    # ── run ──────────────────────────────────────────────────────────────────
    def run( self, targets: list ):
        """Bâtit `targets` (et ce dont elles dépendent), et rien d'autre.

        Deux phases, dans cet ordre : le COMMUN d'abord -- il peut rafraîchir le runtime ou une
        unité de domaine -- puis le PROPRE, qui voit alors une entrée plus récente et relie. Les
        artefacts communs entrent dans le graphe propre comme de simples fichiers d'entrée, sans
        règle : aucun répertoire de noyau ne peut décider de rebâtir le runtime."""
        if self.propre is self.commun:
            self.commun.run( targets )
            return
        self.commun.run( list( self.commun.edges ) )
        self.propre.run( targets )


def runtime_sources() -> list:
    """Les sources de `libloom_runtime` (`loom/cpp/runtime/*.cpp`), depuis un checkout ou un wheel."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        root = dev_root / "loom" / "cpp" / "runtime"
    else:
        root = Path( __file__ ).resolve().parents[ 1 ] / "_cpp" / "runtime"
    return sorted( root.glob( "*.cpp" ) )
