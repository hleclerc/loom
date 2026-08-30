"""Framework de test/benchmark Python, miroir de tests/cpp/test_main.h.

Un fichier ressemble à :

    from loom.testing import test, bench, experiment, Param   # ( + check_grad si besoin )

    if test( "my test", [ "[fast]" ] ):
        assert 0 == 0

    if p := bench( "my bench", nb_diracs = Param( 1000, help = "nb diracs" ) ):
        p.results[ "cost" ] = run_bench( p.nb_diracs )   # -> result.yaml, systematically
        # p.out_dir / "plot.png"  -- write ad hoc files there too if useful

    if p := experiment( "my picture" ):
        viz.write_html( p.out_dir / "scene.html" )       # une SORTIE à regarder, pas une assertion

Les trois `kind` partagent tout -- enregistrement, params, `p.out_dir`, `result.yaml` -- et ne
different que par ce qu'on en ATTEND, donc par la commande qui les lance :

* `test`       -- ça doit passer. Lancé en masse, jugé PASS/FAIL.
* `bench`      -- ça doit être rapide. On en garde des CHIFFRES (`p.results`) datés, comparables.
* `experiment` -- ça doit être REGARDÉ. La sortie est un fichier (html, vtu/pvd, png) qu'un humain
                  ouvre ; rien à comparer d'une date à l'autre, donc pas de répertoire de date :
                  le chemin est STABLE, et c'est ce qui permet de garder l'onglet ouvert et de
                  recharger (voir `_entry_dirs` dans loom.cli.main).

`test`/`bench`/`experiment` fonctionnent en deux phases, pilotées par le runner (loom.cli.main) :

* phase de collecte -> enregistre l'entrée (test, bench OU experiment, avec ses éventuels
  `Param`), retourne une valeur fausse (le corps est sauté)
* phase d'exécution -> retourne un `Args` (vrai, avec les params résolus) pour
  l'entrée en cours, une valeur fausse pour toutes les autres

Le runner recharge chaque module une fois par entrée sélectionnée, de sorte que
chaque corps s'exécute isolément et qu'un échec (assert / exception) soit
capturé entrée par entrée, exactement comme le harnais C++. Une entrée est
identifiée par son SITE D'APPEL (module + ligne), pas par son nom : deux
entrées peuvent donc porter le même nom, y compris dans un même fichier.
"""
from loom.util import info, infox
from loom import new_batch_axis
from .grad_check import check_grad
import os
import sys
from pathlib import Path
from typing import Any

builtins = __import__( 'builtins' )
setattr( builtins, "infox", infox )
setattr( builtins, "info", info )
setattr( builtins, "new_batch_axis", new_batch_axis )


GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


class Param:
    """Un paramètre CLI typé, avec une valeur par défaut.

    Le type est déduit de la valeur par défaut : Param(1000) -> int, Param("BFGS") -> str.
    """
    __slots__ = ( "default", "help", "choices" )

    def __init__( self, default: Any, *, help: str = "", choices: list | None = None ):
        self.default  = default
        self.help     = help
        self.choices  = choices

    @property
    def ptype( self ):
        return type( self.default )


class Args:
    """Params résolus, accessibles comme attributs (p.nb_diracs)."""
    __slots__ = ( "__dict__", )

    def __init__( self, **kwargs: Any ):
        self.__dict__.update( kwargs )

    def __repr__( self ) -> str:
        items = ", ".join( f"{k}={v!r}" for k, v in self.__dict__.items() )
        return f"Args({items})"


class Entry:
    def __init__( self, kind, name, tags, params, file, line, module ):
        self.kind   = kind             # "test" | "bench" | "experiment"
        self.name   = name
        self.tags   = tags             # list[ str ]
        self.params = params           # dict[ str, Param ]
        self.file   = file
        self.line   = line
        self.module = module           # __name__ du module, pour reload + exécution

    @property
    def tag_text( self ):
        # même représentation que côté C++ : "[fast][core]"
        return "".join( self.tags )


all_entries: list[ Entry ] = []

# état de pilotage, positionné par le runner
PHASE_COLLECT   = 0
PHASE_RUN       = 1
test_phase      = PHASE_COLLECT
test_filter     = None                # Entry en cours d'exécution (PHASE_RUN)
current_results: dict = {}            # le dict derrière Args.results de l'entrée en cours --
                                       # le runner (loom.cli.main) le relit juste après le reimport
                                       # pour l'écrire dans result.yaml


def out_dir() -> Path:
    """Répertoire de sortie de l'entrée en cours (SDOT_OUT_DIR, mis en place par
    le runner CLI ; retombe sur ./tmp hors du runner, pour un usage ad hoc)."""
    return Path( os.environ.get( "SDOT_OUT_DIR", "tmp" ) )


def _arg_overrides() -> dict[ str, str ]:
    """Param overrides from env vars (injected by the CLI runner via SDOT_ARG_*).

    Read fresh on every call, NOT a module-load-time snapshot: entries run in
    the runner's own process, so os.environ can still be gaining SDOT_ARG_*
    entries after this module was first imported -- once discovery has
    resolved which --params exist to parse in the first place, and again from
    one combination to the next of a `--nb-diracs=1000,2000` sweep.
    """
    return { k[ 9: ]: v for k, v in os.environ.items() if k.startswith( "SDOT_ARG_" ) }


def _normalize_tags( tags ):
    if tags is None:
        return []
    if isinstance( tags, str ):
        return [ tags ]
    return list( tags )


def driver_is( name ):
    """Le driver de CETTE exécution est-il `name` ( "jax", "torch", ... ) ?

    À mettre EN TÊTE d'un fichier qui importe un backend précis, avant l'import :

        from loom.testing import driver_is
        if not driver_is( "torch" ):
            sys.exit( 0 )
        import torch

    Pourquoi un `sys.exit` et pas un `if` autour du fichier : la découverte des entrées IMPORTE
    chaque fichier candidat, donc un `import torch` en tête s'exécute même quand on tourne sous
    jax -- et sur une machine où torch est cassé (chez `lmo`, `libcusparseLt.so.0`), c'est toute
    la session qui tombe, pas seulement ce fichier. Sortir tôt est la seule chose qui empêche
    l'import d'avoir lieu.

    La réponse vient de `SDOT_DRIVER`, posé par le CLI. À défaut -- fichier lancé à la main -- on
    relit l'env par défaut de `.envs.py` ; et si même ça échoue, on répond OUI, parce que faire
    disparaître des tests en silence est pire que de laisser un import échouer bruyamment.
    """
    import os
    driver = os.environ.get( "SDOT_DRIVER" )
    if not driver:
        try:
            from loom.cli import envs
            cfg = envs.get_env()
            driver = cfg.driver if cfg else None
        except Exception:
            driver = None
    return driver is None or driver == name


def test( name, tags = None, /, **params: Param ):
    """Register a test (collect) or return its parsed Args if selected (run).

    `name`/`tags` sont POSITIONNELS-SEULEMENT : tout le reste des mots-clés appartient à
    l'entrée, donc un `Param` peut s'appeler `name` sans entrer en collision (exp_hc en a un).
    """
    return _register( "test", name, tags, params )


def bench( name, tags = None, /, **params: Param ):
    """Register a benchmark (collect) or return its parsed Args if selected (run)."""
    return _register( "bench", name, tags, params )


def experiment( name, tags = None, /, **params: Param ):
    """Register an experiment (collect) or return its parsed Args if selected (run).

    Même mécanique que `test`/`bench` (site d'appel, params, `p.out_dir`) : ce qui change est
    l'intention -- une sortie à REGARDER plutôt qu'une assertion ou un chiffre -- et donc le fait
    qu'elle ne soit pas embarquée dans les lancements en masse de `./run test`.
    """
    return _register( "experiment", name, tags, params )


def resolve_params( params: dict ) -> dict:
    """Apply SDOT_ARG_* env-var overrides to `params`, else their defaults.

    Used both by `_register` (to build the Args an active test/bench sees)
    and by the CLI (to print a resolved-values summary before running,
    without needing an active `test_filter`).
    """
    overrides = _arg_overrides()
    resolved: dict[ str, Any ] = {}
    for pname, p in params.items():
        raw = overrides.get( pname.upper() )
        if raw is not None:
            try:
                resolved[ pname ] = p.ptype( raw )
            except ( ValueError, TypeError ):
                print( f"  ⚠ Invalid --{pname}: {raw!r} (expected {p.ptype.__name__})", file = sys.stderr )
                resolved[ pname ] = p.default
        else:
            resolved[ pname ] = p.default
    return resolved


def _register( kind, name, tags, params: dict ):
    tags = _normalize_tags( tags )

    # une entrée est identifiée par son SITE D'APPEL (module + ligne), pas par son nom
    frame  = sys._getframe( 2 )  # _register -> test/bench -> caller module
    line   = frame.f_lineno
    module = frame.f_globals.get( "__name__" )

    if test_phase == PHASE_COLLECT:
        file = frame.f_code.co_filename
        all_entries.append( Entry( kind, name, tags, params, file, line, module ) )
        return False

    # phase d'exécution : seul l'appel ciblé (même module + même ligne) s'exécute. On
    # discrimine par ligne et non par nom, sinon des homonymes s'exécuteraient tous à la fois.
    if test_filter is None or module != test_filter.module or line != test_filter.line:
        return None

    current_results.clear()
    return Args( results = current_results, out_dir = out_dir(), **resolve_params( params ) )
