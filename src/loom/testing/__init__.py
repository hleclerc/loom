"""Framework de test/benchmark Python, miroir de tests/cpp/test_main.h.

Un fichier ressemble à :

    from loom.testing import test, bench, Param     # ( + check_grad si besoin )

    if test( "my test", [ "[fast]" ] ):
        assert 0 == 0

    if p := bench( "my bench", nb_diracs = Param( 1000, help = "nb diracs" ) ):
        p.results[ "cost" ] = run_bench( p.nb_diracs )   # -> result.yaml, systematically
        # p.out_dir / "plot.png"  -- write ad hoc files there too if useful

`test`/`bench` fonctionnent en deux phases, pilotées par le runner (loom.cli.main) :

* phase de collecte -> enregistre l'entrée (test OU bench, avec ses éventuels
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
        self.kind   = kind             # "test" | "bench"
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

    Read fresh on every call, NOT a module-load-time snapshot: test/bench run
    in-process (unlike experiment/benchmark, which get a fresh subprocess per
    run), so os.environ can still be gaining SDOT_ARG_* entries after this
    module was first imported -- e.g. once discovery has resolved which
    --params exist to parse in the first place.
    """
    return { k[ 9: ]: v for k, v in os.environ.items() if k.startswith( "SDOT_ARG_" ) }


def _normalize_tags( tags ):
    if tags is None:
        return []
    if isinstance( tags, str ):
        return [ tags ]
    return list( tags )


def test( name, tags = None, **params: Param ):
    """Register a test (collect) or return its parsed Args if selected (run)."""
    return _register( "test", name, tags, params )


def bench( name, tags = None, **params: Param ):
    """Register a benchmark (collect) or return its parsed Args if selected (run)."""
    return _register( "bench", name, tags, params )


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
