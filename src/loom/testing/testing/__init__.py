"""Framework de test Python, miroir de tests/cpp/test_main.h.

Un fichier de test ressemble à :

    from sdot.testing import test        # ( + check_grad si besoin )

    if test( "my test", [ "[fast]" ] ):
        assert 0 == 0

`test( name, tags )` fonctionne en deux phases, pilotées par le runner
(scripts/run_tests.py) :

* phase de collecte -> enregistre le test, retourne False (le corps est sauté)
* phase d'exécution -> retourne True uniquement pour le test en cours

Le runner recharge chaque module de test une fois par test sélectionné, de sorte
que chaque corps de test s'exécute isolément et qu'un échec (assert / exception)
soit capturé test par test, exactement comme le harnais C++.
"""
from sdot.util.info import info, infox
from sdot import new_batch_axis
from .grad_check import check_grad
import sys

builtins = __import__( 'builtins' )
setattr( builtins, "infox", infox )
setattr( builtins, "info", info )
setattr( builtins, "new_batch_axis", new_batch_axis )



GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


class Test:
    def __init__( self, name, tags, file, line, module ):
        self.name   = name
        self.tags   = tags            # list[ str ]
        self.file   = file
        self.line   = line
        self.module = module          # __name__ du module, pour reload + exécution

    @property
    def tag_text( self ):
        # même représentation que côté C++ : "[fast][core]"
        return "".join( self.tags )


all_the_tests: list[ Test ] = []

# état de pilotage, positionné par le runner
PHASE_COLLECT = 0
PHASE_RUN     = 1
test_phase    = PHASE_COLLECT
test_filter   = None                  # Test en cours d'exécution (PHASE_RUN)


def _normalize_tags( tags ):
    if tags is None:
        return []
    if isinstance( tags, str ):
        return [ tags ]
    return list( tags )


def test( name, tags = None ):
    tags = _normalize_tags( tags )

    # un test est identifié par son SITE D'APPEL (module + ligne), pas par son nom : deux tests
    # peuvent donc porter le même nom, y compris dans un même fichier. Le nom ne sert qu'à
    # l'affichage et au filtrage (sous-chaîne) -- il n'a pas à être unique.
    frame  = sys._getframe( 1 )
    line   = frame.f_lineno
    module = frame.f_globals.get( "__name__" )

    if test_phase == PHASE_COLLECT:
        file = frame.f_code.co_filename
        all_the_tests.append( Test( name, tags, file, line, module ) )
        return False

    # phase d'exécution : seul l'appel ciblé (même module + même ligne) exécute son corps. On
    # discrimine par ligne et non par nom, sinon des homonymes s'exécuteraient tous à la fois.
    return ( test_filter is not None
             and module == test_filter.module
             and line == test_filter.line )


def matches( t, filter_names, filter_tags ):
    """Mêmes sémantiques que TestFunc::matches dans test_main.h.

    - un nom filtre par sous-chaîne
    - un tag filtre par sous-chaîne dans la concaténation des tags
    """
    test_names = []
    for filter_name in filter_names:
        s = filter_name.split( "::" )
        if len( s ) == 2:
            test_names.append( s[ 1 ] )
    if test_names:
        if not any( n in t.name for n in test_names ):
            return False
    if filter_tags:
        if not any( tag in t.tag_text for tag in filter_tags ):
            return False
    return True
