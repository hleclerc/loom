"""Le CLI (`./run`). Les entrées -- test, bench, experiment -- sont déclarées dans
`loom.testing`, pas ici ; ce module les ré-exporte pour que les fichiers écrits contre
l'ancien harnais (`from loom.cli import experiment, Param`) continuent de marcher.
"""

# Ré-export PARESSEUX, même règle que `loom/__init__.py` : importer le CLI ne doit rien
# exiger des dépendances d'exécution de loom. Le faire en dur tirait `loom.testing`, donc
# numpy, et rendait `./run` inutilisable précisément quand on en a besoin -- `./run env
# create` s'exécute AVANT que l'environnement existe.
_LAZY = ( "Param", "Args", "test", "bench", "experiment", "driver_is" )


def __getattr__( name: str ):
    if name in _LAZY:
        from loom import testing
        return getattr( testing, name )
    raise AttributeError( f"module { __name__ !r} has no attribute { name !r}" )


def __dir__():
    return sorted( set( globals() ) | set( _LAZY ) )
