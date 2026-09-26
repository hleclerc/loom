"""Les variables d'environnement de loom, en UN endroit.

= Le préfixe est `LOOM_`

Il l'était pour les quelques réglages récents (`LOOM_ZERO_OUTPUTS`, `LOOM_JOURNAL`) et il ne
l'était pas pour tous les autres, qui portaient `SDOT_` -- le nom du paquet qui a fait naître loom.
Un étranger installait loom et recevait sdot : un cache dans `~/.cache/sdot`, un `SDOT_BUILD_DIR`,
des messages d'erreur qui disent « sdot: ». C'est la première friction relevée par
`loom/examples/diffusion`, et la seule qui se règle par un renommage.

Ce qui reste légitimement à `SDOT_` : ce que sdot lit pour lui-même (`SDOT_KTYPE`, le type du noyau
de ses cellules ; `SDOT_CATALOGUE_DIR`, où est SON catalogue). La règle est celle de toujours ici :
le préfixe dit à qui appartient le réglage.

= L'ancien nom est encore lu, une fois, en le disant

Ces noms vivent dans des scripts, des conteneurs, des `Makefile` privés et des shells qu'on ne
versionne pas : les casser d'un coup ne rendrait service à personne. `SDOT_X` est donc encore lu
quand `LOOM_X` est absent, avec un avertissement émis UNE SEULE FOIS par variable -- assez pour
qu'on le voie, pas assez pour polluer une suite de tests.

= Et un seul endroit qui sait lire un interrupteur

`flag()` porte la convention, qui était recopiée telle quelle dans six fichiers :
absent -> le défaut ; `0`, `false`, `no`, `off` ou vide -> faux ; tout le reste -> vrai.
"""
import os
import sys

PREFIXE = "LOOM_"
ANCIEN_PREFIXE = "SDOT_"

_prevenus = set()


def _lu( nom ):
    """La valeur de `LOOM_<nom>`, à défaut celle de `SDOT_<nom>` (en le disant une fois), sinon
    `None`."""
    valeur = os.environ.get( PREFIXE + nom )
    if valeur is not None:
        return valeur
    valeur = os.environ.get( ANCIEN_PREFIXE + nom )
    if valeur is not None and nom not in _prevenus:
        _prevenus.add( nom )
        print( f"loom : { ANCIEN_PREFIXE }{ nom } est l'ancien nom de { PREFIXE }{ nom } "
               f"-- encore lu, à renommer", file = sys.stderr )
    return valeur


def var( nom, defaut = None ):
    """Le réglage `nom`, ou `defaut` s'il n'est pas posé."""
    valeur = _lu( nom )
    return defaut if valeur is None else valeur


def flag( nom, defaut = False ):
    """Le réglage `nom` lu comme un interrupteur (voir la docstring du module)."""
    valeur = _lu( nom )
    if valeur is None:
        return defaut
    return valeur.strip().lower() not in ( "", "0", "false", "no", "off" )


def est_pose( nom ) -> bool:
    """Si le réglage est posé, sous l'un ou l'autre préfixe."""
    return _lu( nom ) is not None


def pose( nom, valeur ):
    """Poser le réglage, pour nous et pour les sous-processus (ce que fait `loom-kernels` avant
    de lancer un relevé ou une compilation de catalogue)."""
    os.environ[ PREFIXE + nom ] = str( valeur )
