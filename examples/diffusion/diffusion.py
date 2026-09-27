"""Un solveur de diffusion DERIVABLE, ecrit par quelqu'un qui n'a jamais entendu parler de
transport optimal.

C'est un USAGER ETRANGER de loom, au sens propre : il n'importe que `loom`, son C++ ne
connait que `<loom/support/...>`, et rien de ce qu'il fait ne ressemble a une cellule de
Laguerre -- grille cartesienne, stencil a cinq points, aucun ragged, aucune geometrie.

L'histoire est celle qu'on raconte a un etranger : « j'ai deja un solveur en C++
( `include/diffusion/pas.h` ), je veux le mettre dans une boucle d'optimisation ». Ici :
retrouver le champ de diffusivite `k` a partir de la temperature observee apres N pas.

    u = etat_initial( ... )
    for _ in range( 20 ):
        u = pas( u, k, coef )

`driver.grad` traverse les 20 appels ; le noyau compile une fois.
"""
from pathlib import Path

from loom import Aggregate, Axis, CtShapeVar, RealTensor, driver
from loom.compilation.FfiCode import FfiCode


class Grille( Aggregate ):
    """L'etat du solveur : deux champs sur la meme grille, donc les memes axes.

    `ny` / `nx` sont des `CtShapeVar` ( connues a la compilation ) : une taille de grille est
    une constante du probleme, et le stencil a tout interet a la voir -- au prix d'une
    compilation par taille.
    """
    temperature : RealTensor[ "y", "x" ]
    diffusivite : RealTensor[ "y", "x" ]

    y           : Axis[ "ny" ]
    x           : Axis[ "nx" ]

    ny          : CtShapeVar
    nx          : CtShapeVar


def axes( n ):
    """Les deux axes d'une grille `n x n` : de quoi batir ses champs avec les fabriques
    ( `RealTensor[ y, x ].ones()`, `.linspace( 0, 1, x )`, ... ) sans jamais repeter la forme."""
    grille = Grille( ny = n, nx = n )
    return grille.y, grille.x


# LES DEUX NOYAUX, et c'est tout ce que Python en dit : un appel a NOTRE fonction C++.
#
# `preamble` est du C++ emis au niveau du namespace, verbatim : nos `#include`, et rien que ce qu'on
# y met. `code` est le corps du handler, ou loom a assemble `args` -- un objet dont les membres
# portent les noms de nos kwargs, plus `queue`, `machine` et `errors`.
#
# Ce qui se passe dedans -- le parcours de la grille, le choix du parallelisme -- vit dans
# `include/diffusion/noyaux.h` et n'a aucune trace ici. Un usager qui prefere Kokkos ou OpenMP
# change ce fichier-la, pas celui-ci.
#
# `include_roots` dit ou vit ce C++. Un noyau sait ou sont ses en-tetes ; ca n'a pas a etre une
# incantation de module, prononcee avant tout le reste et sans rapport visible avec lui.
_RACINE = Path( __file__ ).resolve().parent / "include"

_avant = FfiCode(
    include_roots = [ _RACINE ],
    preamble = '#include "diffusion/noyaux.h"',
    code = "diffusion::pas( args );",
)

# L'adjoint est un noyau comme un autre : c'est l'APPEL qui prend les deux et qui porte le nom.
_arriere = FfiCode(
    preamble = '#include "diffusion/noyaux.h"',
    code = "diffusion::pas_adjoint( args );",
)


def pas( u, k, coef ):
    """UN pas de temps explicite. `u` et `k` sont des tenseurs du driver de forme `( ny, nx )`,
    `coef` vaut `dt / h^2`. Renvoie la temperature mise a jour, derivable par rapport a `u` et
    a `k`."""
    ny, nx = u.shape

    grille = Grille( ny = ny, nx = nx )
    grille.temperature = u
    grille.diffusivite = k

    suivant = RealTensor[ grille.y, grille.x ]()

    driver.call(
        _avant,
        _arriere,
        name = "diffusion_pas",
        grille = grille,
        coef = RealTensor( float( coef ) ),
        suivant = suivant,
        output_attributes = [ "suivant" ],
    )
    return suivant.raw


def evolution( u, k, coef, nb_pas ):
    """`nb_pas` pas de suite -- la chaine que l'adjoint doit remonter."""
    for _ in range( nb_pas ):
        u = pas( u, k, coef )
    return u
