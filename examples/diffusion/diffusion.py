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

import loom.compilation as compilation

# le C++ de CE paquet, enregistre aupres de loom exactement comme sdot enregistre le sien :
# loom ne connait pas ses usagers par leur nom.
compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )

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


# DEUX noyaux : l'aller fait le pas, le retour rend les deux gradients. Les deux se
# contentent d'appeler l'en-tete -- c'est le C++ qu'on avait deja qui travaille. C'est
# l'APPEL qui les prend tous les deux, et qui porte le nom ( voir `FfiCode` ).
# C'est LE CORPS qui lance : `indices_over( ny, nx )` dit le domaine -- un work-item par cellule,
# et deux coordonnees, celles que la grille a deja. Le parallelisme d'un noyau n'est pas un axe de
# `vmap`, et ne devrait pas avoir a s'en deguiser un.
_avant = FfiCode.handler(
    includes = [ "diffusion/pas.h" ],
    functors = { "un_pas": """
        const SI j = item[ 0_c ], i = item[ 1_c ];
        suivant( y = j, x = i ) = diffusion::pas_explicite(
            grille.temperature, grille.diffusivite, j, i, SI( grille.ny ), SI( grille.nx ), coef );
    """ },
    code = """
        launch( indices_over( grille.ny, grille.nx ), un_pas{} );
    """,
)

_arriere = FfiCode.handler(
    includes = [ "diffusion/pas.h" ],
    functors = { "un_pas_adjoint": """
        const SI n = SI( grille.nx ), m = SI( grille.ny );
        const SI j = item[ 0_c ], i = item[ 1_c ];

        // `coef` est une constante du probleme, jamais perturbee : son gradient demanderait une
        // reduction globale ( une somme atomique sur toute la grille ), et il n'est pas ecrit.
        static_assert( DECAYED_TYPE_OF( grad_for_coef.is_valid() )::value == 0,
            "diffusion : le gradient par rapport au coefficient dt/h^2 n'est pas implemente" );

        // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un zero
        // symbolique il faut quand meme ecrire le gradient nul.
        constexpr bool nulle = DECAYED_TYPE_OF( grad_for_suivant.surely_null() )::value;

        if constexpr ( DECAYED_TYPE_OF( grad_for_grille.temperature.is_valid() )::value ) {
            if constexpr ( nulle )
                grad_for_grille.temperature( y = j, x = i ) = 0;
            else
                grad_for_grille.temperature( y = j, x = i ) = diffusion::adjoint_temperature(
                    grille.temperature, grille.diffusivite, grad_for_suivant, j, i, m, n, coef );
        }

        if constexpr ( DECAYED_TYPE_OF( grad_for_grille.diffusivite.is_valid() )::value ) {
            if constexpr ( nulle )
                grad_for_grille.diffusivite( y = j, x = i ) = 0;
            else
                grad_for_grille.diffusivite( y = j, x = i ) = diffusion::adjoint_diffusivite(
                    grille.temperature, grille.diffusivite, grad_for_suivant, j, i, m, n, coef );
        }
    """ },
    code = """
        launch( indices_over( grille.ny, grille.nx ), un_pas_adjoint{} );
    """,
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
