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

from loom import Aggregate, Axis, CtShapeVar, IntTensor, RealTensor, driver
from loom.compilation.FfiCode import FfiCode
from loom.tensor import new_batch_axis


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


class Cellules( Aggregate ):
    """« Qui suis-je ? », pour le work-item qui traite une cellule : son rang plat `j * nx + i`.

    Cet agregat n'existe QUE pour porter l'axe de batch. Un `driver.call` prend son parcours
    ( `global_batch_indices` ) des `batch_axes` de ses arguments AGREGATS uniquement : un tenseur
    NU qui porterait le meme axe est ignore, et le noyau se retrouve avec un `batch_index` vide --
    ce qui echoue au fond d'une erreur de template C++, pas en Python. Voir le README.
    """
    rang        : IntTensor


def axes( n ):
    """Les deux axes d'une grille `n x n` : de quoi batir ses champs avec les fabriques
    ( `RealTensor[ y, x ].ones()`, `.linspace( 0, 1, x )`, ... ) sans jamais repeter la forme."""
    grille = Grille( ny = n, nx = n )
    return grille.y, grille.x


# DEUX noyaux : l'aller fait le pas, le retour rend les deux gradients. Les deux se
# contentent d'appeler l'en-tete -- c'est le C++ qu'on avait deja qui travaille. C'est
# l'APPEL qui les prend tous les deux, et qui porte le nom ( voir `FfiCode` ).
_avant = FfiCode(
    includes = [ "diffusion/pas.h" ],
    code = """
        const SI n = SI( grille.nx ), m = SI( grille.ny );
        const SI p = SI( cellules.rang( batch_index ) ), j = p / n, i = p % n;

        suivant( y = j, x = i ) = diffusion::pas_explicite(
            grille.temperature, grille.diffusivite, j, i, m, n, coef );
    """,
)

_arriere = FfiCode(
    includes = [ "diffusion/pas.h" ],
    code = """
        const SI n = SI( grille.nx ), m = SI( grille.ny );
        const SI p = SI( cellules.rang( batch_index ) ), j = p / n, i = p % n;

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

    # « qui suis-je ? » : un work-item par cellule, et son rang plat. La CLASSE dit le type, donc
    # l'iota est entier sans qu'on ait a le repeter -- et il est bati sur le device.
    cellule = new_batch_axis( ny * nx, prefix = "cellule" )
    cellules = Cellules( batch_axes = [ cellule ] )
    cellules.rang = IntTensor[ cellule ].iota()

    suivant = RealTensor[ grille.y, grille.x ]()

    driver.call(
        _avant,
        _arriere,
        name = "diffusion_pas",
        grille = grille,
        cellules = cellules,
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
