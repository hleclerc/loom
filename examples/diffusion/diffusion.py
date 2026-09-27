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


# LES DEUX NOYAUX. Tout le C++ est ICI, en clair : on lit le tutoriel sans naviguer dans les
# fichiers. Seule la PHYSIQUE reste un en-tete ( `pas.h` ), parce que c'est le code qu'on avait deja
# et qu'on ne veut surtout pas reecrire -- il ne sait rien de loom.
#
# Loom appelle une fonction a signature fixe :
#
#     void kernel( auto &&queue, auto &&batch_axes, auto &&args )
#
#   queue       le contexte d'execution. `queue.run_parallel` est SON outil, pas une obligation :
#               un usager Kokkos ou OpenMP l'ignore et prend `queue.stream` plus les pointeurs et
#               les formes de `args`.
#   batch_axes  le domaine de batch de l'appel ( ce qu'un `vmap` ajoute ) -- une VALEUR, qu'on
#               compose avec le sien par `+`.
#   args        nos arguments sous leurs noms Python, plus `machine` et `errors`.
#
# Le domaine vient d'une VUE ( `args.suivant.axes()` ) : les axes d'une vue sont exactement ses
# dimensions, sans ambiguite. `item` porte des coordonnees NOMMEES, qu'on lit par `coord`.
#
# ( `include_roots` n'est pas dit : par defaut c'est le repertoire de CE fichier, donc `pas.h` se
#   trouve tout seul. )
_avant = FfiCode(
    includes = [ "pas.h" ],
    code = """
        struct UnPas {
            HD void operator()( auto item, auto &&args ) const {
                const SI j = coord( item, y ), i = coord( item, x );
                args.suivant( j, i ) = diffusion::pas_explicite(
                    args.grille.temperature, args.grille.diffusivite, j, i,
                    SI( args.grille.ny ), SI( args.grille.nx ), args.coef );
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( UnPas(), batch_axes + args.suivant.axes(), args );
        }
    """,
)

# L'adjoint est un noyau comme un autre : c'est l'APPEL qui prend les deux et qui porte le nom. Il
# tourne sur d'autres tampons, donc rien ne l'oblige a la meme geometrie que l'aller.
_arriere = FfiCode(
    includes = [ "pas.h" ],
    code = """
        struct UnPasAdjoint {
            HD void operator()( auto item, auto &&args ) const {
                const SI m = SI( args.grille.ny ), n = SI( args.grille.nx );
                const SI j = coord( item, y ), i = coord( item, x );

                // `coef` est une constante du probleme, jamais perturbee : son gradient demanderait
                // une reduction globale, et il n'est pas ecrit.
                static_assert( DECAYED_TYPE_OF( args.grad_for_coef.is_valid() )::value == 0,
                    "diffusion : le gradient par rapport au coefficient dt/h^2 n'est pas implemente" );

                // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un zero
                // symbolique il faut quand meme ecrire le gradient nul.
                constexpr bool nulle = DECAYED_TYPE_OF( args.grad_for_suivant.surely_null() )::value;

                if constexpr ( DECAYED_TYPE_OF( args.grad_for_grille.temperature.is_valid() )::value ) {
                    if constexpr ( nulle )
                        args.grad_for_grille.temperature( j, i ) = 0;
                    else
                        args.grad_for_grille.temperature( j, i ) = diffusion::adjoint_temperature(
                            args.grille.temperature, args.grille.diffusivite,
                            args.grad_for_suivant, j, i, m, n, args.coef );
                }

                if constexpr ( DECAYED_TYPE_OF( args.grad_for_grille.diffusivite.is_valid() )::value ) {
                    if constexpr ( nulle )
                        args.grad_for_grille.diffusivite( j, i ) = 0;
                    else
                        args.grad_for_grille.diffusivite( j, i ) = diffusion::adjoint_diffusivite(
                            args.grille.temperature, args.grille.diffusivite,
                            args.grad_for_suivant, j, i, m, n, args.coef );
                }
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( UnPasAdjoint(), batch_axes + args.grille.temperature.axes(), args );
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
