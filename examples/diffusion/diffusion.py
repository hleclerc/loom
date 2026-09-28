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
# fichiers. Le `namespace { }` est a nous -- nos `#include` vont donc ou on veut, et il donne a tout
# ce qu'il contient une liaison INTERNE ( plusieurs noyaux finissent lies dans une meme
# bibliotheque, voir `compilation/catalogue.py` ).
#
# Loom appelle `void kernel( auto &&queue, auto &&batch_axes, auto &&args )` :
#
#   queue       le contexte d'execution. `queue.run_parallel` est SON outil, pas une obligation :
#               un usager Kokkos ou OpenMP l'ignore et prend `queue.stream` plus les pointeurs et
#               les formes de `args`.
#   batch_axes  le domaine de batch de l'appel ( ce qu'un `vmap` ajoute ) -- une VALEUR.
#   args        nos arguments sous leurs noms Python, plus `machine` et `errors`. `TF` est le
#               scalaire reel de l'appel.
#
# CE QUE LES AXES ACHETENT, et c'est le sujet de l'exemple : le corps ne compte jamais de
# dimensions. `coords.axes - batch_axes` donne les axes PROPRES du noyau, `for_each` les deroule a
# la compilation, et `coords + axis` decale la coordonnee de CET axe en laissant les autres --
# y compris celles du batch. Le stencil est donc ecrit une fois, en dimension quelconque, et il ne
# sait pas qu'il peut etre batche.
_avant = FfiCode(
    code = """
        namespace {
            HD TF conductance( TF ka, TF kb ) { return TF( 0.5 ) * ( ka + kb ); }

            struct UnPas {
                /// le bord porte une temperature imposee : il n'est jamais mis a jour.
                HD bool on_boundary( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.suivant.size( axis );
                    } );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    const TF uc = args.grille.temperature( coords );
                    if ( on_boundary( coords, main_axes, args ) ) {
                        args.suivant( coords ) = uc;
                        return;
                    }

                    const TF kc = args.grille.diffusivite( coords );
                    TF somme = 0;
                    for_each( main_axes, [&]( auto axis ) {
                        somme += conductance( kc, TF( args.grille.diffusivite( coords + axis ) ) )
                               * ( TF( args.grille.temperature( coords + axis ) ) - uc );
                        somme += conductance( kc, TF( args.grille.diffusivite( coords - axis ) ) )
                               * ( TF( args.grille.temperature( coords - axis ) ) - uc );
                    } );
                    args.suivant( coords ) = uc + TF( args.coef ) * somme;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( UnPas(), batch_axes + args.suivant.axes(), args, batch_axes );
            }
        }
    """,
)

# L'adjoint est un noyau comme un autre : c'est l'APPEL qui prend les deux et qui porte le nom. Les
# deux adjoints s'ecrivent en GATHER pur -- chaque cellule lit ses voisines et ecrit sa seule
# valeur -- donc sans accumulation atomique.
_arriere = FfiCode(
    code = """
        namespace {
            HD TF conductance( TF ka, TF kb ) { return TF( 0.5 ) * ( ka + kb ); }

            struct UnPasAdjoint {
                HD bool interieure( auto coords, auto main_axes, const auto &args ) const {
                    return ! any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.grille.temperature.size( axis );
                    } );
                }

                /// vrai si `coords` decale de `d` le long de `axis` est encore dans la grille
                HD bool dedans( auto coords, auto axis, SI d, const auto &args ) const {
                    const SI c = coords[ axis ] + d;
                    return c >= 0 && c < args.grille.temperature.size( axis );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    // `coef` est une constante du probleme, jamais perturbee : son gradient
                    // demanderait une reduction globale, et il n'est pas ecrit.
                    static_assert( DECAYED_TYPE_OF( args.grad_for_coef.is_valid() )::value == 0,
                        "diffusion : le gradient par rapport a dt/h^2 n'est pas implemente" );

                    // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un
                    // zero symbolique il faut quand meme ecrire le gradient nul.
                    constexpr bool nulle = DECAYED_TYPE_OF( args.grad_for_suivant.surely_null() )::value;

                    const TF c  = TF( args.coef );
                    const TF kc = TF( args.grille.diffusivite( coords ) );
                    const TF uc = TF( args.grille.temperature( coords ) );
                    const bool ici = interieure( coords, main_axes, args );

                    TF g_temp = nulle ? TF( 0 ) : TF( args.grad_for_suivant( coords ) );
                    TF g_diff = 0;

                    if constexpr ( ! nulle ) {
                        for_each( main_axes, [&]( auto axis ) {
                            for ( SI d = -1; d <= 1; d += 2 ) {
                                if ( ! dedans( coords, axis, d, args ) )
                                    continue;
                                const auto voisin = d > 0 ? coords + axis : coords - axis;
                                const TF kv = TF( args.grille.diffusivite( voisin ) );
                                const TF uv = TF( args.grille.temperature( voisin ) );
                                const TF kf = conductance( kc, kv );
                                const bool la = ! on_bord( voisin, main_axes, args );

                                if ( ici ) {
                                    g_temp -= c * kf * TF( args.grad_for_suivant( coords ) );
                                    g_diff += c * TF( 0.5 ) * TF( args.grad_for_suivant( coords ) ) * ( uv - uc );
                                }
                                if ( la ) {
                                    g_temp += c * kf * TF( args.grad_for_suivant( voisin ) );
                                    g_diff += c * TF( 0.5 ) * TF( args.grad_for_suivant( voisin ) ) * ( uc - uv );
                                }
                            }
                        } );
                    }

                    if constexpr ( DECAYED_TYPE_OF( args.grad_for_grille.temperature.is_valid() )::value )
                        args.grad_for_grille.temperature( coords ) = g_temp;
                    if constexpr ( DECAYED_TYPE_OF( args.grad_for_grille.diffusivite.is_valid() )::value )
                        args.grad_for_grille.diffusivite( coords ) = g_diff;
                }

                HD bool on_bord( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.grille.temperature.size( axis );
                    } );
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( UnPasAdjoint(), batch_axes + args.grille.temperature.axes(),
                                    args, batch_axes );
            }
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
