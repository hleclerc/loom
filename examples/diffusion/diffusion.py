"""Un solveur de diffusion DERIVABLE, et ce que les AXES achetent.

C'est un USAGER ETRANGER de loom : il n'importe que `loom`, et rien de ce qu'il fait ne ressemble a
une cellule de Laguerre -- grille cartesienne, stencil a cinq points, aucun ragged, aucune geometrie.

    du/dt = k laplacien( u ),   pas de temps explicite, temperature imposee au bord

    u'( a ) = u( a ) + c * somme_{b voisine} ( u( b ) - u( a ) ),   c = dt k / h^2

CE QUE L'EXEMPLE MONTRE : le corps du noyau ne compte JAMAIS de dimensions. Il demande ses axes,
les parcourt, et se deplace le long de l'un d'eux. Le meme corps vaut donc en 2D, en 3D, batche ou
non -- et un `vmap` lui ajoute un axe sans qu'il sache qu'il existe ( c'est teste ).
"""
from loom import Axis, RealTensor, ShapeVar, driver
from loom.compilation.FfiCode import FfiCode


def axes( n ):
    """Les deux axes d'une grille `n x n`.

    Deux tenseurs qui PARTAGENT ces axes sont sur la meme grille -- ce qu'une paire d'entiers ne
    dirait pas, et ce sur quoi tout le reste s'appuie."""
    return Axis( ShapeVar( n ), name = "y" ), Axis( ShapeVar( n ), name = "x" )


# LE NOYAU, en clair : on lit le tutoriel sans naviguer dans les fichiers.
#
# Le `namespace { }` est a nous -- nos `#include` vont donc ou on veut, et il donne a tout ce qu'il
# contient une liaison INTERNE ( plusieurs noyaux finissent lies dans une meme bibliotheque, voir
# `compilation/catalogue.py` ).
#
# Loom appelle `void kernel( auto &&queue, auto &&batch_axes, auto &&args )` :
#
#   queue       le contexte d'execution. `queue.run_parallel` est SON outil, pas une obligation :
#               un usager Kokkos ou OpenMP l'ignore et prend `queue.stream` plus les pointeurs et
#               les formes de `args`.
#   batch_axes  les axes que l'appel a ajoutes ( ce qu'un `vmap` fabrique ).
#   args        nos arguments sous leurs noms Python, plus `machine` et `errors`. `TF` est le
#               scalaire reel de l'appel.
#
# LES TROIS PRIMITIVES D'AXE, et tout en decoule :
#
#   coords[ axis ]              la coordonnee PAR NOM, pas par position
#   coords + axis               le voisin le long de CET axe ; les autres coordonnees ne bougent
#                               pas, y compris celles du batch
#   coords.axes - batch_axes    mes axes propres, par soustraction d'ensembles a la compilation
#
_avant = FfiCode(
    code = """
        namespace {
            struct UnPas {
                /// le bord porte une temperature imposee : il n'est jamais mis a jour.
                HD bool on_bord( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.temperature.size( axis );
                    } );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    const TF uc = args.temperature( coords );
                    if ( on_bord( coords, main_axes, args ) ) {
                        args.suivant( coords ) = uc;
                        return;
                    }

                    TF somme = 0;
                    for_each( main_axes, [&]( auto axis ) {
                        somme += args.temperature( coords + axis )
                               + args.temperature( coords - axis ) - 2 * uc;
                    } );
                    args.suivant( coords ) = uc + args.coef * somme;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( UnPas(), args.suivant.axes(), args, batch_axes );
            }
        }
    """,
)

# L'ADJOINT est un noyau comme un autre : c'est l'APPEL qui prend les deux et qui porte le nom.
#
# `u( a )` intervient dans la sortie de `a` ( le terme identite, et `-2 c D u( a )` si `a` est
# interieure, `D` etant le nombre d'axes ) et dans celle de chaque voisine INTERIEURE `b`. D'ou une
# lecture des voisines et une seule ecriture : un GATHER pur, sans accumulation atomique.
_arriere = FfiCode(
    code = """
        namespace {
            struct UnPasAdjoint {
                HD bool on_bord( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.temperature.size( axis );
                    } );
                }

                /// vrai si `coords` decale de `d` le long de `axis` est encore dans la grille
                HD bool dedans( auto coords, auto axis, SI d, const auto &args ) const {
                    const SI c = coords[ axis ] + d;
                    return c >= 0 && c < args.temperature.size( axis );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    // `coef` est une constante du probleme, jamais perturbee : son gradient
                    // demanderait une reduction globale, et il n'est pas ecrit.
                    static_assert( DECAYED_TYPE_OF( args.grad_for_coef.is_valid() )::value == 0,
                        "diffusion : le gradient par rapport a dt k / h^2 n'est pas implemente" );

                    // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un
                    // zero symbolique il faut quand meme ecrire le gradient nul.
                    constexpr bool nulle = DECAYED_TYPE_OF( args.grad_for_suivant.surely_null() )::value;

                    TF res = 0;
                    if constexpr ( ! nulle ) {
                        constexpr SI D = DECAYED_TYPE_OF( main_axes )::ct_size;
                        const TF c = args.coef;
                        const TF g = args.grad_for_suivant( coords );

                        res = on_bord( coords, main_axes, args ) ? g : g * ( 1 - c * ( 2 * D ) );

                        for_each( main_axes, [&]( auto axis ) {
                            for ( SI s = -1; s <= 1; s += 2 ) {
                                if ( ! dedans( coords, axis, s, args ) )
                                    continue;
                                const auto voisin = s > 0 ? coords + axis : coords - axis;
                                if ( ! on_bord( voisin, main_axes, args ) )
                                    res += c * args.grad_for_suivant( voisin );
                            }
                        } );
                    }

                    if constexpr ( DECAYED_TYPE_OF( args.grad_for_temperature.is_valid() )::value )
                        args.grad_for_temperature( coords ) = res;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( UnPasAdjoint(), args.temperature.axes(), args, batch_axes );
            }
        }
    """,
)


def pas( u, coef ):
    """UN pas de temps explicite.

    `u` est un tenseur du driver de forme `( ny, nx )` et `coef` vaut `dt k / h^2` -- la
    diffusivite est CONSTANTE. Renvoie la temperature mise a jour, derivable par rapport a `u`."""
    ny, nx = u.shape
    y = Axis( ShapeVar( ny ), name = "y" )
    x = Axis( ShapeVar( nx ), name = "x" )

    temperature = RealTensor[ y, x ]( u )
    suivant = RealTensor[ y, x ]()

    driver.call(
        _avant,
        _arriere,
        name = "diffusion_pas",
        temperature = temperature,
        coef = RealTensor( float( coef ) ),
        suivant = suivant,
        output_attributes = [ "suivant" ],
    )
    return suivant.raw


def evolution( u, coef, nb_pas ):
    """`nb_pas` pas de suite -- la chaine que l'adjoint doit remonter."""
    for _ in range( nb_pas ):
        u = pas( u, coef )
    return u
