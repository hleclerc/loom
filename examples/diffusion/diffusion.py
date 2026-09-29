"""Un solveur de diffusion DERIVABLE, et ce que les AXES achetent.

C'est un USAGER ETRANGER de loom : il n'importe que `loom`, et rien de ce qu'il fait ne ressemble a
une cellule de Laguerre -- grille cartesienne, stencil a cinq points, aucun ragged, aucune geometrie.

    du/dt = k laplacien( u ),   pas de temps explicite, temperature imposee au bord

    u'( a ) = u( a ) + c * somme_{b voisine} ( u( b ) - u( a ) ),   c = dt k / h^2

CE QUE L'EXEMPLE MONTRE : le corps du noyau ne compte JAMAIS de dimensions. Il demande ses axes,
les parcourt, et se deplace le long de l'un d'eux. Le meme corps vaut donc en 2D, en 3D, batche ou
non -- et un `vmap` lui ajoute un axe sans qu'il sache qu'il existe ( c'est teste ).
"""
# `loom.X` et non `from loom import X` : dans un tutoriel, on doit voir d'ou vient chaque nom.
import loom


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
_avant = loom.FfiCode(
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
                queue.run_parallel( UnPas(), args.suivant.domain(), args, batch_axes );
            }
        }
    """,
)

# L'ADJOINT est un noyau comme un autre : c'est l'APPEL qui prend les deux et qui porte le nom.
#
# `u( a )` intervient dans la sortie de `a` ( le terme identite, et `-2 c D u( a )` si `a` est
# interieure, `D` etant le nombre d'axes ) et dans celle de chaque voisine INTERIEURE `b`. D'ou une
# lecture des voisines et une seule ecriture : un GATHER pur, sans accumulation atomique.
_arriere = loom.FfiCode(
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
                    static_assert( ! CT_VALUE( args.grad_for_coef.is_valid() ),
                        "diffusion : le gradient par rapport a dt k / h^2 n'est pas implemente" );

                    // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un
                    // zero symbolique il faut quand meme ecrire le gradient nul.
                    constexpr bool nulle = CT_VALUE( args.grad_for_suivant.surely_null() );

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

                    if constexpr ( CT_VALUE( args.grad_for_temperature.is_valid() ) )
                        args.grad_for_temperature( coords ) = res;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( UnPasAdjoint(), args.temperature.domain(), args, batch_axes );
            }
        }
    """,
)


def pas( u, coef ):
    """UN pas de temps explicite.

    `u` est un tableau `( ny, nx )` du framework ( ou un simple numpy ) et `coef` vaut `dt k / h^2`
    -- la diffusivite est CONSTANTE. Renvoie la temperature mise a jour, derivable par rapport
    a `u`."""
    # AUCUN vocabulaire d'axe : les axes sont deduits de la forme de `u` ( anonymes ; ils recoivent
    # des noms distincts a l'abaissement ), et `suivant` reprend LES MEMES -- c'est ce qui dit a
    # loom que les deux tenseurs sont sur la meme grille.
    temperature = loom.RealTensor( u )
    suivant = loom.RealTensor.like( temperature )

    loom.ffi_call(
        _avant,
        _arriere,
        name = "diffusion_pas",
        temperature = temperature,
        coef = loom.RealTensor( float( coef ) ),
        suivant = suivant,
        output_attributes = [ "suivant" ],
    )
    return suivant.raw


def evolution( u, coef, nb_pas ):
    """`nb_pas` pas de suite -- la chaine que l'adjoint doit remonter."""
    for _ in range( nb_pas ):
        u = pas( u, coef )
    return u


if __name__ == "__main__":
    # de quoi voir l'exemple tourner sans rien installer : `python diffusion.py`
    import numpy

    n, nb_pas, coef = 21, 40, 0.2
    centre = ( n - 1 ) / 2
    u = numpy.array( [ [ 0.0 if j in ( 0, n - 1 ) or i in ( 0, n - 1 ) else
                         float( numpy.exp( - ( ( i - centre ) ** 2 + ( j - centre ) ** 2 ) / 8 ) )
                         for i in range( n ) ] for j in range( n ) ] )

    chaud = float( numpy.asarray( u ).max() )
    v = evolution( u, coef, nb_pas )
    print( f"{ nb_pas } pas de diffusion sur une grille { n }x{ n } ( c = { coef } )" )
    print( f"  pic  { chaud:.4f} -> { float( numpy.asarray( v ).max() ):.4f}" )
    # la somme DECROIT : le bord est impose a 0, donc la chaleur s'echappe par les cotes.
    print( f"  somme { float( numpy.asarray( u ).sum() ):.4f}"
           f" -> { float( numpy.asarray( v ).sum() ):.4f}   ( elle fuit par le bord )" )
