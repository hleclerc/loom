#pragma once

#include <loom/support/common_macros.h>   // HD
#include <loom/support/common_types.h>    // SI
#include <loom/support/atomic_add.h>      // atomic_add, atomic_fetch_add
#include <loom/support/containers/ErrorBuffer.h>   // ErrorKind

#include <cmath>
#include <type_traits>

// LE RENDU, tel qu'on l'avait AVANT loom : des fonctions libres sur des vues indexees
// POSITIONNELLEMENT. Aucun nom d'axe, aucun agregat, aucun framework.
//
// On somme des gaussiennes ANISOTROPES sur une image :
//
//     image( p ) = somme_i  opacite_i * exp( -q_i( p ) / 2 ) * couleur_i
//     q_i( p )   = a_i dx^2 + 2 b_i dx dy + c_i dy^2,   ( dx, dy ) = p - centre_i
//
// ou ( a, b, c ) est l'inverse de la covariance. Melange ADDITIF : pas d'ordre, donc pas de tri --
// la composition alpha viendra apres ( voir le README ).
//
// Le point de l'exemple n'est pas la formule, c'est que chaque pixel ne doit regarder QUE les
// splats qui l'atteignent. D'ou un index par TUILE, dont la longueur depend des donnees : c'est
// lui le ragged, et c'est lui qu'un XLA ne peut pas construire sans borner le pire cas d'avance.
namespace splats {

/// le scalaire d'une vue, sans sa constness
template<class V>
using Scalaire = std::remove_const_t<typename std::remove_reference_t<V>::TF>;

/// Au-dela de `RAYON_SIGMA` ecarts-types, un splat ne contribue plus. C'est ce qui lui donne une
/// empreinte BORNEE, donc un rectangle de tuiles -- sans quoi il n'y aurait pas d'index a construire.
static constexpr double RAYON_SIGMA = 3.0;

/// La troncature est CONTINUE : on retranche la valeur au seuil plutot que de couper net.
///
///     w = opacite * max( exp( -q/2 ) - exp( -k^2/2 ), 0 )
///
/// Couper net ferait sauter le poids de `opacite * 1.1e-2` a zero en franchissant la frontiere. Or
/// la frontiere BOUGE quand on deplace un centre ou qu'on change une covariance : la derivee
/// acquiert alors des impulsions, une difference finie mesure un saut divise par 2 eps, et une
/// descente de gradient prend du bruit a chaque pas. Verifie : avec la coupure nette, l'adjoint
/// etait exact a onze chiffres sur `couleurs` et `opacites` -- qui n'entrent pas dans `q` -- et
/// faux de 5 % sur `centres`. Avec celle-ci, les quatre sont exactes.
template<class TF>
HD TF seuil() { return TF( std::exp( -0.5 * RAYON_SIGMA * RAYON_SIGMA ) ); }

/// La forme quadratique d'un splat en un point, et son poids.
template<class TF>
HD TF forme_quadratique( TF a, TF b, TF c, TF dx, TF dy ) {
    return a * dx * dx + TF( 2 ) * b * dx * dy + c * dy * dy;
}

/// Le demi-rectangle qui contient l'ellipse `q <= RAYON_SIGMA^2`, en pixels.
///
/// `( a, b, c )` etant l'inverse de la covariance, la covariance vaut `( c, -b, a ) / det`, et
/// l'extension selon x d'une ellipse de niveau k est `k * sqrt( cov_xx )`. Une covariance inverse
/// non definie positive n'a pas d'empreinte : on rend un rectangle vide plutot que des NaN.
template<class TF>
HD bool demi_rectangle( TF a, TF b, TF c, TF &demi_x, TF &demi_y ) {
    const TF det = a * c - b * b;
    if ( ! ( det > 0 ) || ! ( a > 0 ) || ! ( c > 0 ) )
        return false;
    demi_x = TF( RAYON_SIGMA ) * std::sqrt( c / det );
    demi_y = TF( RAYON_SIGMA ) * std::sqrt( a / det );
    return true;
}

/// Les tuiles qu'un splat touche : `[ tx0, tx1 [ x [ ty0, ty1 [`, deja rognees a l'image.
/// Rend faux si le splat ne touche rien ( hors image, ou covariance degeneree ).
template<class S>
HD bool tuiles_touchees( const S &splats, SI i, SI largeur, SI hauteur, SI cote,
                         SI &tx0, SI &tx1, SI &ty0, SI &ty1 ) {
    using TF = Scalaire<decltype( splats.centres )>;

    TF demi_x, demi_y;
    if ( ! demi_rectangle( TF( splats.cov_inv( i, 0 ) ), TF( splats.cov_inv( i, 1 ) ),
                           TF( splats.cov_inv( i, 2 ) ), demi_x, demi_y ) )
        return false;

    const TF cx = TF( splats.centres( i, 0 ) ), cy = TF( splats.centres( i, 1 ) );
    const SI nb_tx = ( largeur + cote - 1 ) / cote, nb_ty = ( hauteur + cote - 1 ) / cote;

    tx0 = SI( std::floor( ( cx - demi_x ) / cote ) );
    tx1 = SI( std::floor( ( cx + demi_x ) / cote ) ) + 1;
    ty0 = SI( std::floor( ( cy - demi_y ) / cote ) );
    ty1 = SI( std::floor( ( cy + demi_y ) / cote ) ) + 1;

    tx0 = tx0 < 0 ? 0 : tx0;   tx1 = tx1 > nb_tx ? nb_tx : tx1;
    ty0 = ty0 < 0 ? 0 : ty0;   ty1 = ty1 > nb_ty ? nb_ty : ty1;
    return tx0 < tx1 && ty0 < ty1;
}

/// PASSE 1 : inscrire le splat `i` dans la liste de chacune des tuiles qu'il touche.
///
/// Chaque splat reserve sa place par un `atomic_fetch_add` sur le compteur de la tuile : plusieurs
/// work-items ecrivent dans la MEME liste, et le ticket rendu est leur fente. Un ticket au-dela de
/// la capacite est signale dans le tampon d'erreurs de l'appel -- l'hote reserve alors plus grand
/// et relance, ce qui est exactement la machinerie qu'aucun framework n'a. `compte` reste le compte
/// VOULU ( pas le clampe ) : c'est ce que l'hote doit connaitre pour dimensionner la fois suivante.
template<class S, class Compte, class Ids>
HD void inscrire( const S &splats, SI i, SI largeur, SI hauteur, SI cote,
                  const Compte &compte, const Ids &ids ) {
    SI tx0, tx1, ty0, ty1;
    if ( ! tuiles_touchees( splats, i, largeur, hauteur, cote, tx0, tx1, ty0, ty1 ) )
        return;

    const SI nb_tx = ( largeur + cote - 1 ) / cote;
    for ( SI ty = ty0; ty < ty1; ++ty )
        for ( SI tx = tx0; tx < tx1; ++tx ) {
            const SI t = ty * nb_tx + tx;

            // la `ShapeVarView` de CETTE tuile : son compteur, sa capacite ( `max` ), et le tampon
            // d'erreurs de l'appel -- un `ShapeVar` ragged porte les trois par cellule.
            auto cellule = compte( t );
            auto &compteur = cellule.view.ref();
            using TC = std::remove_reference_t<decltype( compteur )>;

            const SI fente = SI( atomic_fetch_add( compteur, TC( 1 ) ) );
            if ( fente < cellule.max )
                ids( t, fente ) = i;
            else
                // le compte VOULU est signale, pas tronque : l'hote reserve plus grand et relance.
                cellule.errors.record( ErrorKind::capacity_overflow, cellule.id, fente + 1 );
        }
}

/// La contribution du splat `i` au pixel `( x, y )`, ajoutee a `( r, g, b )`. Ce qui suit ne depend
/// PAS de la facon dont l'index est represente : c'est ce qui permet d'en comparer deux.
template<class S, class TF>
HD void contribution( const S &splats, SI i, SI x, SI y, TF &r, TF &g, TF &b ) {
    const TF dx = TF( x ) - TF( splats.centres( i, 0 ) );
    const TF dy = TF( y ) - TF( splats.centres( i, 1 ) );
    const TF q = forme_quadratique( TF( splats.cov_inv( i, 0 ) ), TF( splats.cov_inv( i, 1 ) ),
                                    TF( splats.cov_inv( i, 2 ) ), dx, dy );
    if ( q > TF( RAYON_SIGMA * RAYON_SIGMA ) )
        return;
    const TF w = TF( splats.opacites( i ) ) * ( std::exp( -TF( 0.5 ) * q ) - seuil<TF>() );
    r += w * TF( splats.couleurs( i, 0 ) );
    g += w * TF( splats.couleurs( i, 1 ) );
    b += w * TF( splats.couleurs( i, 2 ) );
}

/// PASSE 2 : la couleur d'un pixel, somme sur les splats de SA tuile seulement.
template<class S, class Ids>
HD void rendre_pixel( const S &splats, const Ids &ids, SI nb_dans_la_tuile, SI t,
                      SI x, SI y, auto &&sortie ) {
    using TF = Scalaire<decltype( splats.centres )>;

    TF r = 0, g = 0, b = 0;
    for ( SI k = 0; k < nb_dans_la_tuile; ++k )
        contribution( splats, SI( ids( t, k ) ), x, y, r, g, b );
    sortie( 0 ) = r;
    sortie( 1 ) = g;
    sortie( 2 ) = b;
}

/// Le meme rendu, sur un index CSR : `ids_plat` est une liste unique, la tuile `t` occupant
/// `[ base, base + nb [`. La seule difference avec ci-dessus est la lecture de `i`.
template<class S, class Ids>
HD void rendre_pixel_csr( const S &splats, const Ids &index, SI t,
                          SI x, SI y, auto &&sortie ) {
    using TF = Scalaire<decltype( splats.centres )>;

    // `index( t, k )` : le `t` va chercher dans les offsets, le `k` indexe dans la ligne. Ni
    // borne ni compte a passer a cote -- le tenseur les porte ( `loom.CsrTensor` ).
    TF r = 0, g = 0, b = 0;
    for ( SI k = 0; k < index.row_size( t ); ++k )
        contribution( splats, SI( index( t, k ) ), x, y, r, g, b );
    sortie( 0 ) = r;
    sortie( 1 ) = g;
    sortie( 2 ) = b;
}

/// PASSE 1 du CSR, premiere moitie : COMPTER, sans rien ecrire. Un compteur dense par tuile, de
/// taille connue -- aucun ragged, aucune capacite a deviner.
template<class S, class Comptes>
HD void compter( const S &splats, SI i, SI largeur, SI hauteur, SI cote, const Comptes &comptes ) {
    SI tx0, tx1, ty0, ty1;
    if ( ! tuiles_touchees( splats, i, largeur, hauteur, cote, tx0, tx1, ty0, ty1 ) )
        return;
    const SI nb_tx = ( largeur + cote - 1 ) / cote;
    for ( SI ty = ty0; ty < ty1; ++ty )
        for ( SI tx = tx0; tx < tx1; ++tx ) {
            auto &cellule = comptes( ty * nb_tx + tx ).ref();
            atomic_add( cellule, std::remove_reference_t<decltype( cellule )>( 1 ) );
        }
}

/// PASSE 1 du CSR, seconde moitie : REMPLIR, les offsets etant connus. Le curseur par tuile donne
/// la place dans la liste unique.
template<class S, class Curseurs, class Ids>
HD void remplir( const S &splats, SI i, SI largeur, SI hauteur, SI cote,
                 const Curseurs &curseurs, const Ids &index ) {
    SI tx0, tx1, ty0, ty1;
    if ( ! tuiles_touchees( splats, i, largeur, hauteur, cote, tx0, tx1, ty0, ty1 ) )
        return;
    const SI nb_tx = ( largeur + cote - 1 ) / cote;
    for ( SI ty = ty0; ty < ty1; ++ty )
        for ( SI tx = tx0; tx < tx1; ++tx ) {
            const SI t = ty * nb_tx + tx;
            auto &curseur = curseurs( t ).ref();
            using TC = std::remove_reference_t<decltype( curseur )>;
            const SI k = SI( atomic_fetch_add( curseur, TC( 1 ) ) );
            index( t, k ) = i;
        }
}

/// L'ADJOINT de la passe 2, pour un pixel : chaque splat de la tuile recoit sa part.
///
/// Toutes les contributions sont ACCUMULEES ATOMIQUEMENT : un splat est touche par tous les pixels
/// qu'il couvre, donc par des work-items differents. C'est l'inverse de l'adjoint de
/// `examples/diffusion`, qui etait en gather pur -- et c'est ce qui exerce le semis a zero des
/// sorties partagees ( voir `CallArg_Tensor.cpp_seed_member` ).
///
/// Les termes, pour `L` la perte et `g` la cotangente du pixel :
///     dL/dcouleur_i = w * g                                  avec w = opacite * ( e - E )
///     dL/dopacite_i = ( e - E ) * <g, couleur>
///     dL/dcentre_i  = opacite * e * <g, couleur> * ( a dx + b dy, b dx + c dy )
///     dL/d(a,b,c)_i = opacite * e * <g, couleur> * ( -dx^2/2, -dx dy, -dy^2/2 )
///
/// La VALEUR passe par `( e - E )`, la DERIVEE en geometrie par `e` seul : la constante retranchee
/// par la troncature continue ne depend pas de `q`, donc elle disparait en derivant.
template<class S, class Ids, class G, class Grad>
HD void rendre_pixel_bwd( const S &splats, const Ids &ids, SI nb_dans_la_tuile, SI t,
                          SI x, SI y, const G &g, const Grad &grad ) {
    using TF = Scalaire<decltype( splats.centres )>;

    const TF g0 = TF( g( 0 ) ), g1 = TF( g( 1 ) ), g2 = TF( g( 2 ) );
    for ( SI k = 0; k < nb_dans_la_tuile; ++k ) {
        const SI i = SI( ids( t, k ) );
        const TF a = TF( splats.cov_inv( i, 0 ) ), b = TF( splats.cov_inv( i, 1 ) ), c = TF( splats.cov_inv( i, 2 ) );
        const TF dx = TF( x ) - TF( splats.centres( i, 0 ) );
        const TF dy = TF( y ) - TF( splats.centres( i, 1 ) );
        const TF q = forme_quadratique( a, b, c, dx, dy );
        if ( q > TF( RAYON_SIGMA * RAYON_SIGMA ) )
            continue;

        const TF opacite = TF( splats.opacites( i ) );
        const TF e = std::exp( -TF( 0.5 ) * q );
        const TF w = opacite * ( e - seuil<TF>() );      // la valeur
        const TF de = opacite * e;                       // ce par quoi passe la derivee en q

        // <g, couleur> : ce par quoi passe toute la dependance en geometrie
        const TF gc = g0 * TF( splats.couleurs( i, 0 ) ) + g1 * TF( splats.couleurs( i, 1 ) )
                    + g2 * TF( splats.couleurs( i, 2 ) );

        if constexpr ( grad.couleurs.is_valid ) {
            atomic_add( grad.couleurs( i, 0 ).ref(), w * g0 );
            atomic_add( grad.couleurs( i, 1 ).ref(), w * g1 );
            atomic_add( grad.couleurs( i, 2 ).ref(), w * g2 );
        }
        if constexpr ( grad.opacites.is_valid )
            atomic_add( grad.opacites( i ).ref(), ( e - seuil<TF>() ) * gc );
        if constexpr ( grad.centres.is_valid ) {
            atomic_add( grad.centres( i, 0 ).ref(), de * gc * ( a * dx + b * dy ) );
            atomic_add( grad.centres( i, 1 ).ref(), de * gc * ( b * dx + c * dy ) );
        }
        if constexpr ( grad.cov_inv.is_valid ) {
            atomic_add( grad.cov_inv( i, 0 ).ref(), de * gc * ( -TF( 0.5 ) * dx * dx ) );
            atomic_add( grad.cov_inv( i, 1 ).ref(), de * gc * ( -dx * dy ) );
            atomic_add( grad.cov_inv( i, 2 ).ref(), de * gc * ( -TF( 0.5 ) * dy * dy ) );
        }
    }
}

} // namespace splats
