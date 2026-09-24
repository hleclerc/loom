#pragma once

#include <loom/support/common_macros.h>   // HD
#include <loom/support/common_types.h>    // SI

#include <type_traits>

// LA PHYSIQUE, telle qu'on l'avait AVANT loom : des fonctions libres sur des vues indexees
// POSITIONNELLEMENT ( ligne, colonne ). Aucun nom d'axe, aucun agregat, aucun framework -- c'est
// exactement le point de l'exercice : ce fichier est le code qu'un usager possede deja, et qu'il
// ne veut surtout pas reecrire pour le rendre derivable.
//
//   du/dt = div( k grad u )   sur une grille cartesienne, pas de temps explicite,
//   temperature imposee sur le bord ( les cellules du bord ne bougent pas ).
//
// L'adjoint est ici AUSSI : la derivee d'un solveur fait partie du solveur. Les deux adjoints
// s'ecrivent en GATHER pur ( chaque cellule lit ses quatre voisines et ecrit sa seule valeur ),
// donc sans accumulation atomique -- voir le README pour pourquoi.
namespace diffusion {

/// les quatre voisines d'une cellule, dans l'ordre nord, sud, ouest, est.
static constexpr SI decalage_j[ 4 ] = { -1, 1, 0, 0 };
static constexpr SI decalage_i[ 4 ] = { 0, 0, -1, 1 };

/// vrai si `( j, i )` est une cellule INTERIEURE : le bord porte une temperature imposee, donc
/// il n'est jamais mis a jour ( et il n'a pas ses quatre voisines ).
HD inline bool interieure( SI j, SI i, SI m, SI n ) {
    return j > 0 && i > 0 && j + 1 < m && i + 1 < n;
}

/// la conductance de la face entre deux cellules de diffusivites `ka` et `kb` : leur moyenne
/// arithmetique. ( La moyenne harmonique -- plus juste sur un contraste fort -- se derive aussi
/// bien ; c'est la seule ligne a changer, adjoints compris. )
template<class TF>
HD TF conductance( TF ka, TF kb ) {
    return TF( 0.5 ) * ( ka + kb );
}

/// le scalaire d'une vue de rang 0, sans sa constness.
template<class V>
using Scalaire = std::remove_const_t<typename std::remove_reference_t<V>::TF>;

/// UN PAS EXPLICITE pour la cellule `( j, i )` :
///   u'( a ) = u( a ) + c * somme_{b voisine} K( a, b ) ( u( b ) - u( a ) ),   c = dt / h^2
/// et `u'( a ) = u( a )` sur le bord.
template<class U, class K>
HD auto pas_explicite( const U &u, const K &k, SI j, SI i, SI m, SI n, auto coef ) {
    using TF = Scalaire<U>;

    const TF uc = TF( u( j, i ) );
    if ( ! interieure( j, i, m, n ) )
        return uc;

    const TF kc = TF( k( j, i ) );
    TF somme = 0;
    for ( int v = 0; v < 4; ++v ) {
        const SI jv = j + decalage_j[ v ], iv = i + decalage_i[ v ];
        somme += conductance( kc, TF( k( jv, iv ) ) ) * ( TF( u( jv, iv ) ) - uc );
    }
    return uc + TF( coef ) * somme;
}

/// L'ADJOINT PAR RAPPORT A LA TEMPERATURE : `dL/du( a )`, ou `g` est la cotangente de la sortie.
///
/// `u( a )` intervient dans la sortie de `a` ( toujours : le terme identite, et si `a` est
/// interieure le `- somme K` ) et dans celle de chaque voisine INTERIEURE `b` ( par le `+ K u( a )`
/// de son flux ). D'ou une lecture des quatre voisines, et une seule ecriture.
template<class U, class K, class G>
HD auto adjoint_temperature( const U &u, const K &k, const G &g, SI j, SI i, SI m, SI n, auto coef ) {
    using TF = Scalaire<U>;

    const TF c = TF( coef );
    const TF kc = TF( k( j, i ) );
    const bool ici = interieure( j, i, m, n );

    TF res = TF( g( j, i ) );                         // u'( a ) = u( a ) + ...
    for ( int v = 0; v < 4; ++v ) {
        const SI jv = j + decalage_j[ v ], iv = i + decalage_i[ v ];
        if ( jv < 0 || iv < 0 || jv >= m || iv >= n )
            continue;
        const TF kf = conductance( kc, TF( k( jv, iv ) ) );
        if ( ici )                                    // ... - c K u( a ), dans la sortie de `a`
            res -= c * kf * TF( g( j, i ) );
        if ( interieure( jv, iv, m, n ) )             // ... + c K u( a ), dans la sortie de `b`
            res += c * kf * TF( g( jv, iv ) );
    }
    return res;
}

/// L'ADJOINT PAR RAPPORT A LA DIFFUSIVITE : `dL/dk( a )`.
///
/// `k( a )` entre dans la conductance de chacune des faces qui TOUCHENT `a` -- par le `0.5 k( a )`
/// de la moyenne -- donc dans la sortie de `a` ( si elle est interieure ) et dans celle de chaque
/// voisine interieure. Meme parcours que ci-dessus, meme gather.
template<class U, class K, class G>
HD auto adjoint_diffusivite( const U &u, const K &k, const G &g, SI j, SI i, SI m, SI n, auto coef ) {
    using TF = Scalaire<U>;

    const TF c = TF( coef );
    const TF uc = TF( u( j, i ) );
    const bool ici = interieure( j, i, m, n );

    TF res = 0;
    for ( int v = 0; v < 4; ++v ) {
        const SI jv = j + decalage_j[ v ], iv = i + decalage_i[ v ];
        if ( jv < 0 || iv < 0 || jv >= m || iv >= n )
            continue;
        const TF uv = TF( u( jv, iv ) );
        if ( ici )                                    // dK/dk( a ) = 1/2, dans la sortie de `a`
            res += c * TF( 0.5 ) * TF( g( j, i ) ) * ( uv - uc );
        if ( interieure( jv, iv, m, n ) )             // idem, dans la sortie de la voisine
            res += c * TF( 0.5 ) * TF( g( jv, iv ) ) * ( uc - uv );
    }
    return res;
}

} // namespace diffusion
