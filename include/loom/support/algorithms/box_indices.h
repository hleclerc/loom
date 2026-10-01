#pragma once

#include "../containers/Tuple.h"   // tuple, with_appended_value
#include "../common_macros.h"      // HD
#include "../common_types.h"       // SI
#include "../Ct.h"

namespace sdot {

/// LE PONT entre les positions -- connues à la compilation -- et une boucle `for ( d = 0; d < D; ++d )`.
///
/// Un corps qui veut être écrit UNE FOIS pour toutes les dimensions calcule naturellement avec ses
/// coordonnées comme avec des nombres : une boucle sur `d`, un tableau de `D` entiers. Un tenseur,
/// lui, s'indexe par un `Tuple`, dont les positions sont des types. Ces deux fonctions font
/// l'aller-retour, et c'est tout ce qui manque pour que les deux mondes se parlent.
///
/// `to_array` lit une forme ( ou n'importe quel `Tuple` d'entiers ) dans un tableau ; `to_tuple`
/// refait un `Tuple` à partir du tableau, qu'un tenseur accepte alors directement :
///
///     SI nb[ D ];  to_array<D>( grid.shape(), nb );
///     ...
///     ids( to_tuple<D>( at ), slot ) = i;
template<int N>
HD void to_array( const auto &values, SI *dst ) {
    if constexpr ( N > 0 ) {
        dst[ N - 1 ] = SI( values[ Ct<int,N-1>() ] );
        to_array<N-1>( values, dst );
    }
}

template<int N>
HD auto to_tuple( const SI *src ) {
    if constexpr ( N == 0 )
        return tuple();
    else
        return to_tuple<N-1>( src ).with_appended_value( src[ N - 1 ] );
}

/// TOUS LES MULTI-INDICES D'UNE BOÎTE `[ lo, hi [`, passés à `f` sous forme de `Tuple`. Rien du
/// tout si la boîte est vide le long d'un seul axe.
///
/// UN ODOMÈTRE, et pas `D` boucles imbriquées : `D` est connu à la compilation mais les BORNES ne
/// le sont pas, et un nid de profondeur `D` ne s'écrit pas sans récursion. L'odomètre, lui, se lit
/// -- et c'est ce qui permet à un noyau de parcourir l'empreinte d'un objet en nD sans qu'une seule
/// ligne de son corps mentionne `x` ou `y`.
///
/// `CartesianIndices` ne fait pas ça : il part de 0 et ses extents sont dans son type, ce qui est
/// la bonne forme pour un domaine de LANCEMENT. Ici la boîte dépend de la donnée ( l'empreinte d'un
/// splat ), donc elle ne peut être qu'une valeur.
template<int D>
HD void for_each_in_box( const SI *lo, const SI *hi, auto &&f ) {
    SI at[ D > 0 ? D : 1 ];
    for ( int d = 0; d < D; ++d ) {
        if ( lo[ d ] >= hi[ d ] )
            return;
        at[ d ] = lo[ d ];
    }

    while ( true ) {
        f( to_tuple<D>( at ) );

        int d = 0;
        while ( d < D ) {
            if ( ++at[ d ] < hi[ d ] )
                break;
            at[ d ] = lo[ d ];
            ++d;
        }
        if ( d == D )
            return;
    }
}

} // namespace sdot
