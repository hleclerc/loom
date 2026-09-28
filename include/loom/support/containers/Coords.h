#pragma once

#include "AxisNames.h"
#include "Tuple.h"
#include <type_traits>

namespace sdot {

// ── UN ENSEMBLE D'AXES ────────────────────────────────────────────────────────────────────────
//
/// Les axes d'un domaine ou d'un multi-indice, comme un ENSEMBLE : on les soustrait, on les
/// parcourt. C'est ce qui permet d'ecrire
///
///     const auto main_axes = coords.axes - batch_axes;
///
/// donc de separer les axes PROPRES d'un noyau de ceux qu'un `vmap` lui a ajoutes -- et donc
/// d'ecrire un stencil qui ne sait pas qu'il est batche. L'ensemble est vide de donnees : il n'existe
/// qu'a la compilation, et `for_each` s'y deroule en un pli.
template<class... Axes>
struct AxisSet {
    SCInt ct_size = sizeof...( Axes );
};

namespace detail {
    template<class A,class Set> constexpr bool axis_in_set = false;
    template<class A,class... B> constexpr bool axis_in_set<A,AxisSet<B...>> = ( std::is_same_v<A,B> || ... );

    /// les axes de `Rest...` qui ne sont pas dans `Sub`, accumules dans `Acc`
    template<class Sub,class Acc,class... Rest> struct AxisDiff;
    template<class Sub,class... Keep> struct AxisDiff<Sub,AxisSet<Keep...>> { using type = AxisSet<Keep...>; };
    template<class Sub,class... Keep,class Head,class... Tail>
    struct AxisDiff<Sub,AxisSet<Keep...>,Head,Tail...> {
        using type = std::conditional_t<axis_in_set<Head,Sub>,
            typename AxisDiff<Sub,AxisSet<Keep...>,Tail...>::type,
            typename AxisDiff<Sub,AxisSet<Keep...,Head>,Tail...>::type>;
    };

    /// l'axe porte par un element. Deux formes d'entree : un multi-indice ( des `AxisIndex`, dont
    /// on prend l'`axis_type` ) ou des NOMS d'axes deja nus ( ce que porte un `CartesianIndices` ).
    /// Les confondre etait un defaut silencieux : `batch_axes` se reduisait a `UnnamedAxis`, donc
    /// `coords.axes - batch_axes` ne retirait rien et le stencil parcourait l'axe de batch.
    template<class T> struct AxisOf { using type = std::conditional_t<is_axis<T>,T,UnnamedAxis>; };
    template<class A,class I,bool O> struct AxisOf<AxisIndex<A,I,O>> { using type = A; };

    template<class Tup> struct AxesOfTuple;
    template<class... T> struct AxesOfTuple<Tuple<T...>> { using type = AxisSet<typename AxisOf<T>::type...>; };
}

/// LA DIFFERENCE : les axes de `a` qui ne sont pas dans `b`.
template<class... A,class... B>
HD constexpr auto operator-( AxisSet<A...>, AxisSet<B...> ) {
    return typename detail::AxisDiff<AxisSet<B...>,AxisSet<>,A...>::type{};
}

/// pour chaque axe. Un PLI, pas une boucle : les axes n'existent qu'a la compilation, donc le corps
/// est deroule et `axis` est un type different a chaque tour ( c'est ce qui permet `coords + axis` ).
template<class... Axes,class F>
HD constexpr void for_each( AxisSet<Axes...>, F &&f ) {
    ( f( Axes{} ), ... );
}

/// vrai s'il existe un axe pour lequel `f` est vraie.
template<class... Axes,class F>
HD constexpr bool any_of( AxisSet<Axes...>, F &&f ) {
    return ( f( Axes{} ) || ... );
}

// ── UN MULTI-INDICE NOMME ─────────────────────────────────────────────────────────────────────
//
/// CE QU'UN ITEM EST : des coordonnees qui savent le nom de leur axe.
///
///     coords[ y ]          la coordonnee le long de `y`
///     coords + y           les memes, decalees de +1 le long de `y` ( voisinage d'un stencil )
///     coords.axes          l'ensemble des axes portes
///     tenseur( coords )    indexe par NOM, et un tenseur qui n'a pas l'axe l'ignore
///
/// C'est ce qui remplace un `Tuple` nu passe de main en main : un corps qui veut un entier le
/// demande par nom, au lieu de compter les positions. Un tenseur accepte les DEUX ( voir
/// `TensorView::operator()` ) -- le `Tuple` reste utilisable pour qui l'a deja.
template<class Tup>
struct Coords {
    using Values = Tup;
    using Axes   = typename detail::AxesOfTuple<Tup>::type;

    SCInt  ct_size = Tup::ct_size;

    /// les axes qu'on porte. Sans donnees, donc gratuit.
    static constexpr Axes axes = {};

    /// la coordonnee, par NOM d'axe ( `coords[ y ]` ) ou par POSITION ( `coords[ 0_c ]` ).
    /// Les deux, parce que les deux ont un sens : le nom quand on parle d'un axe, la position quand
    /// le domaine est anonyme ( `indices_over( n )` ).
    HD constexpr auto operator[]( auto axis_ou_position ) const {
        if constexpr ( is_axis<DECAYED_TYPE_OF( axis_ou_position )> )
            return coord( values, axis_ou_position );
        else
            return values[ axis_ou_position ];
    }

    Tup values;
};

template<class Tup> HD constexpr auto coords_of( Tup values ) { return Coords<Tup>{ values }; }

template<class T> struct IsCoords                 : std::false_type {};
template<class T> struct IsCoords<Coords<T>>      : std::true_type  {};

namespace detail {
    /// les memes coordonnees, celle de `Axis` decalee de `d`
    template<class Axis>
    HD constexpr auto shift_coord( const auto &values, Axis, SI d ) {
        return map( values, [&]( auto e ) {
            using E = DECAYED_TYPE_OF( e );
            if constexpr ( IsAxisIndex<E>::value && std::is_same_v<typename E::axis_type,Axis> )
                return E{ e.index + d };
            else
                return e;
        } );
    }
}

/// LE VOISINAGE : `coords + y` est `coords` avec sa coordonnee `y` augmentee de 1. Les autres ne
/// bougent pas -- y compris les axes de batch, ce qui est exactement ce qu'on veut d'un stencil.
template<class Tup,class Axis> requires ( is_axis<Axis> )
HD constexpr auto operator+( const Coords<Tup> &c, Axis a ) { return coords_of( detail::shift_coord( c.values, a, +1 ) ); }

template<class Tup,class Axis> requires ( is_axis<Axis> )
HD constexpr auto operator-( const Coords<Tup> &c, Axis a ) { return coords_of( detail::shift_coord( c.values, a, -1 ) ); }

} // namespace sdot
