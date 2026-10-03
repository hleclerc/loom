#pragma once

#include "AxisNames.h"
#include "Tuple.h"
#include <type_traits>

namespace sdot {

// ── A SET OF AXES ─────────────────────────────────────────────────────────────────────────────
//
/// The axes of a domain or of a multi-index, as a SET: they can be subtracted, they can be
/// traversed. This is what makes it possible to write
///
///     const auto main_axes = coords.axes - batch_axes;
///
/// and therefore to separate a kernel's OWN axes from those a `vmap` added to it -- and therefore
/// to write a stencil that does not know it is batched. The set is empty of data: it only exists
/// at compile time, and `for_each` unrolls over it as a fold.
template<class... Axes>
struct AxisSet {
    SCInt ct_size = sizeof...( Axes );
};

namespace detail {
    template<class A,class Set> constexpr bool axis_in_set = false;
    template<class A,class... B> constexpr bool axis_in_set<A,AxisSet<B...>> = ( std::is_same_v<A,B> || ... );

    /// the axes of `Rest...` that are not in `Sub`, accumulated in `Acc`
    template<class Sub,class Acc,class... Rest> struct AxisDiff;
    template<class Sub,class... Keep> struct AxisDiff<Sub,AxisSet<Keep...>> { using type = AxisSet<Keep...>; };
    template<class Sub,class... Keep,class Head,class... Tail>
    struct AxisDiff<Sub,AxisSet<Keep...>,Head,Tail...> {
        using type = std::conditional_t<axis_in_set<Head,Sub>,
            typename AxisDiff<Sub,AxisSet<Keep...>,Tail...>::type,
            typename AxisDiff<Sub,AxisSet<Keep...,Head>,Tail...>::type>;
    };

    /// the axis carried by an element. Two input forms: a multi-index ( `AxisIndex` values, from which
    /// we take the `axis_type` ) or bare axis NAMES ( what a `CartesianIndices` carries ).
    /// Confusing them was a silent defect: `batch_axes` collapsed to `UnnamedAxis`, so
    /// `coords.axes - batch_axes` removed nothing and the stencil traversed the batch axis.
    template<class T> struct AxisOf { using type = std::conditional_t<is_axis<T>,T,UnnamedAxis>; };
    template<class A,class I,bool O> struct AxisOf<AxisIndex<A,I,O>> { using type = A; };

    template<class Tup> struct AxesOfTuple;
    template<class... T> struct AxesOfTuple<Tuple<T...>> { using type = AxisSet<typename AxisOf<T>::type...>; };
}

/// THE DIFFERENCE: the axes of `a` that are not in `b`.
template<class... A,class... B>
HD constexpr auto operator-( AxisSet<A...>, AxisSet<B...> ) {
    return typename detail::AxisDiff<AxisSet<B...>,AxisSet<>,A...>::type{};
}

/// THE DIFFERENCE WITH A SINGLE AXIS: `image.axes - num_channel`, which is how a body that
/// does not know its dimension says "all the axes but that one". An axis is not an `AxisSet`,
/// hence the `requires`: nothing to disambiguate from the set subtraction above.
template<class... A,class B> requires ( is_axis<B> )
HD constexpr auto operator-( AxisSet<A...> a, B ) {
    return a - AxisSet<B>{};
}

/// for each axis. A FOLD, not a loop: the axes only exist at compile time, so the body
/// is unrolled and `axis` is a different type at each turn ( this is what makes `coords + axis` possible ).
template<class... Axes,class F>
HD constexpr void for_each( AxisSet<Axes...>, F &&f ) {
    ( f( Axes{} ), ... );
}

/// THE SAME FOLD, WITH THE POSITION: `f( axis, d )`, `d` counting from 0.
///
/// This is what is needed to cross the boundary between the axes -- which only exist at
/// compile time -- and a computation written as `for ( d = 0; d < D; ++d )`, which is the natural way
/// to write a geometry in D dimensions. A body copies its coordinates into an array there and
/// never has to name an axis again:
///
///     SI x[ D ];
///     for_each_indexed( main_axes, [&]( auto axis, int d ) { x[ d ] = coords[ axis ]; } );
///
/// The order is guaranteed: a fold over `,` evaluates left to right.
template<class... Axes,class F>
HD constexpr void for_each_indexed( AxisSet<Axes...>, F &&f ) {
    int index = 0;
    ( f( Axes{}, index++ ), ... );
}

/// true if there is an axis for which `f` is true.
template<class... Axes,class F>
HD constexpr bool any_of( AxisSet<Axes...>, F &&f ) {
    return ( f( Axes{} ) || ... );
}

// ── A NAMED MULTI-INDEX ───────────────────────────────────────────────────────────────────────
//
/// WHAT AN ITEM IS: coordinates that know the name of their axis.
///
///     coords[ y ]          the coordinate along `y`
///     coords + y           the same, shifted by +1 along `y` ( neighbourhood of a stencil )
///     coords.axes          the set of carried axes
///     tensor( coords )     indexes by NAME, and a tensor that does not have the axis ignores it
///
/// This is what replaces a bare `Tuple` passed from hand to hand: a body that wants an integer
/// asks for it by name, instead of counting positions. A tensor accepts BOTH ( see
/// `TensorView::operator()` ) -- the `Tuple` stays usable for whoever already has one.
template<class Tup>
struct Coords {
    using Values = Tup;
    using Axes   = typename detail::AxesOfTuple<Tup>::type;

    SCInt  ct_size = Tup::ct_size;

    /// the axes we carry. Without data, hence free.
    static constexpr Axes axes = {};

    /// the coordinate, by axis NAME ( `coords[ y ]` ) or by POSITION ( `coords[ 0_c ]` ).
    /// Both, because both make sense: the name when talking about an axis, the position when
    /// the domain is anonymous ( `indices_over( n )` ).
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
    /// the same coordinates, with that of `Axis` shifted by `d`
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

/// THE NEIGHBOURHOOD: `coords + y` is `coords` with its coordinate `y` increased by 1. The others do not
/// move -- including the batch axes, which is exactly what we want from a stencil.
template<class Tup,class Axis> requires ( is_axis<Axis> )
HD constexpr auto operator+( const Coords<Tup> &c, Axis a ) { return coords_of( detail::shift_coord( c.values, a, +1 ) ); }

template<class Tup,class Axis> requires ( is_axis<Axis> )
HD constexpr auto operator-( const Coords<Tup> &c, Axis a ) { return coords_of( detail::shift_coord( c.values, a, -1 ) ); }

} // namespace sdot
