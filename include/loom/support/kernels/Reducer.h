#pragma once

#include <loom/support/common_macros.h> // HD

#include <limits>

namespace sdot {

/// Les opérateurs de réduction que `RedList( op )` accepte. Chacun connaît son identité, c'est ce
/// qui initialise la ligne de chaque fil.
template<class T> struct plus    { HD T operator()( T a, T b ) const { return a + b; }  HD static T identity() { return T( 0 ); } };
template<class T> struct maximum { HD T operator()( T a, T b ) const { return a > b ? a : b; } HD static T identity() { return std::numeric_limits<T>::lowest(); } };
template<class T> struct minimum { HD T operator()( T a, T b ) const { return a < b ? a : b; } HD static T identity() { return std::numeric_limits<T>::max(); } };

/// Ce que le corps reçoit à la place d'une cible de réduction : un accumulateur PRIVÉ (un par fil
/// sur CPU), combiné dans la cible hôte une fois le kernel fini. `combine( v )` ou `+= v`.
template<class Op,class T>
struct Reducer {
    HD static T identity( const Op & ) { return Op::identity(); }

    HD void  combine   ( T v ) { value = op( value, v ); }
    HD Reducer &operator+=( T v ) { combine( v ); return *this; }

    Op op;
    T  value;
};

} // namespace sdot
