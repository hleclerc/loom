#pragma once

#include <limits>

namespace sdot {

/// Les opérateurs de réduction que `RedList( op )` accepte. Chacun connaît son identité, c'est ce
/// qui initialise la ligne de chaque fil.
template<class T> struct plus    { T operator()( T a, T b ) const { return a + b; }        static T identity() { return T( 0 ); } };
template<class T> struct maximum { T operator()( T a, T b ) const { return a > b ? a : b; } static T identity() { return std::numeric_limits<T>::lowest(); } };
template<class T> struct minimum { T operator()( T a, T b ) const { return a < b ? a : b; } static T identity() { return std::numeric_limits<T>::max(); } };

/// Ce que le corps reçoit à la place d'une cible de réduction : un accumulateur PRIVÉ (un par fil
/// sur CPU), combiné dans la cible hôte une fois le kernel fini. `combine( v )` ou `+= v`.
template<class Op,class T>
struct Reducer {
    static T identity( const Op & ) { return Op::identity(); }

    void     combine   ( T v ) { value = op( value, v ); }
    Reducer &operator+=( T v ) { combine( v ); return *this; }

    Op op;
    T  value;
};

} // namespace sdot
