#pragma once

#include <loom/support/common_macros.h> // HD

#include <limits>

namespace sdot {

/// The reduction operators that `RedList( op )` accepts. Each knows its identity, which is what
/// initializes each thread's row.
template<class T> struct plus    { HD T operator()( T a, T b ) const { return a + b; }  HD static T identity() { return T( 0 ); } };
template<class T> struct maximum { HD T operator()( T a, T b ) const { return a > b ? a : b; } HD static T identity() { return std::numeric_limits<T>::lowest(); } };
template<class T> struct minimum { HD T operator()( T a, T b ) const { return a < b ? a : b; } HD static T identity() { return std::numeric_limits<T>::max(); } };

/// What the body receives in place of a reduction target: a PRIVATE accumulator (one per thread
/// on CPU), combined into the host target once the kernel is done. `combine( v )` or `+= v`.
template<class Op,class T>
struct Reducer {
    HD static T identity( const Op & ) { return Op::identity(); }

    HD void  combine   ( T v ) { value = op( value, v ); }
    HD Reducer &operator+=( T v ) { combine( v ); return *this; }

    Op op;
    T  value;
};

} // namespace sdot
