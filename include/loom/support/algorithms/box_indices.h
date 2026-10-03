#pragma once

#include "../containers/Tuple.h"   // tuple, with_appended_value
#include "../common_macros.h"      // HD
#include "../common_types.h"       // SI
#include "../Ct.h"

namespace sdot {

/// THE BRIDGE between positions -- known at compile time -- and a `for ( d = 0; d < D; ++d )` loop.
///
/// A body that wants to be written ONCE for all dimensions naturally computes with its
/// coordinates as with numbers: a loop over `d`, an array of `D` integers. A tensor,
/// for its part, is indexed by a `Tuple`, whose positions are types. These two functions do
/// the round trip, and that is all that is missing for the two worlds to talk to each other.
///
/// `to_array` reads a shape ( or any `Tuple` of integers ) into an array; `to_tuple`
/// rebuilds a `Tuple` from the array, which a tensor then accepts directly:
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

/// ALL THE MULTI-INDICES OF A BOX `[ lo, hi [`, passed to `f` as a `Tuple`. Nothing at
/// all if the box is empty along a single axis.
///
/// AN ODOMETER, and not `D` nested loops: `D` is known at compile time but the BOUNDS are not,
/// and a nest of depth `D` cannot be written without recursion. The odometer, for its part, reads
/// -- and that is what lets a kernel walk an object's footprint in nD without a single
/// line of its body mentioning `x` or `y`.
///
/// `CartesianIndices` does not do that: it starts at 0 and its extents are in its type, which is
/// the right shape for a LAUNCH domain. Here the box depends on the data ( the footprint of a
/// splat ), so it can only be a value.
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
