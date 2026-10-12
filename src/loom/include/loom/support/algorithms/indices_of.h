#pragma once

#include <loom/support/common_macros.h> // HD

#include "CartesianIndices.h"
#include "intersection.h"

namespace sdot {

namespace detail {
    /// `x` -> `CartesianIndices`: tensor (via `.shape()`) or a shape passed directly.
    HD auto cartesian_indices_of( auto &&x ) {
        if constexpr ( requires { x.shape(); } )
            return CartesianIndices<DECAYED_TYPE_OF( x.shape() )>{ x.shape() };
        else
            return CartesianIndices<DECAYED_TYPE_OF( x )>{ FORWARD( x ) };
    }
}

/// `indices_of( a )` -> multi-indices of `a`; `indices_of( a, b, ... )` -> intersection of the traversals
/// (common indices). Each argument is a tensor (via `.shape()`) or a shape.
HD auto indices_of( auto &&...xs ) {
    return intersection( detail::cartesian_indices_of( FORWARD( xs ) )... );
}

} // namespace sdot
