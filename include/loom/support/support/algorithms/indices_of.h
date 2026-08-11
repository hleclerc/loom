#pragma once

#include "CartesianIndices.h"
#include "intersection.h"

namespace sdot {

namespace detail {
    /// `x` -> `CartesianIndices` : tenseur (via `.shape()`) ou forme passée directement.
    auto cartesian_indices_of( auto &&x ) {
        if constexpr ( requires { x.shape(); } )
            return CartesianIndices<DECAYED_TYPE_OF( x.shape() )>{ x.shape() };
        else
            return CartesianIndices<DECAYED_TYPE_OF( x )>{ FORWARD( x ) };
    }
}

/// `indices_of( a )` -> multi-indices de `a` ; `indices_of( a, b, ... )` -> intersection des parcours
/// (indices communs). Chaque argument est un tenseur (via `.shape()`) ou une forme.
auto indices_of( auto &&...xs ) {
    return intersection( detail::cartesian_indices_of( FORWARD( xs ) )... );
}

} // namespace sdot
