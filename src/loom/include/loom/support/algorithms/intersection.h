#pragma once

#include "../common_macros.h"

namespace sdot {

/// Intersection of index sets (e.g. `CartesianIndices`), folded term by term via the
/// member method `.intersection`. Generic: works for any set that provides it.
HD auto intersection( auto &&first ) {
    return FORWARD( first );
}

HD auto intersection( auto &&first, auto &&second, auto &&...rest ) {
    return intersection( first.intersection( second ), FORWARD( rest )... );
}

} // namespace sdot
