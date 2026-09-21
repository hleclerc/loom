#pragma once

#include "../common_macros.h"

namespace sdot {

/// Intersection d'ensembles d'indices (p.ex. `CartesianIndices`), repliée terme à terme via la
/// méthode membre `.intersection`. Générique : fonctionne pour tout ensemble qui la fournit.
HD auto intersection( auto &&first ) {
    return FORWARD( first );
}

HD auto intersection( auto &&first, auto &&second, auto &&...rest ) {
    return intersection( first.intersection( second ), FORWARD( rest )... );
}

} // namespace sdot
