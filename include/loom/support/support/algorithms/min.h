#pragma once

#include "sdot/support/common_macros.h"
#include "../util/TypePromote.h"

namespace sdot {

constexpr auto min( auto &&a ) {
    return FORWARD( a );
}

constexpr auto min( auto &&a, auto &&b ) {
    auto ieq = a <= b;
    if constexpr ( requires { DECAYED_TYPE_OF( ieq )::value; } ) {
        if constexpr ( DECAYED_TYPE_OF( ieq )::value )
            return FORWARD( a );
        else
            return FORWARD( b );
    } else {
        using TR = TypePromote<DECAYED_TYPE_OF( a ),DECAYED_TYPE_OF( b )>::type;
        return ieq ? TR( FORWARD( a ) ) : TR( FORWARD( b ) );
    }
}

constexpr auto min( auto &&a, auto &&b, auto &&...tail ) {
    return min( min( FORWARD( a ), FORWARD( b ) ), FORWARD( tail )... );
}

} // namespace sdot
