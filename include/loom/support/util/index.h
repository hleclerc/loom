#pragma once

#include "../common_types.h"

namespace sdot {

T_TA SI index( const T &inputs, A &&func ) {
    for( SI i = 0; i < inputs.size(); ++i )
        if ( func( inputs[ i ] ) )
            return i;
    return -1;
}

T_T SI index( T &&func ) {
    for( SI i = 0; ; ++i )
        if ( func( i ) )
            return i;
}

} // namespace sdot
