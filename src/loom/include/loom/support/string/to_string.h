#pragma once

#include "../display.h"
#include <sstream>

namespace sdot {

T_T std::string to_string( const T &v ) {
    std::ostringstream ss;
    display( ss, v );
    return ss.str();
}

} // namespace sdot
