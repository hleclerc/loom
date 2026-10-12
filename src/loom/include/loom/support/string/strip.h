#pragma once

#include <string_view>

namespace sdot {

inline std::string_view strip( std::string_view res ) {
    while ( res.size() && res.back() == ' ' )
        res.remove_suffix( 1 );
    while ( res.size() && res.front() == ' ' )
        res.remove_prefix( 1 );
    return res;
}

} // namespace sdot
