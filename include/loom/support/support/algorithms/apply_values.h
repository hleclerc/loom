#pragma once

#include "../common_macros.h" // HD, FORWARD

namespace sdot {

auto apply_values( auto &&list, auto &&func ) requires requires { list.apply_values( FORWARD( func ) ); } {
    return list.apply_values( FORWARD( func ) );
}

} // namespace sdot
