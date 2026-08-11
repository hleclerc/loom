#pragma once

#include "string/read_arg_name.h"
#include "common_macros.h" // HD, GD, DECAYED_TYPE_OF
#include "display.h"
#include <mutex>

namespace sdot {

T_VT void __print_with_mutex( std::ostream &os, std::string_view arg_names, const T &...arg_values ) {
    static std::mutex m;
    m.lock();

    // write
    int cpt = 0;
    auto get_item = [&]( const auto &arg_value ) {
        if ( cpt++ )
            os << "\t";
        display( os << "\033[90m" << read_arg_name( arg_names ) << ":\033[0m ", arg_value );
    };
    ( get_item( arg_values ), ... );
    os << std::endl;

    //
    m.unlock();
}

// ----------------- unified entry -----------------
// HD so the same info(...) works in host and device code (the macro expands identically in both);
// the body picks std::cout (host) or printf (device, prefixed with block/thread id) at compile time.
T_VT void __info( const char *arg_names, const T &...arg_values ) {
    __print_with_mutex( std::cout, arg_names, arg_values... );
}

#define INFO( ... ) sdot::__info( #__VA_ARGS__, __VA_ARGS__ )

} // namespace sdot
