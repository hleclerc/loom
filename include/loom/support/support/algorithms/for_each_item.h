#pragma once

#include "apply_values.h" // FORWARD
#include <iterator>

namespace sdot {

auto for_each_item( auto &&list, auto &&func )
        requires (
            requires { apply_values( FORWARD( list ), detail::AnyFunc<>() ); } ||
            requires { list.for_each_item( FORWARD( func ) ); } ||
            requires { list.begin(); }
        )  {
    if constexpr ( requires { apply_values( FORWARD( list ), detail::AnyFunc<>() ); } )
        apply_values( FORWARD( list ), [&]( auto &&...values ) { ( func( values ), ... ); } );
    else if constexpr ( requires { list.for_each_item( FORWARD( func ) ); } )
        list.for_each_item( FORWARD( func ) );
    else {
        for( auto &&v : list )
            func( v );
    }
}

} // namespace sdot
