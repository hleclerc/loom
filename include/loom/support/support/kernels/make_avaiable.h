#pragma once

#include "sdot/support/algorithms/apply_values.h"
#include "sdot/support/common_macros.h"
#include <type_traits>

namespace sdot {

auto make_available( auto &&queue, auto &&io_category, auto &&arg, auto &&cont );

namespace detail::MakeAvailable {
    // CPS fold: make each value available in turn, keeping every mapped value alive on
    // the stack, then call `cont` with the whole pack of mapped values.
    auto _seq( auto &&/*queue*/, auto &&/*io_category*/, auto &&cont, auto &&...mapped ) {
        return cont( FORWARD( mapped )... );
    }

    auto _seq( auto &&queue, auto &&io_category, auto &&cont, auto &&head, auto &&...tail ) {
        return make_available( queue, io_category, FORWARD( head ), [&]( auto &&mapped_head ) {
            return _seq( queue, io_category, [&]( auto &&...mapped_tail ) {
                return cont( FORWARD( mapped_head ), FORWARD( mapped_tail )... );
            }, FORWARD( tail )... );
        } );
    }
}

auto make_available( auto &&queue, auto &&io_category, auto &&arg, auto &&cont ) {
    using T = DECAYED_TYPE_OF( arg );
    if constexpr ( requires { arg.make_available( FORWARD( queue ), FORWARD( io_category ), FORWARD( cont ) ); } )
        return arg.make_available( FORWARD( queue ), FORWARD( io_category ), FORWARD( cont ) );
    else if constexpr ( requires { apply_values( FORWARD( arg ), []( auto &&... ) {} ); } )
        return apply_values( FORWARD( arg ), [&]( auto &&...values ) { using TR = DECAYED_TYPE_OF( arg );
            return detail::MakeAvailable::_seq( queue, io_category, [&]( auto &&...mapped ) {
                return cont( TR::make_variant( FORWARD( mapped )... ) );
            }, FORWARD( values )... );
        } );
    else if constexpr ( std::is_arithmetic_v<T> )
        return cont( arg );
    else
        return arg.theres_no_make_available_func();
}

} // namespace sdot
