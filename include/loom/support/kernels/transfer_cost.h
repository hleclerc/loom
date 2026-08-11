#pragma once

#include "../algorithms/apply_values.h"
#include "../Ct.h"

namespace sdot {

///
auto transfer_cost( auto &&queue, auto io_category, auto &&arg ) {
    if constexpr( requires { arg.transfer_cost( queue, io_category ); } )
        return arg.transfer_cost( queue, io_category );
    else if constexpr ( std::is_trivial_v<DECAYED_TYPE_OF( arg )> )
        return 0_c;
    else if constexpr ( requires { apply_values( arg, []( auto &&... ) {} ); } )
        return apply_values( arg, [&]( auto &&...values ) { return ( transfer_cost( queue, io_category, values ) + ... + 0_c ); } );
    else {
        arg.no_transfer_cost();
        return 0_c;
    }
}

} // namespace sdot
