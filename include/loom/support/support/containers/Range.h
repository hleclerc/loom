#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI
// #include "../Ct.h" // PI

namespace sdot {

template<class TI,class TC=PI>
struct Range {
    T_U  void for_each_item_split( PI rel, PI mod, U &&func ) const { for( TC i = rel; i < TC( end ); i += mod ) func( TI( i ) ); }
    auto      make_available     ( auto &&queue, auto &&io_category, auto &&cont ) const { return cont( *this ); }
    T_U  void for_each_item      ( U &&func ) const { for( TC i = 0; i < TC( end ); ++i ) func( i ); }
    T_U  TC   operator[]         ( U index ) const { return index; } ///< the index-th item (a Range yields its own index)
    TI        size               () const { return end; }

    TI        end;               ///<
};

T_T constexpr auto range( T &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

// template<class TI,class EC>
// auto transfer_cost( const EC &/* execution_context */, const Range<TI> &/* arg */ ) {
//     return 0_c;
// }

} // namespace sdot
