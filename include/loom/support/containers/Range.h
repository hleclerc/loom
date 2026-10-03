#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI
#include "../Ct.h"

namespace sdot {

template<class TI,class TC=PI>
struct Range {
    T_U     void for_each_item_split( PI rel, PI mod, U &&func ) const { for( TC i = rel; i < TC( end ); i += mod ) func( TI( i ) ); }
       auto   kernel_form        ( auto &&/*queue*/, auto /*io_category*/ ) const { return *this; }
    // A range designates NOTHING in memory: its transfer cost is zero, whatever the type
    // of its bound. Say so here rather than counting on `std::is_trivial_v` in `transfer_cost`,
    // because a compile-time bound (`Range<Ct<SI,2>>`, what `range( nb_items() )` returns on a
    // tensor whose extents are all in the type) is NOT trivial -- `Ct` has a
    // constructor -- and then fell through to the error branch.
       constexpr auto transfer_cost ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    T_U  HD void for_each_item   ( U &&func ) const { for( TC i = 0; i < TC( end ); ++i ) func( i ); }
    T_U  HD TC   operator[]      ( U index ) const { return index; } ///< the index-th item (a Range yields its own index)
    HD TI     size               () const { return end; }

    TI        end;               ///<
};

T_T HD constexpr auto range( T &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

} // namespace sdot
