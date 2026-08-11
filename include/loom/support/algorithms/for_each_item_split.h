#pragma once

#include "../common_types.h"
#include "for_each_item.h"

namespace sdot {

namespace detail {
    template<class T,class=void> struct has_for_each_item_split_method : std::false_type {};
    T_T struct has_for_each_item_split_method<T,void_t<decltype(std::declval<T>().for_each_item_split( 0, 1, AnyFunc<>() ))>> : std::true_type {};
}


T_TA constexpr auto for_each_item_split( T &&list, PI rel, PI mod, A &&func ) -> std::enable_if_t<detail::has_for_each_item_split_method<T>::value || detail::has_for_each_item_method<T>::value> {
    if constexpr ( detail::has_for_each_item_split_method<T>::value ) {
        list.for_each_item_split( rel, mod, FORWARD( func ) );
    } else {
        PI cpt = 0;
        for_each_item( FORWARD( list ), [&]( auto &&item ) {
            if ( cpt++ % mod == rel )
                func( FORWARD( item ) );
        } );
    }
}

} // namespace sdot

