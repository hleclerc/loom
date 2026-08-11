#pragma once

#include "algorithms/for_each_item.h"
#include <iostream>

namespace sdot {

void display( std::ostream &os, const auto &value ) {
    if constexpr ( requires { value.display( os ); } ) { // method
        value.display( os );
    } else if constexpr ( requires { value.shape().size(); } ) { // tensor like
        constexpr int rank = DECAYED_TYPE_OF( value.shape().size() )::value;
        const auto shape = value.shape();
        if constexpr ( rank == 0 ) {
            os << value.value();
        } else if constexpr ( rank == 1 ) {
            for ( std::size_t i = 0; i < shape[ 0 ]; ++i )
                display( os << ( i ? ", " : "" ), value[ i ] );
        } else {
            for ( std::size_t i = 0; i < shape[ 0 ]; ++i )
                display( os << "\n  ", value( i ) );
        }
    } else if constexpr ( requires { for_each_item( value, []( const auto & ) {} ); } ) { // list
        std::size_t cpt = 0;
        for_each_item( value, [&]( const auto &item ) {
            display( os << ( cpt++ ? ", " : "" ), item );
        } );
    } else if constexpr ( requires { os << value; } ) { // operator<<
        os << value;
    } else {
        os << "TODO: display of " << typeid( value ).name();
    }
}

} // namespace sdot
