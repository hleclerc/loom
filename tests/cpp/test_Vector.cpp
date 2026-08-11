#include <loom/support/containers/Vector.h>
#include <loom/support/common_macros.h>
#include <array>
#include "main.h"

using namespace sdot;

auto array( auto a, auto...values ) {
    return std::array<DECAYED_TYPE_OF( a ),1+sizeof...( values )>{ a, values... };
}

TEST_CASE( "Vector", "" ) {
    SECTION( "constant" ) {
        Vector<double,3> v( Values(), 1, 2, 3 );
        CHECK_REPR( v, array( 1, 2, 3 ) );
        CHECK_REPR( v.with_pushed_value( 17 ), array( 1, 2, 3, 17 ) );
        CHECK_REPR( v.without_index( 1 ), array( 1, 3 ) );
        CHECK_REPR( v.size(), 3_c );
    }
}
