// #include "../../src/cpp/sdot/support/containers/CartesianProduct.h"
// #include "../../src/cpp/sdot/support/containers/Range.h"
#include <sdot/support/containers/Tuple.h>
#include "main.h"
#include <array>

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "AxisValues", "" ) {
    auto t = tuple( 1, 2, 3u );
    auto c = tuple( 1, 2_c, 3u );
    auto z = tuple();

    SECTION( "repr" ) {
        CHECK_REPR( t, array( 1, 2, 3 ) );
    }

    SECTION( "without_index" ) {
        CHECK_REPR( t.without_index( Ct<int,0>() ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,1>() ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,2>() ), array( 1, 2 ) );

        CHECK_REPR( t.without_index( 0 ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( 1 ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( 2 ), array( 1, 2 ) );
    }

    SECTION( "operator[]" ) {
        CHECK_REPR( c[ 0_c ], 1   );
        CHECK_REPR( c[ 1_c ], 2_c );
        CHECK_REPR( c[ 2_c ], 3u  );

        CHECK_REPR( c[ 0 ], 1 );
        CHECK_REPR( c[ 1 ], 2 );
        CHECK_REPR( c[ 2 ], 3 );
    }

    SECTION( "with_appended_value" ) {
        CHECK_REPR( t.with_appended_value( 17 ), array( 1, 2, 3, 17 ) );
        CHECK_REPR( z.with_appended_value( 17 ), array( 17 ) );
    }

    SECTION( "map" ) {
        CHECK_REPR( map( c, []( auto v ) { return 2_c * v; } ), tuple( 2, 4_c, 6u ) );
    }

    SECTION( "operator==" ) {
        CHECK_REPR( t == c, 1 );
        CHECK_REPR( tuple( 1_c ) == tuple( 1_c ), (Ct<bool,true>()) );
        CHECK_REPR( tuple( 1_c ) == tuple( 2_c ), (Ct<bool,false>()) );
        CHECK_REPR( tuple( 1_c, 2_c ) == tuple( 2_c ), (Ct<bool,false>()) );
        CHECK_REPR( tuple( 1_c, 2_c ) == tuple( 1_c, 2_c ), (Ct<bool,true>()) );
    }
}
