// #include "../../src/cpp/sdot/support/containers/CartesianProduct.h"
// #include "../../src/cpp/sdot/support/containers/Range.h"
#include <sdot/support/containers/TupleRep.h>
#include "main.h"
#include <array>

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "AxisValues", "" ) {
    auto t = tuple_rep( 1, 2 );
    INFO( t[ 0 ] );
    INFO( t[ 1 ] );
    INFO( t[ 2 ] );
}
