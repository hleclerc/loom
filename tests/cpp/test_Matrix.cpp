#include <sdot/support/containers/Matrix.h>
#include "main.h"

using namespace sdot;

TEST_CASE( "Matrix", "" ) {
    SECTION( "constant" ) {
        Matrix<double,3> v( Function(), []( auto r, auto c ) { return r * 10 + c + ( r == c ) + 1; } );
        CHECK_REPR( v.determinant(), -23.0 );
    }
}
