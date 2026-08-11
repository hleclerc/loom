// Smoke test for the AdaptiveCpp toolchain: a plain SYCL vector add.
// Compiled with the `acpp` driver (see sdot.compilation.adaptive_cpp). Returns non-zero
// on mismatch so the test runner can detect failure.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::size_t n = 1024;

    std::vector<float> a( n ), b( n ), c( n, 0.f );
    for ( std::size_t i = 0; i < n; ++i ) {
        a[ i ] = float( i );
        b[ i ] = float( 2 * i );
    }

    sycl::queue q;
    std::printf( "Running on: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str() );

    {
        sycl::buffer<float> ba( a.data(), sycl::range<1>( n ) );
        sycl::buffer<float> bb( b.data(), sycl::range<1>( n ) );
        sycl::buffer<float> bc( c.data(), sycl::range<1>( n ) );

        q.submit( [&]( sycl::handler & h ) {
            sycl::accessor aa( ba, h, sycl::read_only );
            sycl::accessor ab( bb, h, sycl::read_only );
            sycl::accessor ac( bc, h, sycl::write_only, sycl::no_init );
            h.parallel_for( sycl::range<1>( n ), [=]( sycl::id<1> i ) {
                ac[ i ] = aa[ i ] + ab[ i ];
            } );
        } );
    } // buffers flush back to host vectors on destruction

    for ( std::size_t i = 0; i < n; ++i ) {
        const float expected = float( i ) + float( 2 * i );
        if ( c[ i ] != expected ) {
            std::printf( "MISMATCH at %zu: %f != %f\n", i, c[ i ], expected );
            return 1;
        }
    }

    std::printf( "OK: SYCL vector add of %zu elements\n", n );
    return 0;
}
