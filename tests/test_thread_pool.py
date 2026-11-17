import loom
from loom import Axis, RealTensor, ShapeVar
from loom.compilation.FfiCode import FfiCode
from errand import test
from loom.testing import need, host

# THE CPU THREAD POOL ( `cpp/runtime/cpu_thread_pool.cpp` ), from inside a kernel: the real runtime library every kernel links.
# The virtual threads of a launch are CLAIMED by whoever is free, the caller included, and a launch must neither run one twice
# nor return before all are done -- whatever the sizes, whoever launches, and however late a sleeping worker wakes into the next
# launch ( the race that made a worker claim a slot past the end of the last launch with the size of the next one: run twice, and
# the caller waited forever ).

_STRESS = r"""
    using namespace sdot;
    std::atomic<long> failures{ 0 };
    auto launcher = [&]( unsigned seed, int nb_launches ) {
        std::mt19937 rng( seed );
        for ( int l = 0; l < nb_launches; ++l ) {
            const int n = 1 + int( rng() % 64 );
            const int mode = int( rng() % 4 );        // 0: empty, 1: a few µs, 2: uneven, 3: after a pause ( the workers asleep )
            if ( mode == 3 )
                std::this_thread::sleep_for( std::chrono::microseconds( 200 + rng() % 800 ) );
            std::vector<std::atomic<int>> count( n );
            for ( auto &c : count )
                c = 0;
            std::atomic<int> running{ 0 };
            cpu_thread_pool().run_threads( n, [&]( int t ) {
                if ( t < 0 || t >= n ) { ++failures; return; }
                ++running;
                if ( mode == 1 ) { volatile double x = 0; for ( int i = 0; i < 2000; ++i ) x = x + i; }
                if ( mode == 2 && t % 7 == 0 ) std::this_thread::sleep_for( std::chrono::microseconds( 50 ) );
                ++count[ t ];
                --running;
            } );
            if ( running.load() != 0 )
                ++failures;
            for ( auto &c : count )
                if ( c.load() != 1 )
                    ++failures;
        }
    };
    launcher( 1, NB_LAUNCHES );
    std::thread a( launcher, 2, NB_LAUNCHES ), b( launcher, 3, NB_LAUNCHES );    // two launchers at once ( XLA does that )
    a.join();
    b.join();

    // a tiny launch after a pause: the caller does it alone, without waiting for the workers to wake
    double worst = 0;
    for ( int k = 0; k < 20; ++k ) {
        std::this_thread::sleep_for( std::chrono::milliseconds( 3 ) );
        auto s = std::chrono::steady_clock::now();
        cpu_thread_pool().run_threads( 14, []( int ) {} );
        worst = std::max( worst, std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - s ).count() );
    }
    args.outputs.nb_failures.fill_with( queue, double( failures.load() ) );
    args.outputs.worst_us.fill_with( queue, worst );
"""


if test( "every_virtual_thread_runs_once" ):
    need( "cpu" )
    nb_failures = RealTensor[ Axis( ShapeVar( 1 ), name = "num_f" ) ]()       # ( `failures` is a keyword of `ffi_call` )
    worst_us = RealTensor[ Axis( ShapeVar( 1 ), name = "num_w" ) ]()
    loom.ffi_call(
        "test_thread_pool_stress",
        FfiCode.inline( _STRESS.replace( "NB_LAUNCHES", "2000" ),
                        includes = [ "loom/support/kernels/CpuThreadPool.h", "atomic", "chrono", "random", "thread", "vector" ] ),
        nb_failures = loom.out( nb_failures ), worst_us = loom.out( worst_us ) )
    assert float( host( nb_failures.raw ).reshape( -1 )[ 0 ] ) == 0
    # ( milliseconds when the caller waited for sleeping workers; a few µs to a few tens now -- a loose bound for loaded machines )
    worst = float( host( worst_us.raw ).reshape( -1 )[ 0 ] )
    assert worst < 1500, worst
