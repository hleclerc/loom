// #include "../../src/cpp/sdot/support/containers/contiguous_strides.h"
// #include "../../src/cpp/sdot/support/hardware/MemorySpace_CpuRam.h"
// #include "../../src/cpp/sdot/support/hardware/Run.h"
#include <loom/support/containers/TensorView.h>
#include <loom/support/kernels/run_parallel.h>
#include <loom/support/algorithms/CartesianIndices.h>
#include <loom/support/algorithms/indices_of.h>
#include <loom/support/algorithms/reductions.h>
#include <loom/support/containers/Range.h>
// #include "sdot_test_matrix.h"
#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuHostMemorySpace.h>
#include "main.h"
#include <algorithm>
#include <numeric>

using namespace sdot;

// struct Test {
//     auto apply_values( auto &&func ) { return func( a ); }
//     static Test make_variant( int a ) { return { a }; }
//     void display( auto &ds ) const { ds << a; }
//     int a;
// };

// template<class Op,class Mt>
// struct LimitNbThread {
//     auto max_nb_threads( auto&&...args ) const { return mt( FORWARD( args )... ); }
//     auto operator()    ( auto&&...args ) const { return op( FORWARD( args )... ); }
//     Op   op;
//     Mt   mt;
// };
//

/// nom
DEFINE_AXIS( row_ );
DEFINE_AXIS( col_ );
DEFINE_AXIS( dim );


TEST_CASE( "TensorView — indexation (positions + axes nommés)", "" ) {
    using TF = double;
    TF data[] = { 1, 2, 3, 4 }; // 2x2 row-major : [[1,2],[3,4]]

    // axe 0 anonyme, axe 1 nommé `dim`
    auto t = tensor_view( data, tuple( 2, 2 ), tuple( UnnamedAxis{}, dim ) );

    // indexation mixte position / nom
    CHECK( t( 0, dim = 0 ).value() == 1 );
    CHECK( t( 0, dim = 1 ).value() == 2 );
    CHECK( t( 1, dim = 0 ).value() == 3 );
    CHECK( t( 1, 1 ).value() == 4 );          // tout positionnel

    // squeeze direct (1 et 2 args) et row
    CHECK( t.squeeze( 0_c, 1 ).squeeze( dim = 0 ).value() == 3 );
    CHECK( t.row( 0 )( dim = 1 ).value() == 2 );

    // ref() écrit en place
    t( 1, dim = 1 ).ref() = 40;
    CHECK( data[ 3 ] == 40 );
}


TEST_CASE( "TensorView — itérateur rang 1 (std::iota / std::sort)", "" ) {
    // contigu : iota écrit en place, la distance end - begin est correcte, sort réordonne.
    double data[] = { 9, 9, 9, 9, 9 };
    auto t = tensor_view( data, tuple( 5 ) );

    CHECK( t.end() - t.begin() == 5 );                 // operator- (sens + unité)
    std::iota( t.begin(), t.end(), 0.0 );              // les écritures persistent
    CHECK( data[ 0 ] == 0 && data[ 2 ] == 2 && data[ 4 ] == 4 );

    std::sort( t.begin(), t.end(), []( double a, double b ) { return a > b; } );
    CHECK( data[ 0 ] == 4 && data[ 4 ] == 0 );

    // stride non contigu (16 octets = un double sur deux) : on ne trie que les indices pairs,
    // les impairs restent intacts -- valide que le stride est bien interprété.
    double raw[] = { 5, -1, 3, -1, 1, -1 };
    auto s = tensor_view( raw, tuple( 3 ), tuple( UnnamedAxis{} ), tuple( 16 ) );
    std::sort( s.begin(), s.end(), []( double a, double b ) { return a < b; } );
    CHECK( raw[ 0 ] == 1 && raw[ 2 ] == 3 && raw[ 4 ] == 5 );
    CHECK( raw[ 1 ] == -1 && raw[ 3 ] == -1 && raw[ 5 ] == -1 );
}


TEST_CASE( "TensorView — strides non contigus, rang 3, offset", "" ) {
    // 2x3 row-major : [[1,2,3],[4,5,6]] ; vue transposée 3x2 via strides (8, 24) octets
    double m[] = { 1, 2, 3, 4, 5, 6 };
    auto tT = tensor_view( m, tuple( 3, 2 ), tuple( row_, col_ ), tuple( 8, 24 ) );
    CHECK( tT( 0, 0 ).value() == 1 );
    CHECK( tT( 2, 1 ).value() == 6 );
    CHECK( tT( row_ = 0, col_ = 1 ).value() == 4 ); // indexation par noms, ordre indifférent
    CHECK( tT( col_ = 1, row_ = 2 ).value() == 6 );

    // rang 3 contigu 2x2x2 : c[i][j][k] = 4i + 2j + k
    double c[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    auto t3 = tensor_view( c, tuple( 2, 2, 2 ) );
    CHECK( t3( 1, 0, 1 ).value() == 5 );
    CHECK( t3( 0, 1, 1 ).value() == 3 );

    // offset : sous-vue à partir de l'index 1 sur l'axe 0
    double v[] = { 10, 20, 30, 40 };
    auto o = tensor_view( v, tuple( 4 ) ).offset( 1 ); // [20,30,40]
    CHECK( o.shape( 0_c ) == 3 );
    CHECK( o( 0 ).value() == 20 );
    CHECK( o( 2 ).value() == 40 );
}

TEST_CASE( "TensorView — fill_with / for_each_scalar", "" ) {
    double d[ 6 ] = {};
    auto t = tensor_view( d, tuple( 2, 3 ) );

    CHECK_REPR( t.indices_col_ordering( 0 ), tuple( 0, 0 ) );
    CHECK_REPR( t.indices_col_ordering( 1 ), tuple( 0, 1 ) );
    CHECK_REPR( t.indices_col_ordering( 2 ), tuple( 0, 2 ) );
    CHECK_REPR( t.indices_col_ordering( 3 ), tuple( 1, 0 ) );
    CHECK_REPR( t.indices_col_ordering( 4 ), tuple( 1, 1 ) );

    t.fill_with( 7 );
    for ( double x : d )
        CHECK( x == 7 );

    double s = 0;
    t.for_each_scalar( [&]( auto e ) { s += e.value(); } );
    CHECK( s == 42 );

    // fill d'une sous-vue (ligne 1 seulement)
    double d2[ 6 ] = {};
    tensor_view( d2, tuple( 2, 3 ) ).row( 1 ).fill_with( 5 );
    CHECK( d2[ 0 ] == 0 );
    CHECK( d2[ 3 ] == 5 );
    CHECK( d2[ 5 ] == 5 );
}

TEST_CASE( "TensorView — fill_with avec contextes (run_parallel)", "" ) {
    auto ql = tuple( CpuQueue() );

    // handle ignoré -> RAII : wait à la destruction du temporaire (synchrone)
    double d[ 6 ] = {};
    auto t = tensor_view( d, tuple( 2, 3 ) );
    t.fill_with( ql, 9 );
    for ( double x : d )
        CHECK( x == 9 );

    // handle gardé -> asynchrone, puis wait() explicite (consomme l'event)
    double e[ 4 ] = {};
    auto u = tensor_view( e, tuple( 4 ) );
    {
        auto h = u.fill_with( ql, 3 );
        h.wait();
    }
    for ( double x : e )
        CHECK( x == 3 );

    // chaînage : le 2e run dépend du 1er via after() (qui consomme h1)
    double f[ 4 ] = {};
    auto w = tensor_view( f, tuple( 4 ) );
    auto h1 = w.fill_with( ql, 1 );
    auto h2 = w.fill_with( ql, after( h1 ), 2 );
    h2.wait();
    for ( double x : f )
        CHECK( x == 2 );
}

TEST_CASE( "TensorView — opérateurs élémentaires (boucle simple)", "" ) {
    double a[] = { 1, 2, 3, 4 };
    double b[] = { 10, 20, 30, 40 };
    auto ta = tensor_view( a, tuple( 2, 2 ) );
    auto tb = tensor_view( b, tuple( 2, 2 ) );

    ta += 1;                       // broadcast scalaire -> {2,3,4,5}
    CHECK( a[ 0 ] == 2 && a[ 3 ] == 5 );
    ta *= 2;                       // {4,6,8,10}
    CHECK( a[ 0 ] == 4 && a[ 3 ] == 10 );

    ta += tb;                      // élémentaire (même forme) -> {14,26,38,50}
    CHECK( a[ 0 ] == 14 && a[ 1 ] == 26 && a[ 2 ] == 38 && a[ 3 ] == 50 );

    ta = tb;                       // copie profonde -> a <- b
    CHECK( a[ 0 ] == 10 && a[ 3 ] == 40 );

    ta = 0;                        // operator= scalaire (broadcast)
    for ( double x : a )
        CHECK( x == 0 );

    tb.row( 0 ) -= 5;              // sur une sous-vue -> b = {5,15,30,40}
    CHECK( b[ 0 ] == 5 && b[ 1 ] == 15 && b[ 2 ] == 30 );
}

TEST_CASE( "TensorView — run_parallel élémentaire via indices_of (Mut/Inp)", "" ) {
    auto ql = tuple( CpuQueue() );
    double a[] = { 1, 2, 3, 4 };
    double b[] = { 10, 20, 30, 40 };
    auto ta = tensor_view( a, tuple( 2, 2 ) );
    auto tb = tensor_view( b, tuple( 2, 2 ) );

    // a += b en parallèle, écrit « à la main » hors de TensorView (tags Mut/Inp explicites)
    run_parallel( ql, indices_of( ta, tb ),
        []( auto indices, auto a, auto b ) { a[ indices ] += b[ indices ]; },
        MutList(), ta, InpList(), tb
    ); // QueueEvent ignoré -> wait RAII

    CHECK( a[ 0 ] == 11 && a[ 1 ] == 22 && a[ 2 ] == 33 && a[ 3 ] == 44 );
}

TEST_CASE( "TensorView — indices_of intersection (formes différentes)", "" ) {
    auto ql = tuple( CpuQueue() );
    double a[] = { 1, 2, 3, 4, 5, 6 };  // 2x3
    double b[] = { 10, 20, 30, 40, 50, 60 }; // 3x2
    auto ta = tensor_view( a, tuple( 2, 3 ) );
    auto tb = tensor_view( b, tuple( 3, 2 ) );

    // intersection des parcours -> (min(2,3), min(3,2)) = (2,2) : seuls les indices communs
    run_parallel( ql, indices_of( ta, tb ),
        []( auto indices, auto a, auto b ) { a[ indices ] += b[ indices ]; },
        MutList(), ta, InpList(), tb
    );

    // a[i][j] += b[i][j] pour i<2, j<2 ; les colonnes j>=2 de a restent intactes
    CHECK( a[ 0 ] == 11 && a[ 1 ] == 22 && a[ 2 ] == 3 );  // ligne 0 : 1+10, 2+20, 3 (intact)
    CHECK( a[ 3 ] == 34 && a[ 4 ] == 45 && a[ 5 ] == 6 );  // ligne 1 : 4+30, 5+40, 6 (intact)
}

TEST_CASE( "TensorView — réductions sum/max (SYCL reduction)", "" ) {
    auto ql = tuple( CpuQueue() );
    double d[] = { 1, 2, 3, 4, 5, 6 };
    auto t = tensor_view( d, tuple( 2, 3 ) );

    CHECK( sum( ql, t ) == 21 );          // 1+..+6
    CHECK( max( ql, t ) == 6 );
    CHECK( sum( ql, t.row( 1 ) ) == 15 ); // 4+5+6 sur une sous-vue
}

TEST_CASE( "TensorView — fill_with avec contextes (run_parallel) et shapes éclatées", "" ) {
    auto ql = tuple( CpuQueue() );

    double d[ 4*4 ] = {};
    for( PI i = 0; i < 4*4; ++i )
        d[ i ] = i;

    auto t = tensor_view( d, tuple( 2, 4 ), tuple( row_, col_ ), tuple( sizeof( double ) * 2, sizeof( double ) * 4 ) );
    CHECK_REPR( t( 1, 1 ), 6 );
    t.fill_with( ql, 100 );

    for( PI i = 0; i < 4*4; i += 2 ) {
        CHECK_REPR( d[ i + 0 ], 100 );
        CHECK_REPR( d[ i + 1 ], i + 1 );
    }
}

DEFINE_AXIS( vmap_0 );

TEST_CASE( "TensorView — indexation par un multi-indice de batch (axes nommés, optionnels)", "" ) {
    // what a `vmap` produces: a batch axis, carried by the mapped values and by them only.
    double b[ 2*3 ] = { 1, 2, 3, 4, 5, 6 };
    auto batched = tensor_view( b, tuple( 2, 3 ), tuple( vmap_0, col_ ) );

    double s[ 3 ] = { 10, 20, 30 };
    auto shared = tensor_view( s, tuple( 3 ), tuple( col_ ) ); // not mapped along vmap_0

    CartesianIndices<Tuple<SI>,Tuple<_vmap_0>> items{ tuple( SI( 2 ) ) };
    CHECK( items.size() == 2 );

    // an item is a NAMED coordinate (`vmap_0 = 1`), so it selects the axis by name: the batched
    // view loses its batch axis, the shared one simply lets the index through.
    auto batch_index = items[ 1 ];
    CHECK( batched( batch_index, col_ = 0 ).value() == 4 );
    CHECK( shared ( batch_index, col_ = 0 ).value() == 10 );

    // unbatched: the empty multi-index indexes nothing -- the same body works either way.
    CartesianIndices<Tuple<>> no_batch;
    CHECK( no_batch.size() == 1 );
    CHECK( shared( no_batch[ 0 ], col_ = 2 ).value() == 30 );
}
