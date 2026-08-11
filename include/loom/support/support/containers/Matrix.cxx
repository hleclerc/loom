#pragma once

// #include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "Matrix.h"
#include <utility>
#include <cmath>

namespace sdot {

#define UTP  template<class T,int ct_rows,int ct_cols>
#define DTP  Matrix<T,ct_rows,ct_cols>

UTP Matrix<T,ct_rows-1,ct_cols-1> DTP::without_row_and_col( auto wr, auto wc ) const {
    Matrix<T,ct_rows-1,ct_cols-1> res;
    for( PI r = 0; r < res.nb_rows(); ++r )
        for( PI c = 0; c < res.nb_cols(); ++c )
            res( r, c ) = operator()( r + ( r >= wr ), c + ( c >= wc ) );
    return res;
}

UTP DTP DTP::with_func( auto &&func ) {
    Matrix res;
    for( PI r = 0; r < ct_rows; ++r )
        for( PI c = 0; c < ct_cols; ++c )
            res( r, c ) = func( r, c );
    return res;
}

UTP DTP DTP::identity() {
    Matrix res;
    for( PI r = 0; r < ct_rows; ++r )
        for( PI c = 0; c < ct_cols; ++c )
            res( r, c ) = ( r == c );
    return res;
}

UTP DTP DTP::with_replaced_col( auto c, const auto &col ) const {
    Matrix res = *this;
    for( PI r = 0; r < nb_rows(); ++r )
        res( r, c ) = col[ r ];
    return res;
}

UTP auto DTP::diagonal() const {
    return Vector<T,min(ct_rows,ct_cols)>( Function(), [&]( auto i ) { return operator()( i, i ); } );
}

UTP T DTP::determinant() const {
    static_assert( ct_rows == ct_cols );
    if constexpr ( ct_rows == 1 ) {
        return operator()( 0, 0 );
    } else {
        T sgn = 1, res = 0;
        for( PI r = 0; r < nb_rows(); ++r, sgn = -sgn )
            res += sgn * operator()( r, 0 ) * without_row_and_col( r, 0 ).determinant();
        return res;
    }
}

UTP DTP DTP::cholesky() const {
    const PI nd = nb_rows();
    Matrix L( nd );
    for ( PI i = 0; i < nd; ++i )
        for ( PI j = 0; j < nd; ++j )
            L( i, j ) = T( 0 );

    for ( PI j = 0; j < nd; ++j ) {
        // diagonal
        T s = operator()( j, j );
        for ( PI k = 0; k < j; ++k ) s -= L( j, k ) * L( j, k );
        L( j, j ) = std::sqrt( s );

        // column below diagonal
        const T inv_ljj = T( 1 ) / L( j, j );
        for ( PI i = j + 1; i < nd; ++i ) {
            T t = operator()( i, j );
            for ( PI k = 0; k < j; ++k ) t -= L( i, k ) * L( j, k );
            L( i, j ) = t * inv_ljj;
        }
    }
    return L;
}

UTP Vector<T,ct_cols> DTP::solve_det( const auto &vec ) const {
    T d = determinant();
    T sgn = 1;
    Vector<T,ct_cols> res;
    for( PI c = 0; c < nb_rows(); ++c, sgn = -sgn )
        res[ c ] = with_replaced_col( c, vec ).determinant() / d;
    return res;
}

UTP Vector<T,ct_cols> DTP::solve_ge( const auto &mat, auto b ) {
    const PI n = ct_rows;
    // `mat` may be a Matrix or a bare TensorView: solve_ge copies into a working matrix anyway,
    // so it takes anything indexable as ( r, c ) and materializes its own `A`.
    Matrix A = with_func( [&]( PI r, PI c ) { return T( mat( r, c ) ); } );

    // forward elimination with partial pivoting
    for ( PI p = 0; p < n; ++p ) {
        PI pivot = p;
        for ( PI r = p + 1; r < n; ++r )
            if ( std::abs( A( r, p ) ) > std::abs( A( pivot, p ) ) )
                pivot = r;
        for ( PI c = p; c < n; ++c ) std::swap( A( p, c ), A( pivot, c ) );
        std::swap( b[ p ], b[ pivot ] );

        if ( A( p, p ) == T( 0 ) ) continue;  // zero pivot: degenerate row, leave as 0

        const T inv = T( 1 ) / A( p, p );
        for ( PI r = p + 1; r < n; ++r ) {
            const T factor = A( r, p ) * inv;
            for ( PI c = p + 1; c < n; ++c )
                A( r, c ) -= factor * A( p, c );
            b[ r ] -= factor * b[ p ];
        }
    }

    // back substitution (x initialised to 0 so zero-pivot rows stay 0)
    Vector<T,ct_cols> x;
    for ( PI i = 0; i < n; ++i )
        x[ i ] = T( 0 );
    for ( PI p = n; p-- > 0; ) {
        if ( A( p, p ) == T( 0 ) )
            continue;
        T s = b[ p ];
        for ( PI q = p + 1; q < n; ++q )
            s -= A( p, q ) * x[ q ];
        x[ p ] = s / A( p, p );
    }
    return x;
}

UTP DTP DTP::inverse() const {
    const PI n = nb_rows();
    Matrix A = *this;
    Matrix inv = with_func( []( PI r, PI c ) -> T { return r == c ? T(1) : T(0); } );

    for ( PI p = 0; p < n; ++p ) {
        // partial pivot
        PI pivot = p;
        for ( PI r = p + 1; r < n; ++r )
            if ( std::abs( A( r, p ) ) > std::abs( A( pivot, p ) ) )
                pivot = r;
        for ( PI c = 0; c < n; ++c )
            std::swap( A( p, c ), A( pivot, c ) );
        for ( PI c = 0; c < n; ++c )
            std::swap( inv( p, c ), inv( pivot, c ) );

        if ( A( p, p ) == T( 0 ) )
            continue;

        const T inv_diag = T( 1 ) / A( p, p );
        for ( PI c = 0; c < n; ++c ) {
            A( p, c ) *= inv_diag;
            inv( p, c ) *= inv_diag;
        }

        for ( PI r = 0; r < n; ++r ) {
            if ( r == p ) continue;
            const T f = A( r, p );
            for ( PI c = 0; c < n; ++c ) {
                A( r, c ) -= f * A( p, c );
                inv( r, c ) -= f * inv( p, c );
            }
        }
    }
    return inv;
}


#undef UTPH
#undef UTP
#undef DTP

} // namespace sdot
