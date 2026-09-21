#pragma once

#include <loom/support/common_macros.h> // HD

#include "Vector.h"
#include "Tuple.h"

namespace sdot {

// _size
template<class T,int ct_rows,int ct_cols=ct_rows>
class Matrix {
public:
    struct                EigenSystem             { Vector<T,ct_rows> values; /* ascending order */ Matrix<T,ct_rows,ct_cols> vectors; /* row i = eigenvector i */ };
    using                 value_type              = T;
    using                 Content                 = Vector<T,ct_rows*ct_cols>;

    /* */                 HD Matrix               ( FillWith, auto &&value ) : _content( FillWith(), value ) {}
    /* */                 HD Matrix               ( Function, auto &&func ) : _content( Function(), [&]( auto index ) { return func( index / ct_cols, index % ct_cols ); } ) {}
    /* */                 HD Matrix               () {}

    HD static Matrix      with_func               ( auto &&func );
    HD static Matrix      identity                ();

    HD const T&           operator()              ( auto r, auto c ) const { return _content[ r * ct_rows + c ]; }
    HD T&                 operator()              ( auto r, auto c ) { return _content[ r * ct_rows + c ]; }
    HD auto               operator()              ( auto r ) const { return Vector<T,ct_cols>( Function(), [&]( auto c ) { return operator()( r, c ); } ); }

    HD auto               without_row_and_col     ( auto r, auto c ) const -> Matrix<T,ct_rows-1,ct_cols-1>;
    HD auto               with_replaced_col       ( auto c, const auto &col ) const -> Matrix;
    HD EigenSystem        eigen_system            () const;
    HD T                  determinant             () const;
    HD auto               diagonal                () const;
    HD Matrix             cholesky                () const;  ///< returns L s.t. *this = L * L^T (H must be SPD)
    HD Matrix             inverse                 () const;  ///< Gauss-Jordan on [A | I]; zero pivot row → identity row in result

    HD Vector<T,ct_cols>  solve_det               ( const auto &b ) const;
    HD static Vector<T,ct_cols> solve_ge          ( const auto &mat, auto b ); ///< Gaussian elimination with partial pivoting on a copy of `mat` (Matrix or TensorView); zero pivot → x[p]=0 (handles degenerate cells)

    HD auto               is_valid                () const { return Ct<bool,true>(); }

    HD constexpr auto     nb_rows                 () const { return Ct<int,ct_rows>(); }
    HD constexpr auto     nb_cols                 () const { return Ct<int,ct_cols>(); }
    HD constexpr auto     shape                   () const { return tuple( nb_rows(), nb_cols() ); }
    HD constexpr auto     shape                   ( auto index ) const { return shape()[ index ]; }

    HD const T*           data                    () const { return _content.data(); }
    HD T*                 data                    () { return _content.data(); }

    HD auto               begin                   () const { return _content.begin(); }
    HD auto               begin                   () { return _content.begin(); }
    HD auto               end                     () const { return _content.end(); }
    HD auto               end                     () { return _content.end(); }

    Content               _content;
};

} // namespace sdot

#include "Matrix.cxx"
