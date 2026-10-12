#pragma once

// the members + the generated methods ( `operator()`, `kernel_form`, ... ) as macros
// that this struct drops in, written into the include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/CsrTensor.h>
#include <loom/support/common_macros.h>

namespace sdot {

/// A RAGGED TENSOR AS CSR: `offsets` says where each row starts in `values`, and `values`
/// is the flat list of all the content.
///
///     offsets  [ 0, 2, 2, 5 ]        four bounds for three rows
///     values   [ a, b, c, d, e ]
///     => row 0 = { a, b }, row 1 = {}, row 2 = { c, d, e }
///
/// WHAT IT TRADES AGAINST PADDING. A padded ragged costs `rows x longest_row`;
/// a CSR costs exactly what is useful. Measured on `examples/splats`: the padded form costs from x2.38 to
/// x5.08 the memory of the CSR. The price is one more PASS at construction -- count, then
/// fill, the offsets being the prefix sum of the counts ( `loom.cumsum( ..., exclusive = True )` )
/// -- and a HOST read of the total, which sizes `values`.
///
/// `offsets` carries `nb_rows + 1` bounds and not `nb_rows` counts: the size of a row is then
/// a SUBTRACTION of two neighbors, with no extra array, and the last bound is the total.
SDOT_TEMPLATE_DECL_FOR_CsrTensor
struct CsrTensor {
    SDOT_ATTRIBUTES_OF_CsrTensor

    using TF = DECAYED_TYPE_OF( values )::TF;

    /// where row `row` starts in `values`
    HD SI row_begin( SI row ) const { return SI( offsets( num_bound = row ) ); }

    /// how many elements row `row` carries
    HD SI row_size( SI row ) const { return SI( offsets( num_bound = row + 1 ) ) - row_begin( row ); }

    /// how many rows -- the last bound is not one
    HD SI nb_rows_of() const { return offsets.size( num_bound ) - 1; }

    /// `a( i, j )`: THE `i` IS LOOKED UP IN THE OFFSETS, the `j` indexes within the row. This is the
    /// only thing a user needs to know about the CSR, and it reads like a two-index
    /// array.
    ///
    /// Non-template and exactly two `SI`: the generated variadic `operator()` ( the one that
    /// indexes the WHOLE aggregate, `csr( batch_index )` ) is a template, so this overload wins
    /// for two integers and leaves it the rest.
    HD decltype( auto ) operator()( SI row, SI slot ) const { return values( num_slot = row_begin( row ) + slot ); }

    /// the same, under a name, for a caller who prefers to spell it out
    HD decltype( auto ) at( SI row, SI slot ) const { return operator()( row, slot ); }
};

}
