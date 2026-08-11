#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// A tensor that is NOT there: an attribute this call did not bind (no value given, and not
/// declared as an output -- an optional field the kernel does not use).
///
/// The absence is carried by the TYPE, not by a null pointer to be tested at runtime: a
/// `NoneTensor` has no data, no shape value, nothing to dereference. It keeps the DECLARED
/// `TF` / `Shape` / `AxisNames` though, so generic code can still read what the attribute
/// would have been.
///
/// A kernel discriminates at compile time -- `is_valid()` returns a `Ct<bool,false>`, so
/// `if constexpr ( DECAYED_TYPE_OF( t.is_valid() )::value )` compiles the branch away, and a
/// `static_assert` can forbid touching it at all.
template<class _TF, class _Shape, class _AxisNames>
struct NoneTensor {
    using            TF                     = _TF;
    using            Shape                  = _Shape;
    using            AxisNames              = _AxisNames;
    SCInt            ct_rank                = Shape::ct_size;

    constexpr auto   is_valid               () const { return Ct<bool,false>(); } ///< no data at all
    constexpr auto   surely_null            () const { return Ct<bool,true >(); } ///< nothing to read: reads nothing but zero

    /// selecting axes on something that is not there gives... something that is not there. What
    /// this buys is that a batch index applies to a WHOLE aggregate (`cell( batch_index )`) even
    /// when one of its attributes is unbound: absence stays absence, and stays a compile-time fact.
    constexpr auto   operator()             ( auto &&.../*index*/ ) const { return *this; }

    // as a `run_parallel` argument: there is no storage to make available anywhere, so it
    // crosses into the kernel unchanged, at no cost, whatever the queue and the io category.
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "NoneTensor"; }
};

} // namespace sdot
