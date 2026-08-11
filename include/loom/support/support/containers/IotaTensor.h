#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// A storageless SYMBOLIC tensor whose value at a multi-index is its LAST index: `t( d, k )`
/// reads as `k`. It is what an unbound `knots` becomes by default (the documented `0, 1, 2, ...`
/// per dimension), the counterpart of `NoneTensor`/`ZeroTensor` for a value that is a plain
/// index rather than absent or zero.
///
/// Like the others it carries no data and says so in its TYPE: `is_valid()` is a `Ct<bool,true>`
/// (a legitimate value, so `with_defaults`' recursion stops substituting once knots is this) and
/// `surely_null()` is `Ct<bool,false>` (its entries are generally non-zero). It is READ-ONLY and
/// indexed POSITIONALLY -- enough for `Image::measure`, which reads `knots( axis, k )`; it is not
/// a general tensor. `TF` is the scalar the indices are returned as.
template<class _TF>
struct IotaTensor {
    using            TF                     = _TF;

    constexpr auto   is_valid               () const { return Ct<bool,true >(); }
    constexpr auto   surely_null            () const { return Ct<bool,false>(); }

    /// value at a multi-index = its last coordinate (fully indexed). No index at all reads as 0.
    constexpr TF     operator()             ( auto &&...index ) const {
        TF last = 0;
        ( ( last = TF( index ) ), ... );
        return last;
    }

    // as a `run_parallel` argument: no storage backs it, so it crosses into the kernel unchanged,
    // at no cost, whatever the queue and the io category (mirrors `ZeroTensor`/`NoneTensor`).
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "IotaTensor"; }
};

} // namespace sdot
