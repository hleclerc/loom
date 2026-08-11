#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// A tensor that IS there but is symbolically zero: every element reads as 0, and no storage
/// backs it.
///
/// Like `NoneTensor`, it says so in its TYPE rather than through a runtime test, so an
/// algorithm can drop a whole term at compile time (`surely_null()` is a `Ct<bool,true>`)
/// instead of multiplying by a buffer of zeros. Unlike `NoneTensor`, it is a legitimate
/// VALUE: reading it is well defined (it yields 0), only writing it is not.
template<class _TF, class _Shape, class _AxisNames>
struct ZeroTensor {
    using            TF                     = _TF;
    using            Shape                  = _Shape;
    using            AxisNames              = _AxisNames;
    SCInt            ct_rank                = Shape::ct_size;

    constexpr auto   is_valid               () const { return Ct<bool,true>(); } ///< a real value, merely a storageless one
    constexpr auto   surely_null            () const { return Ct<bool,true>(); }

    // indexing a zero tensor yields a zero tensor (of rank 0 once fully indexed); reading it
    // yields 0. There is no `operator=`: a zero tensor is not somewhere one writes.
    constexpr auto   operator()             ( auto &&...  ) const { return ZeroTensor<TF, Tuple<>, Tuple<>>(); }
    constexpr TF     value                  () const { return 0; }
    constexpr        operator TF            () const { return 0; }

    // as a `run_parallel` argument: no storage backs it, so it crosses into the kernel
    // unchanged, at no cost, whatever the queue and the io category.
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "ZeroTensor"; }
};

} // namespace sdot
