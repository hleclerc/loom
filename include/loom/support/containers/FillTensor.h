#pragma once

#include "../common_types.h"
#include "Tuple.h"
#include "../Ct.h"

namespace sdot {

/// A tensor whose every element reads as the SAME runtime value, with no per-element storage: one
/// scalar (a rank-0 buffer) backs the value, and indexing ignores the index and yields it.
///
/// Same spirit as `ZeroTensor` (a storageless, read-only VALUE known through its TYPE), with two
/// differences: the value is NOT baked into the type -- it is read through `data` at run time, so the
/// SAME compiled kernel serves any fill value (a `1/n` never recompiles) -- and it carries its logical
/// `Shape` (runtime extents), so it honours the full tensor contract (`size()`, `shape()`). The
/// extents are not derived from any storage of its own (there is none); the boundary fills them in
/// from a real buffer that shares the same axes (see `CallArg_Tensor` fill lowering). Read only: no
/// `operator=` (nowhere to write a broadcast). Lowered from a symbolic `Tensor.full( v )`.
template<class _TF, class _Shape, class _AxisNames>
struct FillTensor {
    using            TF                     = _TF;
    using            Shape                  = _Shape;
    using            AxisNames              = _AxisNames;
    SCInt            ct_rank                = Shape::ct_size;

    const TF        *data;                  ///< points at the ONE scalar every element reads as
    Shape            _shape;                ///< logical extents (filled from a sibling real buffer)

    constexpr auto   is_valid               () const { return Ct<bool,true>(); } ///< a real value, storageless

    auto             shape                  ( auto d ) const { return _shape[ d ]; }
    Shape            shape                  () const { return _shape; }
    auto             size                   () const { static_assert( ct_rank == 1, "size() is for rank 1" ); return _shape[ Ct<int,0>() ]; }

    // indexing yields a rank-0 fill over the same scalar; reading it yields the value.
    constexpr auto   operator()             ( auto &&... ) const { return FillTensor<TF, Tuple<>, Tuple<>>{ data, {} }; }
    TF               value                  () const { return *data; }
    /* */            operator TF            () const { return *data; }

    // as a `run_parallel` argument: the scalar is already where the kernel runs (an FFI input XLA put
    // on the device), so it crosses unchanged, like `ZeroTensor`.
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "FillTensor(" << *data << ")"; }
};

} // namespace sdot
