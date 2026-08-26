#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// A rank-0 value carried BY VALUE: no buffer, no pointer, no memory space.
///
/// This is what a plain (non-ragged) count crosses as when the host already knows it. A count is
/// one integer; sending it through a device buffer costs an allocation, a transfer and a
/// dereference per read, and buys nothing -- the value is uniform over the whole call and cannot
/// change during it. So it travels as an XLA FFI attribute and lands here, in the kernel's own
/// registers.
///
/// The contrast on either side:
/// * `TensorView`  -- a POINTER into memory the kernel dereferences. What a count must still be
///   when it is ragged (one count per segment) or when the KERNEL is the one writing it.
/// * `Ct<SI,n>`    -- the value in the TYPE, compile-time known, part of the library hash. What
///   a `CtShapeVar` is: it enables specialization, at the price of a library per value.
/// `ScalarValue` sits between them: a runtime value, one compiled library for every value.
///
/// It answers the same reading interface as the view it replaces (`value()`, indexing), so
/// `ShapeVarView` wraps either without knowing which. Read-only on purpose: there is nowhere for
/// a kernel to write a result into a value passed by copy, and an input count is never written.
template<class T>
struct ScalarValue {
    T                v;

    T                value                  () const { return v; }

    /// selecting axes on a value that has none: nothing to select, so it gives back itself --
    /// which is what lets a batch index be applied uniformly (`cell.nb_dims( batch_index )`).
    constexpr auto   operator()             ( auto &&.../*index*/ ) const { return *this; }

    /// as a `run_parallel` argument: the value is in the argument itself, so there is nothing in
    /// memory to make accessible -- it crosses into the kernel untouched, at zero cost.
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "ScalarValue(" << v << ")"; }
};

} // namespace sdot
