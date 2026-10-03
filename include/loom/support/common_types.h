#pragma once

#include "common_macros.h" // HD
#include "ASSERT.h" // IWYU pragma: export
#include "TODO.h" // IWYU pragma: export
#include "INFO.h" // IWYU pragma: export

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sdot {

/// The value an OUTPUT buffer is filled with under `LOOM_ZERO_OUTPUTS=poison`: what a
/// never-written element will be worth.
///
/// Seeding with zero makes a partially written output HARMLESS (see
/// `CallArg_Tensor.cpp_seed_member`) -- but it also makes it CREDIBLE: zero is very often a
/// plausible value, and a forgotten write then passes the tests silently. Poison does not
/// let it through: a NaN propagates through everything it touches, and the chosen integer is big
/// enough to push out of bounds any loop it would end up bounding.
///
/// It is a DEBUG tool, not a default: the default stays zero, which is safe.
template<class TF>
HD constexpr TF poison_value() {
    if constexpr ( std::is_floating_point_v<TF> )
        return std::numeric_limits<TF>::quiet_NaN();
    else
        return TF( 0x7B7A7978 );
}

using FP64 = double;
using FP32 = float;

using PI8  = std::uint8_t;
using PI32 = std::uint32_t;
using PI64 = std::uint64_t;

using SI8  = std::int8_t;
using SI32 = std::int32_t;
using SI64 = std::int64_t;

using SI = long long;
using PI = std::size_t;

// ctor args
struct SizeAndCtorArgs {};
struct Function {};
struct Reserved {};
struct FillWith {};
struct Values {};
struct Shape {};
struct Rank {};
struct Size {};

} // namespace sdot
