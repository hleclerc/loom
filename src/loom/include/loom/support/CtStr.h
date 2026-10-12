#pragma once

#include <loom/support/common_macros.h> // HD

#include <cstddef>

namespace sdot {

/// Constant string usable as a template parameter (NTTP "fixed-string", C++20).
/// `FixedStr` is a structural type -> two distinct values give distinct
/// specializations (this is what makes `CtStr<"x">` and `CtStr<"y">` different).
template<std::size_t N>
struct FixedStr {
    char data[ N ] {};

    HD constexpr FixedStr( const char (&s)[ N ] ) { for ( std::size_t i = 0; i < N; ++i ) data[ i ] = s[ i ]; }

       constexpr bool operator==( const FixedStr & ) const = default;

    static constexpr std::size_t size = N; ///< includes the trailing '\0'
};

/// Tag of a string known at compile time: `CtStr<"x">`, `CtStr<"">` -> distinct types.
template<FixedStr S>
struct CtStr {
    static constexpr auto str = S;

    HD void display( auto &os ) const { os << S.data; }
};

} // namespace sdot
