#pragma once

#include <cstddef>

namespace sdot {

/// Chaîne constante utilisable comme paramètre template (NTTP « fixed-string », C++20).
/// `FixedStr` est un type structurel -> deux valeurs distinctes donnent des spécialisations
/// distinctes (c'est ce qui rend `CtStr<"x">` et `CtStr<"y">` différents).
template<std::size_t N>
struct FixedStr {
    char data[ N ] {};

    constexpr FixedStr( const char (&s)[ N ] ) { for ( std::size_t i = 0; i < N; ++i ) data[ i ] = s[ i ]; }

    constexpr bool operator==( const FixedStr & ) const = default;

    static constexpr std::size_t size = N; ///< inclut le '\0' final
};

/// Tag de chaîne connue à la compilation : `CtStr<"x">`, `CtStr<"">` -> types distincts.
template<FixedStr S>
struct CtStr {
    static constexpr auto str = S;

    void display( auto &os ) const { os << S.data; }
};

} // namespace sdot
