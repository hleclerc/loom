#pragma once

#include "common_macros.h" // HD
#include "ASSERT.h" // IWYU pragma: export
#include "TODO.h" // IWYU pragma: export
#include "INFO.h" // IWYU pragma: export

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sdot {

/// La valeur dont on remplit un tampon de SORTIE sous `LOOM_ZERO_OUTPUTS=poison` : ce qu'un
/// élément jamais écrit vaudra.
///
/// Semer à zéro rend une sortie partiellement écrite INOFFENSIVE (voir
/// `CallArg_Tensor.cpp_seed_member`) -- mais ça la rend aussi CRÉDIBLE : zéro est très souvent une
/// valeur plausible, et une écriture oubliée passe alors les tests en silence. Le poison ne la
/// laisse pas passer : un NaN se propage dans tout ce qu'il touche, et l'entier choisi est assez
/// gros pour faire sortir des bornes toute boucle qu'il viendrait à borner.
///
/// C'est un outil de DEBUG, pas un défaut : le défaut reste zéro, qui est sûr.
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
