#pragma once

#include "common_macros.h" // HD_INLINE
#include <type_traits>
#include <atomic>

namespace sdot {

namespace detail {

/// CUDA N'A PAS D'ATOMIQUE POUR UN ENTIER SIGNE DE 64 BITS : ses surcharges couvrent `int`,
/// `unsigned int`, `unsigned long long`, `float` et `double`, et rien entre les deux. Or `SI` est
/// un `int64_t`, donc tout compteur d'items en 64 bits tombait dessus -- `no instance of
/// overloaded function "atomicAdd" matches the argument list`, sur la carte seulement.
///
/// En COMPLEMENT A DEUX, l'addition et le OU bit a bit sont les MEMES operations signees ou non :
/// on passe par la surcharge non signee, et le resultat est exact ( ce n'est pas une troncature
/// ni une approximation, c'est la meme suite de bits ). C'est l'idiome usuel du code CUDA.
template<class T>
constexpr bool est_signe_64 = std::is_integral_v<T> && std::is_signed_v<T> && sizeof( T ) == 8;

using U64 = unsigned long long;

}

/// Atomic `target += value`, for scattering a value that MANY work-items contribute to the same slot
/// of (e.g. a ProjectedSumOfDiracs backward: every angle adds d cost / d position onto the SAME
/// shared 2D-point gradient). Relaxed order is all we need -- correctness of the sum, not any
/// ordering.
///
/// Le shim par device : `std::atomic_ref` sur CPU (C++20, `fetch_add` défini pour les flottants
/// aussi), `atomicAdd` dans du code device CUDA. C'est l'UN des deux endroits où un `#if` sur la
/// cible est légitime (l'autre est `math.h`) : une intrinsèque n'a pas d'autre forme.
template<class T>
HD_INLINE void atomic_add( T &target, T value ) {
#ifdef __CUDA_ARCH__
    if constexpr ( detail::est_signe_64<T> )
        atomicAdd( reinterpret_cast<detail::U64 *>( &target ), detail::U64( value ) );
    else
        atomicAdd( &target, value );
#else
    std::atomic_ref<T>( target ).fetch_add( value, std::memory_order_relaxed );
#endif
}

/// Atomic `target |= value`, same reasoning as `atomic_add` -- used to build a per-BUCKET "which
/// lanes of my sub-group share this digit" mask (one bit per lane) without a CUDA-only warp-match
/// intrinsic: every lane ORs its own bit into its bucket's mask cell, safe/commutative regardless of
/// interleaving, see `OtPlan1d.cxx::sort_diracs`'s scatter phase.
template<class T>
HD_INLINE void atomic_or( T &target, T value ) {
#ifdef __CUDA_ARCH__
    if constexpr ( detail::est_signe_64<T> )
        atomicOr( reinterpret_cast<detail::U64 *>( &target ), detail::U64( value ) );
    else
        atomicOr( &target, value );
#else
    std::atomic_ref<T>( target ).fetch_or( value, std::memory_order_relaxed );
#endif
}

/// Same as `atomic_add`, but returns the value BEFORE the addition (a ticket / a slot reservation).
template<class T>
HD_INLINE T atomic_fetch_add( T &target, T value ) {
#ifdef __CUDA_ARCH__
    if constexpr ( detail::est_signe_64<T> )
        return T( atomicAdd( reinterpret_cast<detail::U64 *>( &target ), detail::U64( value ) ) );
    else
        return atomicAdd( &target, value );
#else
    return std::atomic_ref<T>( target ).fetch_add( value, std::memory_order_relaxed );
#endif
}

/// Same as `atomic_add`/`atomic_or`, but for a target that only ever lives in ONE work-group's
/// `local_scratch` and is never touched cross-device (e.g. `OtPlan1d.cxx::sort_diracs`'s
/// per-sub-group histogram/match-mask rows) -- on a GPU a much cheaper fence than device scope.
template<class T>
HD_INLINE void atomic_add_local( T &target, T value ) { atomic_add( target, value ); }
template<class T>
HD_INLINE void atomic_or_local( T &target, T value ) { atomic_or( target, value ); }

}
