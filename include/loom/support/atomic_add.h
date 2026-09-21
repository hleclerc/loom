#pragma once

#include "common_macros.h" // HD_INLINE
#include <atomic>

namespace sdot {

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
    atomicOr( &target, value );
#else
    std::atomic_ref<T>( target ).fetch_or( value, std::memory_order_relaxed );
#endif
}

/// Same as `atomic_add`, but returns the value BEFORE the addition (a ticket / a slot reservation).
template<class T>
HD_INLINE T atomic_fetch_add( T &target, T value ) {
#ifdef __CUDA_ARCH__
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
