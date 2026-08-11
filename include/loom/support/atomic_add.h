#pragma once

#include <SYCL/sycl.hpp>

namespace sdot {

/// Atomic `target += value`, for scattering a value that MANY work-items contribute to the same slot
/// of (e.g. a ProjectedSumOfDiracs backward: every angle adds d cost / d position onto the SAME
/// shared 2D-point gradient). `relaxed` order + `device` scope is all we need -- correctness of the
/// sum, not any ordering. `generic_space` so it works whatever memory the buffer lives in.
template<class T>
void atomic_add( T &target, T value ) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space> ref( target );
    ref.fetch_add( value );
}

/// Atomic `target |= value`, same reasoning as `atomic_add` -- used to build a per-BUCKET "which
/// lanes of my sub-group share this digit" mask (one bit per lane) without a CUDA-only warp-match
/// intrinsic: every lane ORs its own bit into its bucket's mask cell, safe/commutative regardless of
/// interleaving, see `OtPlan1d.cxx::sort_diracs`'s scatter phase.
template<class T>
void atomic_or( T &target, T value ) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space> ref( target );
    ref.fetch_or( value );
}

/// Same as `atomic_add`/`atomic_or`, but `work_group` SCOPE + `local_space`: for a target that only
/// ever lives in ONE work-group's `local_scratch` and is never touched cross-device (e.g.
/// `OtPlan1d.cxx::sort_diracs`'s per-sub-group histogram/match-mask rows) -- a much cheaper fence
/// than `device` scope, which is a correct but needlessly heavy superset for local-memory-only data.
template<class T>
void atomic_add_local( T &target, T value ) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space> ref( target );
    ref.fetch_add( value );
}
template<class T>
void atomic_or_local( T &target, T value ) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space> ref( target );
    ref.fetch_or( value );
}

}
