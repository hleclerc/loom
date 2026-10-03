#pragma once

#include "CpuHostMemorySpace.h"
#include "../common_macros.h"
#include "../common_types.h"
#include <type_traits>

namespace sdot {

// ---------------------------------------------------------------------------
// "Informed" pointer: a raw address paired with its memory zone (`MemorySpace`).
//
// Kernel side, the `MemorySpace` is an empty tag (`*KernelMemorySpace`) -> same size
// as a bare pointer. Host side it may carry runtime state (queue/device, GPU
// number, affinity...); this context is what `kernel_form` strips.
//
// Dereferencing is chosen in constexpr from `MemorySpace::directly_accessible`:
//   - directly_accessible (local kernel zone, or CpuRam host side) -> direct deref
//   - otherwise (e.g. GlobalCudaRam host side) -> only `value()` works, via a one-element transfer
// `MemorySpace::kernel_context` additionally says whether this `Ptr` lives in a kernel.
// ---------------------------------------------------------------------------
template<class T, class _MemorySpace>
struct Ptr {
    using            difference_type = SI;
    using            MemorySpace     = _MemorySpace;
    using            value_type      = T;

    HD explicit      Ptr             ( T *raw = nullptr, MemorySpace memory_space = {} ) : memory_space( memory_space ), raw( raw ) {}

    // byte arithmetic (T is typically std::byte / const std::byte for strided views)
    HD auto          operator+       ( auto off ) const { return Ptr( raw + off, memory_space ); }
    HD auto          operator-       ( auto off ) const { return Ptr( raw - off, memory_space ); }

    HD void          operator++      () { ++raw; }

    // reinterpretation as another element type, keeping the memory zone
    T_U HD U*        as              () const { return reinterpret_cast<U *>( raw ); }

    HD explicit      operator bool   () const { return raw != nullptr; }
    HD bool          operator==      ( const Ptr &o ) const { return memory_space == o.memory_space && raw == o.raw; }
    HD bool          operator!=      ( const Ptr &o ) const { return ! operator==( o ); }

    HD T&            operator*       () const { static_assert( MemorySpace::directly_accessible, "operator* on a zone that is not directly accessible: use value() (transfer) or kernel_form" ); return *raw; }

    HD T             value           () const {
        if constexpr ( MemorySpace::directly_accessible )
            return *raw;
        else { // host zone not accessible -> bring one element back onto the stack
            static_assert( MemorySpace::kernel_context == false );
            T res;
            copy( Ptr<T, CpuHostMemorySpace>( &res ), *this, 1 );
            return res;
        }
    }

    HD void          set             ( auto &&value ) const {
        if constexpr ( MemorySpace::directly_accessible ) {
            *raw = value;
        } else {
            static_assert( MemorySpace::kernel_context == false );
            if constexpr ( std::is_same_v<DECAYED_TYPE_OF( value ),T> ) { // host zone not accessible -> bring one element back onto the stack
                copy( *this, Ptr<const T,CpuHostMemorySpace>( &value ), 1 );
            } else {
                T tmp = value;
                copy( *this, Ptr<const T,CpuHostMemorySpace>( &tmp ), 1 );
            }
        }
    }

    MemorySpace      memory_space; ///< empty structure for kernel zones
    T*               raw;
};

} // namespace sdot
