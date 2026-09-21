#pragma once

#include "CpuHostMemorySpace.h"
#include "../common_macros.h"
#include "../common_types.h"
#include <type_traits>

namespace sdot {

// ---------------------------------------------------------------------------
// Pointeur "informé" : une adresse brute associée à sa zone mémoire (`MemorySpace`).
//
// Côté kernel le `MemorySpace` est un tag vide (`*KernelMemorySpace`) -> même taille
// qu'un pointeur nu. Côté hôte il peut porter du runtime (queue/device SYCL, numéro de
// GPU, affinité...) ; c'est ce contexte qui est retiré par `kernel_form`.
//
// Le déréférencement se choisit en constexpr depuis `MemorySpace::directly_accessible` :
//   - directly_accessible (zone kernel locale, ou CpuRam côté hôte) -> déréf direct
//   - sinon (ex. GlobalCudaRam côté hôte) -> seul `value()` marche, via un transfert d'un élément
// `MemorySpace::kernel_context` indique en plus si ce `Ptr` vit dans un kernel.
// ---------------------------------------------------------------------------
template<class T, class _MemorySpace>
struct Ptr {
    using            difference_type = SI;
    using            MemorySpace     = _MemorySpace;
    using            value_type      = T;

    HD explicit      Ptr             ( T *raw = nullptr, MemorySpace memory_space = {} ) : memory_space( memory_space ), raw( raw ) {}

    // arithmétique en octets (T vaut typiquement std::byte / const std::byte pour les vues stridées)
    HD auto          operator+       ( auto off ) const { return Ptr( raw + off, memory_space ); }
    HD auto          operator-       ( auto off ) const { return Ptr( raw - off, memory_space ); }

    HD void          operator++      () { ++raw; }

    // réinterprétation vers un autre type d'élément, en gardant la zone mémoire
    T_U HD U*        as              () const { return reinterpret_cast<U *>( raw ); }

    HD explicit      operator bool   () const { return raw != nullptr; }
    HD bool          operator==      ( const Ptr &o ) const { return memory_space == o.memory_space && raw == o.raw; }
    HD bool          operator!=      ( const Ptr &o ) const { return ! operator==( o ); }

    HD T&            operator*       () const { static_assert( MemorySpace::directly_accessible, "operator* sur une zone non accessible directement : utiliser value() (transfert) ou kernel_form" ); return *raw; }

    HD T             value           () const {
        if constexpr ( MemorySpace::directly_accessible )
            return *raw;
        else { // zone hôte non accessible -> on rapatrie un élément sur la pile
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
            if constexpr ( std::is_same_v<DECAYED_TYPE_OF( value ),T> ) { // zone hôte non accessible -> on rapatrie un élément sur la pile
                copy( *this, Ptr<const T,CpuHostMemorySpace>( &value ), 1 );
            } else {
                T tmp = value;
                copy( *this, Ptr<const T,CpuHostMemorySpace>( &tmp ), 1 );
            }
        }
    }

    MemorySpace      memory_space; ///< structure vide pour les zones kernel
    T*               raw;
};

} // namespace sdot
