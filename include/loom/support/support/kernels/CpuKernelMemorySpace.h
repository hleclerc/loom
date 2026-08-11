#pragma once

namespace sdot {

/// Mémoire vue depuis l'intérieur d'un kernel CPU : aucun attribut runtime, trivialement copiable.
/// `make_available` retype le `MemorySpace` source d'un argument vers ce tag quand le contexte
/// d'exécution choisi est un `CpuQueue`. Dans le kernel, un `Ptr<T, CpuKernelMemorySpace>` se
/// déréférence directement : la donnée est déjà locale.
struct CpuKernelMemorySpace {
    static constexpr bool kernel_context      = true; ///< ce Ptr vit dans un kernel
    static constexpr bool directly_accessible = true; ///< donnée locale au kernel -> déréf direct

    bool operator==( const CpuKernelMemorySpace & ) const = default;
    void display    ( auto &os ) const { os << "CpuKernelMemorySpace"; }
};

} // namespace sdot
