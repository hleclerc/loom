#pragma once

namespace sdot {

/// Mémoire vue depuis l'intérieur d'un kernel CUDA : aucun attribut runtime, trivialement copiable.
/// `make_available` retype le `MemorySpace` source d'un argument vers ce tag quand le contexte
/// d'exécution choisi cible le GPU. Dans le kernel, un `Ptr<T, CudaKernelMemorySpace>` se
/// déréférence directement : la donnée est déjà locale.
struct CudaKernelMemorySpace {
    static constexpr bool kernel_context      = true; ///< ce Ptr vit dans un kernel
    static constexpr bool directly_accessible = true; ///< donnée locale au kernel (mémoire device) -> déréf direct

    bool operator==( const CudaKernelMemorySpace & ) const = default;
    void display    ( auto &os ) const { os << "CudaKernelMemorySpace"; }
};

} // namespace sdot
