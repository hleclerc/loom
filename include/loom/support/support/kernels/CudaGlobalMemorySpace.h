#pragma once

namespace sdot {

/// mémoire globale du GPU, telle que vue depuis l'hôte : une adresse device, qu'on ne peut pas
/// déréférencer ici. C'est la zone dans laquelle XLA nous remet DÉJÀ les buffers d'un call GPU,
/// donc celle qu'un `TensorView` généré porte dans son type quand le device est un GPU (rien à
/// transférer : voir `transfer_cost_per_byte` près de `CudaQueue`).
struct CudaGlobalMemorySpace {
    static constexpr bool directly_accessible = false; ///< adresse device : pas de déréf hôte
    static constexpr bool kernel_context      = false; ///< vue hôte, pas un tag kernel

    bool operator==( const CudaGlobalMemorySpace & ) const = default;
    void display    ( auto &os ) const { os << "CudaGlobalMemorySpace"; }
};

} // namespace sdot
