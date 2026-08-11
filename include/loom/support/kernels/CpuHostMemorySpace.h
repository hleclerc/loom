#pragma once

namespace sdot {

/// mémoire hôte paginée (RAM CPU standard), telle que décrite/manipulée depuis l'hôte
struct CpuHostMemorySpace {
    static constexpr bool directly_accessible = true;  ///< déréférençable directement depuis l'hôte
    static constexpr bool kernel_context      = false; ///< zone hôte, pas un tag kernel

    bool operator==( const CpuHostMemorySpace & ) const = default;
    void display    ( auto &os ) const { os << "CpuHostMemorySpace"; }
};

} // namespace sdot
