#pragma once

// #include "hipSYCL/sycl/queue.hpp"
#include "CpuKernelMemorySpace.h"
#include "CpuHostMemorySpace.h"
#include "../Ct.h"
#include <SYCL/sycl.hpp>

namespace sdot {

struct CpuQueue {
    /// zone mémoire par défaut vue par les kernels lancés sur cette queue (un contexte
    /// d'exécution peut exposer plusieurs zones ; celle-ci est celle utilisée par défaut)
    using DefaultKernelMemorySpace = CpuKernelMemorySpace;

    sycl::queue queue{ sycl::cpu_selector_v };
};

/// Coût de transfert (secondes par octet) pour rendre une zone source accessible depuis ce
/// contexte d'exécution. Un contexte connaît les zones qu'il peut atteindre (l'inverse non),
/// donc ces surcharges vivent près du contexte.
///   CPU -> CPU : la donnée est déjà en RAM hôte, rien à transférer.
constexpr auto transfer_cost_per_byte( const CpuQueue &, CpuHostMemorySpace ) { return Ct<double,0.0>(); }

}
