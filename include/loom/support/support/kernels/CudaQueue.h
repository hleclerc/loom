#pragma once

#include "CudaGlobalMemorySpace.h"
#include "CudaKernelMemorySpace.h"
#include "../Ct.h"
#include <SYCL/sycl.hpp>

namespace sdot {

struct CudaQueue {
    /// zone mémoire par défaut vue par les kernels lancés sur cette queue
    using DefaultKernelMemorySpace = CudaKernelMemorySpace;

    sycl::queue queue{ sycl::gpu_selector_v };
};

/// Coût de transfert (secondes par octet) pour rendre une zone source accessible depuis ce
/// contexte d'exécution.
///   GPU -> GPU : la donnée est déjà en mémoire globale du device (c'est là que XLA nous remet
///   les buffers d'un call GPU), rien à transférer -- `make_available` ne fait que retyper le Ptr.
constexpr auto transfer_cost_per_byte( const CudaQueue &, CudaGlobalMemorySpace ) { return Ct<double,0.0>(); }

} // namespace sdot
