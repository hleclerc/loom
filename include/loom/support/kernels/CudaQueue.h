#pragma once

#include "CudaGlobalMemorySpace.h"
#include "CudaKernelMemorySpace.h"
#include "../Ct.h"

// LE CONTEXTE D'EXÉCUTION CUDA -- étape 3 de la refonte (CPU d'abord, en C++ nu ; puis le
// découpage en couches ; puis CUDA via nvcc). Le contrat est celui de `CpuQueue.h` : une
// `DefaultKernelMemorySpace`, `transfer_cost_per_byte` pour les zones atteignables, et les deux
// lancements `submit_kernel` / `submit_kernel_grouped` trouvés par ADL depuis `run_parallel` --
// ici un `__global__` sur un flux, `CudaGroup` sur `__syncthreads`, `__shared__` pour
// `local_scratch`. Rien de tout cela n'est écrit encore : inclure cet en-tête est une erreur
// nommée plutôt qu'un échec obscur plus loin.

#error "CudaQueue : le backend CUDA n'est pas encore porté (étape 3 de la refonte, voir CpuQueue.h pour le contrat)"

namespace sdot {

struct CudaQueue {
    using DefaultKernelMemorySpace = CudaKernelMemorySpace;
};

constexpr auto transfer_cost_per_byte( const CudaQueue &, CudaGlobalMemorySpace ) { return Ct<double,0.0>(); }

} // namespace sdot
