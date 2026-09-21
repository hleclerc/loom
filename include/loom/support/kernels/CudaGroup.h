#pragma once

#include "../common_macros.h"

namespace sdot {

/// Le groupe de voies d'un kernel coopératif, vu de CUDA : le bloc. Même contrat que `CpuGroup`
/// (`group_barrier`, `get_local_linear_range`), sur `__syncthreads`.
struct CudaGroup {
    HD int get_local_linear_range() const { return size; }
    int size;
};

HD_INLINE void group_barrier( const CudaGroup & ) {
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
}

/// Le sous-groupe : le warp de la voie. `get_local_linear_range` est la largeur RÉELLE du warp
/// (32, ou ce qu'il reste dans un bloc qui n'en est pas un multiple), `get_group_linear_id` son
/// rang dans le bloc -- ce qu'`OtPlan1d.cxx::sort_diracs` dérive de `local_index`.
struct CudaSubGroup {
    HD int get_local_linear_id   () const { return lane; }
    HD int get_local_linear_range() const { return size; }
    HD int get_group_linear_id   () const { return warp; }
    int lane, size, warp;
};

HD_INLINE void group_barrier( const CudaSubGroup & ) {
#ifdef __CUDA_ARCH__
    __syncwarp();
#endif
}

} // namespace sdot
