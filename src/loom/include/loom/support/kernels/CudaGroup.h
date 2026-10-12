#pragma once

#include "../common_macros.h"

namespace sdot {

/// The group of lanes of a cooperative kernel, as seen from CUDA: the block. Same contract as `CpuGroup`
/// (`group_barrier`, `get_local_linear_range`), on `__syncthreads`.
struct CudaGroup {
    HD int get_local_linear_range() const { return size; }
    int size;
};

HD_INLINE void group_barrier( const CudaGroup & ) {
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
}

/// The sub-group: the lane's warp. `get_local_linear_range` is the ACTUAL width of the warp
/// (32, or what is left in a block that is not a multiple of it), `get_group_linear_id` its
/// rank in the block -- what `OtPlan1d.cxx::sort_diracs` derives from `local_index`.
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
