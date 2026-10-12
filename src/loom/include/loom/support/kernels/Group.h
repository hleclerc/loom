#pragma once

#include <loom/support/common_macros.h> // HD

#include <barrier>

namespace sdot {

/// The lane group of a cooperative kernel, seen from the CPU: `group_size` system threads around a
/// `std::barrier` (null when the group is reduced to one lane -- the production case on CPU, where
/// `group_barrier` then does nothing).
///
/// What the body can do with it is the SUBSET common to devices: `group_barrier`,
/// `get_local_linear_range`. The CUDA version (`CudaGroup`) will expose the same contract on
/// `__syncthreads`.
struct CpuGroup {
    HD int get_local_linear_range() const { return size; }

    std::barrier<> *barrier; ///< null if `size == 1`
    int             size;
};

HD inline void group_barrier( const CpuGroup &group ) {
    if ( group.barrier )
        group.barrier->arrive_and_wait();
}

/// The subgroup (the warp) of a lane. On CPU each lane is its own subgroup: nothing runs in
/// lockstep there, so `get_local_linear_range() == 1` and the barrier is empty. A body that
/// cooperates per subgroup must degenerate correctly at that width (see
/// `OtPlan1d.cxx::sort_diracs`).
struct CpuSubGroup {
    HD int get_local_linear_id   () const { return 0; }
    HD int get_local_linear_range() const { return size; }
    HD int get_group_linear_id   () const { return lane; }

    int lane;
    int size;
};

HD inline void group_barrier( const CpuSubGroup & ) {}

} // namespace sdot
