#pragma once

#include "../common_macros.h" // HD
#include "../common_types.h" // SI

namespace sdot {

/// WHAT THE KERNEL CAN KNOW ABOUT THE MACHINE, in a form that mentions no particular machine.
///
/// This is what was missing to write a portable kernel: the launch geometry was either hard-coded
/// ( `const int block = 128` in `CudaQueue.h` ), or guessed in Python, or copied from
/// tutorial to tutorial. A kernel that wants to choose its group size or its shared memory
/// budget must be able to ASK for it.
///
/// The four fields are deliberately few, and each has a meaning on CPU as on GPU:
/// a kernel written once reads the same names everywhere. They are HINTS and budgets, not
/// laws -- `run_parallel` reads none of them by itself, it is the kernel that decides.
///
/// Obtained through `queue.machine()`. The driver queries are made ONCE and kept.
struct Machine {
    /// how many work-items can progress at the same time.
    /// CPU: the size of the thread pool. CUDA: SMs x threads per SM.
    /// What it is for: sizing a PER-THREAD scratch on the concurrent workers and not on
    /// the items, so a big batch does not blow up memory.
    SI  nb_workers;

    /// how many lanes advance in lockstep ( a warp ).
    /// CPU: 1, there are no lanes. CUDA: `warpSize`, 32 in practice.
    /// What it is for: a cooperative algorithm ( scan, histogram ) splits itself on it.
    SI  sub_group_width;

    /// shared memory budget per group, in bytes -- what `local_mem_elems` must respect.
    /// CUDA: the maximum per block, read from the driver. CPU: a NOTIONAL value ( there is no
    /// hardware shared memory, `CpuQueue` takes a `std::vector` on the heap ), chosen so
    /// that a portable kernel sizes something sensible rather than dividing by zero.
    SI  local_mem_bytes;

    /// a group size that works, when the kernel has no reason to prefer another.
    /// CPU: 1. CUDA: the warp width.
    SI  suggested_group;

    /// a trivially copyable POD: its kernel form is itself, so it goes all the way into a
    /// kernel ( see `make_avaiable.h` ). Useful for a body that adapts its splitting on the spot.
    HD Machine kernel_form( auto &&, auto ) const { return *this; }
};

/// the notional value of the shared memory budget on a machine that has none.
static constexpr SI cpu_notional_local_mem_bytes = 64 * 1024;

} // namespace sdot
