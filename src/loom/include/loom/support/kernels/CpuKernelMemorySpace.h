#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

/// Memory seen from inside a CPU kernel: no runtime attribute, trivially copyable.
/// `kernel_form` retypes the source `MemorySpace` of an argument to this tag when the chosen
/// execution context is a `CpuQueue`. In the kernel, a `Ptr<T, CpuKernelMemorySpace>` is
/// dereferenced directly: the data is already local.
struct CpuKernelMemorySpace {
    static constexpr bool kernel_context      = true; ///< this Ptr lives in a kernel
    static constexpr bool directly_accessible = true; ///< data local to the kernel -> direct deref

       bool operator==( const CpuKernelMemorySpace & ) const = default;
    HD void display ( auto &os ) const { os << "CpuKernelMemorySpace"; }
};

} // namespace sdot
