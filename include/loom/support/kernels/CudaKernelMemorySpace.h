#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

/// Memory seen from inside a CUDA kernel: no runtime attribute, trivially copyable.
/// `kernel_form` retypes an argument's source `MemorySpace` to this tag when the chosen
/// execution context targets the GPU. In the kernel, a `Ptr<T, CudaKernelMemorySpace>` is
/// dereferenced directly: the data is already local.
struct CudaKernelMemorySpace {
    static constexpr bool kernel_context      = true; ///< this Ptr lives in a kernel
    static constexpr bool directly_accessible = true; ///< data local to the kernel (device memory) -> direct deref

       bool operator==( const CudaKernelMemorySpace & ) const = default;
    HD void display ( auto &os ) const { os << "CudaKernelMemorySpace"; }
};

} // namespace sdot
