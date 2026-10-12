#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

/// global memory of the GPU, as seen from the host: a device address, which cannot be
/// dereferenced here. It is the space in which XLA ALREADY hands us the buffers of a GPU call,
/// hence the one a generated `TensorView` carries in its type when the device is a GPU (nothing to
/// transfer: see `transfer_cost_per_byte` near `CudaQueue`).
struct CudaGlobalMemorySpace {
    static constexpr bool directly_accessible = false; ///< device address: no host dereference
    static constexpr bool kernel_context      = false; ///< host view, not a kernel tag

       bool operator==( const CudaGlobalMemorySpace & ) const = default;
    HD void display ( auto &os ) const { os << "CudaGlobalMemorySpace"; }
};

} // namespace sdot
