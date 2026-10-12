#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

/// paged host memory (standard CPU RAM), as described/handled from the host
struct CpuHostMemorySpace {
    static constexpr bool directly_accessible = true;  ///< directly dereferenceable from the host
    static constexpr bool kernel_context      = false; ///< host zone, not a kernel tag

       bool operator==( const CpuHostMemorySpace & ) const = default;
    HD void display ( auto &os ) const { os << "CpuHostMemorySpace"; }
};

} // namespace sdot
