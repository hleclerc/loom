#pragma once

#ifdef __CUDACC__

#include "ExecutionContext_Cuda.h"
#include "ExecutionContext_Cpu.h"
#include "Ptr.h"
#include "../Ct.h"

namespace sdot {

/// page-locked host memory: zero-copy reachable from both host and device
struct MemorySpace_PinnedCpuRam {
    /// allocate `n` page-locked items of T, expose them (as an informed Ptr) to `func`, then release.
    /// cudaMallocHost/cudaFreeHost are host APIs -> __host__ only.
    T_T __host__ void with_reservation( PI n, auto &&func ) const {
        T *p = nullptr;
        cudaMallocHost( reinterpret_cast<void **>( &p ), n * sizeof( T ) );
        func( Ptr<T,MemorySpace_PinnedCpuRam>( p, *this ) );
        cudaFreeHost( p );
    }
    void         display( auto &os ) const { os << "PinnedCpuRam"; }
};

constexpr auto transfer_cost_per_byte( ExecutionContext_Cuda, MemorySpace_PinnedCpuRam ) { return 0_c; }
constexpr auto transfer_cost_per_byte( ExecutionContext_Cpu , MemorySpace_PinnedCpuRam ) { return 0_c; }

constexpr auto operator==( MemorySpace_PinnedCpuRam, MemorySpace_PinnedCpuRam ) { return Ct<bool,true >(); }
constexpr auto operator==( MemorySpace_PinnedCpuRam, auto                     ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                    , MemorySpace_PinnedCpuRam ) { return Ct<bool,false>(); }

} // namespace sdot

#endif
