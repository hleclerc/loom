#pragma once

#ifdef __CUDACC__

#include "ExecutionContext_Cuda.h"
#include "ExecutionContext_Cpu.h"
#include "CpuHostMemorySpace.h"
#include "../Ct.h"
#include "Ptr.h"

#include <SYCL/sycl.hpp>

namespace sdot {

/// device global memory
struct MemorySpace_GlobalCudaRam {
    // auto HD execution_space() const { return ExecutionContext_Cuda{}; }

    /// allocate `n` items of T on the device, expose them (as an informed Ptr) to `func`, then release.
    /// cudaMalloc/cudaFree are host APIs -> __host__ only.
    // T_T __host__ void with_reservation( PI n, auto &&func ) const {
    //     T *p = nullptr;
    //     cudaMalloc( reinterpret_cast<void **>( &p ), n * sizeof( T ) );
    //     func( Ptr<T,MemorySpace_GlobalCudaRam>( p, *this ) );
    //     cudaFree( p );
    // }
    void         display( auto &os ) const { os << "GlobalCudaRam"; }

    sycl::memory_space ...;
};

constexpr auto transfer_cost_per_byte( ExecutionContext_Cuda, MemorySpace_GlobalCudaRam ) { return 0_c; }
constexpr auto transfer_cost_per_byte( ExecutionContext_Cpu , MemorySpace_GlobalCudaRam ) { return 1_c; }
constexpr auto transfer_cost_per_byte( ExecutionContext_Cuda, CpuHostMemorySpace        ) { return 1_c; }


constexpr auto operator==( MemorySpace_GlobalCudaRam, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_GlobalCudaRam, auto                      ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                     , MemorySpace_GlobalCudaRam ) { return Ct<bool,false>(); }


// Transfer primitive copy( dst_ptr, src_ptr, nb_items[, es] ): the (dst,src) memory-space types
// select the direction; the execution space carries the stream.
//
// Inter-space transfers go through cudaMemcpy, which is a host API -> __host__ only (calling them
// from device code is therefore a natural compile error).
T_T HD void copy( Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,CpuHostMemorySpace> src, PI nb_items, ExecutionContext_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        __trap();
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyHostToDevice, es.stream );
    #endif
}
T_T HD void copy( Ptr<T,CpuHostMemorySpace> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items, ExecutionContext_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        __trap();
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToHost, es.stream );
        cudaStreamSynchronize( es.stream );
    #endif
}

// host<->device transfers may be initiated from a non-CUDA context too — e.g. when the host pulls
// device results back to read/display them, the run that produced them is already finished and the
// "current" context is the CPU. The copy direction stays fixed by the Ptr memory spaces; only the
// driving stream is unspecified, so fall back to the process default stream.
T_T HD void copy( Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,CpuHostMemorySpace> src, PI nb_items, ExecutionContext_Cpu ) {
    copy( dst, src, nb_items, ExecutionContext_Cuda{} );
}
T_T HD void copy( Ptr<T,CpuHostMemorySpace> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items, ExecutionContext_Cpu ) {
    copy( dst, src, nb_items, ExecutionContext_Cuda{} );
}

// device -> device: feasible from both sides -> HD, and this is where __CUDA_ARCH__ legitimately
// picks an *implementation* (both branches are valid): in a kernel, an internal element copy; from
// the host, cudaMemcpy. (TODO: peer copy for different devices.)
T_T HD void copy( Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items, ExecutionContext_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        for ( PI i = 0; i < nb_items; ++i )
            dst.raw[ i ] = src.raw[ i ];
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToDevice, es.stream );
    #endif
}

namespace detail {
    T_T struct ZeroOnGlobalCudaRam {
        ZeroOnGlobalCudaRam() {
            cudaMalloc( reinterpret_cast<void **>( &p ), sizeof( T ) );
            T zero = 0;
            cudaMemcpyAsync( p, &zero, sizeof( T ), cudaMemcpyHostToDevice, ExecutionContext_Cuda::default_stream );
        }

        T *p = nullptr;
    };
}

T_T T *zero_for( MemorySpace_GlobalCudaRam /* memory_space */ ) { static detail::ZeroOnGlobalCudaRam<T> res; return res.p; }

} // namespace sdot

#endif
