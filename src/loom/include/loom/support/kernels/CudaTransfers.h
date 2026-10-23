#pragma once

#include "CudaQueue.h"
#include "../containers/TensorView.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace sdot {

/// The INPUTS of a call on the card that sit in host memory ( a Torch tensor left on the CPU, a member of an aggregate
/// as well as a root argument ): the handler copies each one to the card before anything runs ( `in` ), and the kernels
/// read the copy -- `run_parallel` as well as a hand-written launch, which both take the views the handler built.
///
/// The copy is the span the view covers, strides included ( a transposed or sliced view keeps its strides, read on the
/// copy ), on the call's stream: ordered before the launches, and freed after them ( `cudaFreeAsync` at the end of the
/// handler ). Python decides which inputs come here ( `CallArg_Tensor._needs_transfer` ).
struct CudaTransfers {
    explicit CudaTransfers( const CudaQueue &queue ) : stream( queue.stream ) {}
    CudaTransfers( const CudaTransfers & ) = delete;
    CudaTransfers &operator=( const CudaTransfers & ) = delete;
    ~CudaTransfers() {
        for ( void *p : copies )
            cudaFreeAsync( p, stream );                  // ( no throw in a destructor: a failure here shows at the next call )
    }

    template<class TF,class Shape,class AxisNames,class Strides>
    auto in( const TensorView<TF,Shape,CpuHostMemorySpace,AxisNames,Strides> &view ) {
        using R = TensorView<TF,Shape,CudaGlobalMemorySpace,AxisNames,Strides>;
        using T = std::remove_const_t<TF>;

        // the bytes the view reaches, from its lowest element to the end of its highest ( a negative stride reads backwards )
        const Shape shape = view.shape();
        const Strides strides = view.strides();
        SI lo = 0, hi = SI( sizeof( T ) );
        bool empty = false;
        [&]<int... d>( std::integer_sequence<int,d...> ) {
            ( [&] {
                const SI n = SI( shape[ Ct<int,d>() ] ), s = SI( strides[ Ct<int,d>() ] );
                if ( n <= 0 ) empty = true;
                else if ( s >= 0 ) hi += ( n - 1 ) * s;
                else lo += ( n - 1 ) * s;
            }(), ... );
        }( std::make_integer_sequence<int,int( Shape::ct_size )>() );
        if ( empty )
            return R( typename R::DataPtr( nullptr, CudaGlobalMemorySpace{} ), shape, strides );

        std::byte *dev = nullptr;
        const auto *src = reinterpret_cast<const std::byte *>( view.data().raw ) + lo;
        cuda_check( cudaMallocAsync( ( void ** ) &dev, size_t( hi - lo ), stream ), "malloc ( host input -> card )" );
        cuda_check( cudaMemcpyAsync( dev, src, size_t( hi - lo ), cudaMemcpyHostToDevice, stream ), "memcpy ( host input -> card )" );
        copies.push_back( dev );
        return R( typename R::DataPtr( reinterpret_cast<TF *>( dev - lo ), CudaGlobalMemorySpace{} ), shape, strides );
    }

    cudaStream_t        stream;
    std::vector<void *> copies;
};

} // namespace sdot
