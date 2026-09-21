#pragma once

#include "CudaGlobalMemorySpace.h"
#include "CudaKernelMemorySpace.h"
#include "CpuHostMemorySpace.h"
#include "../common_macros.h"
#include "../common_types.h"
#include "QueueEvent.h"
#include "IoCategory.h"
#include "CudaGroup.h"
#include "Reducer.h"
#include "Ptr.h"
#include "../Ct.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <tuple>

namespace sdot {

/// Le contexte d'exécution CUDA : un FLUX, et les deux formes de lancement qu'un kernel peut
/// demander (`submit_kernel`, `submit_kernel_grouped`) -- le pendant de `CpuQueue.h`, même
/// contrat, trouvé par ADL depuis `run_parallel`, qui ne sait rien du device.
///
/// Le flux est celui que XLA nous donne pour l'appel (`ffi::PlatformStream`, voir `JaxFfi`) : ce
/// que XLA a lancé avant nous sur ce flux est fini quand nous commençons, et ce que nous y lançons
/// est fini quand XLA lit nos sorties -- l'ordre est celui du flux, sans synchronisation. Un
/// `QueueEvent` attend quand même le flux à sa destruction (« synchrone par défaut », voir
/// `QueueEvent.h`) : c'est ce qui rend les réductions et les relectures hôte correctes ; un appelant
/// qui enchaîne des lancements et n'a rien à relire peut `detach()`.
///
/// Sur GPU la répartition est en FOULÉES (`index += nb_threads`), pas en tranches : des fils
/// voisins lisent des items voisins, c'est ce qui coalesce les accès.
struct CudaQueue {
    using DefaultKernelMemorySpace = CudaKernelMemorySpace;

    explicit CudaQueue( cudaStream_t stream ) : stream( stream ) {}
    CudaQueue() : stream( 0 ) {}

    cudaStream_t stream;
};

constexpr auto transfer_cost_per_byte( const CudaQueue &, CudaGlobalMemorySpace ) { return Ct<double,0.0>(); }

inline void cuda_check( cudaError_t err, const char *what ) {
    if ( err != cudaSuccess )
        throw std::runtime_error( std::string( "CUDA: " ) + what + ": " + cudaGetErrorString( err ) );
}

// ── relecture / écriture hôte d'un élément en mémoire globale (`Ptr::value` / `Ptr::set`) ──
template<class T>
void copy( Ptr<T,CpuHostMemorySpace> dst, Ptr<const T,CudaGlobalMemorySpace> src, SI n ) {
    cuda_check( cudaMemcpy( dst.raw, src.raw, sizeof( T ) * n, cudaMemcpyDeviceToHost ), "memcpy device -> host" );
}
template<class T>
void copy( Ptr<T,CpuHostMemorySpace> dst, Ptr<T,CudaGlobalMemorySpace> src, SI n ) {
    cuda_check( cudaMemcpy( dst.raw, src.raw, sizeof( T ) * n, cudaMemcpyDeviceToHost ), "memcpy device -> host" );
}
template<class T>
void copy( Ptr<T,CudaGlobalMemorySpace> dst, Ptr<const T,CpuHostMemorySpace> src, SI n ) {
    cuda_check( cudaMemcpy( dst.raw, src.raw, sizeof( T ) * n, cudaMemcpyHostToDevice ), "memcpy host -> device" );
}

// ── réductions : un accumulateur en registres par fil, combiné atomiquement à la fin ──────────
namespace detail::CudaQueueLaunch {
    /// `target = op( target, value )` atomique, par CAS sur la représentation 32 ou 64 bits --
    /// générique sur l'opérateur, ce que `atomicAdd` seul ne donne pas (`maximum` sur un double).
    template<class Op,class T>
    __device__ void atomic_combine( T *target, T value, const Op &op ) {
        if constexpr ( sizeof( T ) == 4 ) {
            unsigned *addr = reinterpret_cast<unsigned *>( target );
            unsigned old = *addr, assumed;
            do {
                assumed = old;
                T cur; memcpy( &cur, &assumed, 4 );
                T nxt = op( cur, value );
                unsigned bits; memcpy( &bits, &nxt, 4 );
                old = atomicCAS( addr, assumed, bits );
            } while ( old != assumed );
        } else {
            static_assert( sizeof( T ) == 8, "atomic_combine : 32 ou 64 bits" );
            unsigned long long *addr = reinterpret_cast<unsigned long long *>( target );
            unsigned long long old = *addr, assumed;
            do {
                assumed = old;
                T cur; memcpy( &cur, &assumed, 8 );
                T nxt = op( cur, value );
                unsigned long long bits; memcpy( &bits, &nxt, 8 );
                old = atomicCAS( addr, assumed, bits );
            } while ( old != assumed );
        }
    }

    template<class Op,class T>
    struct CudaReducer {
        HD void         combine   ( T v ) { value = op( value, v ); }
        HD CudaReducer &operator+=( T v ) { combine( v ); return *this; }
        __device__ void flush     () { atomic_combine( target, value, op ); }

        Op op;
        T  value;
        T *target;
    };

    /// une cible de réduction côté device : le scalaire qui reçoit les contributions, et sa
    /// relecture / libération une fois le kernel fini
    template<class Op,class T>
    struct DeviceTarget {
        Op  op;
        T  *host;
        T  *dev;
    };

    template<class Op,class T>
    DeviceTarget<Op,T> device_target_for( const ReductionTarget<Op,T> &target, cudaStream_t stream ) {
        T *dev;
        cuda_check( cudaMallocAsync( ( void ** ) &dev, sizeof( T ), stream ), "malloc (réduction)" );
        const T identity = Op::identity();
        cuda_check( cudaMemcpyAsync( dev, &identity, sizeof( T ), cudaMemcpyHostToDevice, stream ), "memcpy (identité)" );
        cuda_check( cudaStreamSynchronize( stream ), "sync (identité)" ); // `identity` est sur la pile
        return { target.op, target.host, dev };
    }

    template<class Op,class T>
    __device__ CudaReducer<Op,T> reducer_for( const DeviceTarget<Op,T> &t ) {
        return { t.op, Op::identity(), t.dev };
    }

    template<class Func,class Item,class... Args>
    __device__ void call( Func &func, Item item, int thread_index, int nb_threads, Args &...args ) {
        if constexpr ( requires { func( item, thread_index, nb_threads, args... ); } )
            func( item, thread_index, nb_threads, args... );
        else
            func( item, args... );
    }

    template<class Func,class ItemList,class Targets,class... Args>
    __global__ void flat_kernel( Func func, ItemList item_list, int nb_items, int nb_threads, Targets targets, Args... args ) {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if ( t >= nb_threads )
            return;
        std::apply( [&]( auto &...tgts ) {
            auto reducers = std::make_tuple( reducer_for( tgts )... );
            std::apply( [&]( auto &...reds ) {
                for ( int index = t; index < nb_items; index += nb_threads )
                    call( func, item_list[ index ], t, nb_threads, reds..., args... );
                ( reds.flush(), ... );
            }, reducers );
        }, targets );
    }

    template<class Func,class ItemList,class... Args>
    __global__ void grouped_kernel( Func func, ItemList item_list, int nb_items, int nb_groups, Args... args ) {
        extern __shared__ std::int32_t local_scratch[];
        const int g          = blockIdx.x;
        const int local_size = blockDim.x;
        const int lane       = threadIdx.x;
        CudaGroup    group{ local_size };
        CudaSubGroup sub_group{ lane % 32, min( 32, local_size - ( lane / 32 ) * 32 ), lane / 32 };
        for ( int index = g; index < nb_items; index += nb_groups )
            func( item_list[ index ], g, lane, local_size, group, local_scratch, sub_group, args... );
    }
}

/// Lancement plat : `nb_threads` fils, en foulées sur les items ; les réductions par fil en
/// registres, combinées atomiquement dans un scalaire device relu par le finalizer de l'événement.
template<class Deps,class Func,class ItemList,class Targets,class... Args>
auto submit_kernel( const CudaQueue &queue, const Deps &deps, Func &&func, ItemList &&item_list,
                    int nb_items, int nb_threads, Targets reduction_targets, Args &&...args ) {
    using namespace detail::CudaQueueLaunch;
    deps.wait_all();
    cudaStream_t stream = queue.stream;

    auto targets = std::apply( [&]( auto &...t ) { return std::make_tuple( device_target_for( t, stream )... ); }, reduction_targets );

    const int block = 128, grid = ( nb_threads + block - 1 ) / block;
    flat_kernel<<<grid, block, 0, stream>>>( func, item_list, nb_items, nb_threads, targets, args... );
    cuda_check( cudaGetLastError(), "lancement" );

    QueueEvent ev( [stream]{ cuda_check( cudaStreamSynchronize( stream ), "sync" ); } );
    std::apply( [&]( auto &...t ) {
        ( ev.finalizers.push_back( [t]{
            cuda_check( cudaMemcpy( t.host, t.dev, sizeof( *t.host ), cudaMemcpyDeviceToHost ), "memcpy (réduction)" );
            cudaFree( t.dev );
        } ), ... );
    }, targets );
    return ev;
}

/// Lancement coopératif : un BLOC de `group_size` voies par item concurrent, `local_elems` mots
/// `int32` de mémoire partagée (`local_scratch`), `group_barrier` sur `__syncthreads`.
template<class Deps,class Func,class ItemList,class Targets,class... Args>
auto submit_kernel_grouped( const CudaQueue &queue, const Deps &deps, Func &&func, ItemList &&item_list,
                            int nb_items, int nb_groups, int group_size, int local_elems, Targets, Args &&...args ) {
    using namespace detail::CudaQueueLaunch;
    static_assert( std::tuple_size_v<Targets> == 0, "les réductions ne sont pas supportées dans un kernel de groupe" );
    deps.wait_all();
    cudaStream_t stream = queue.stream;
    grouped_kernel<<<nb_groups, group_size, sizeof( std::int32_t ) * std::max( local_elems, 1 ), stream>>>( func, item_list, nb_items, nb_groups, args... );
    cuda_check( cudaGetLastError(), "lancement (groupes)" );
    return QueueEvent( [stream]{ cuda_check( cudaStreamSynchronize( stream ), "sync (groupes)" ); } );
}

} // namespace sdot
