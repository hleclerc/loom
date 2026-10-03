#pragma once

#include "CudaGlobalMemorySpace.h"
#include "CudaKernelMemorySpace.h"
#include "Machine.h"
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
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace sdot {

// ── the free form, declared elsewhere ─────────────────────────────────────────────────────
/// declared by `run_parallel.h` ( which includes us ): the FREE form, which the method below
/// calls. Redeclared here so that the method can name it without depending on the order of
/// the includes.
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest );

/// The CUDA execution context: a STREAM, and the two launch forms that a kernel can
/// request (`submit_kernel`, `submit_kernel_grouped`) -- the counterpart of `CpuQueue.h`, same
/// contract, found by ADL from `run_parallel`, which knows nothing about the device.
///
/// The stream is the one XLA gives us for the call (`ffi::PlatformStream`, see `JaxFfi`): what
/// XLA launched before us on this stream is finished when we start, and what we launch on it
/// is finished when XLA reads our outputs -- the order is that of the stream, without synchronization. A
/// `QueueEvent` still waits for the stream on its destruction ("synchronous by default", see
/// `QueueEvent.h`): this is what makes reductions and host re-reads correct; a caller
/// that chains launches and has nothing to re-read can `detach()`.
///
/// On GPU the distribution is STRIDED (`index += nb_threads`), not in slices: neighbouring
/// threads read neighbouring items, which is what coalesces the accesses.
struct CudaQueue {
    using DefaultKernelMemorySpace = CudaKernelMemorySpace;

    explicit CudaQueue( cudaStream_t stream ) : stream( stream ) {}
    CudaQueue() : stream( 0 ) {}

    /// what a kernel can know about this card ( see `Machine.h` ). Queries the driver ONCE:
    /// the attributes do not change, and one call per kernel would be wasted time.
    Machine machine() const;


    /// LAUNCH: `queue.run_parallel( Functor(), domain, args )`.
    ///
    /// Three things, and not a list of ( io, value ) pairs: `args` goes through in ONE piece, its generated
    /// `kernel_form` carrying the io policy of each argument. The functor therefore receives
    /// `( item, args )`, where `args` is the KERNEL form -- same data, without the queue.
    ///
    /// The free form `sdot::run_parallel( queue, domain, func, io, value, ... )` remains for
    /// whoever wants another policy or a subset.
    auto run_parallel( auto &&func, auto &&items, auto &&args ) {
        return sdot::run_parallel( *this, FORWARD( items ), FORWARD( func ), MutList(), FORWARD( args ) );
    }

    /// the same, passing ONE more value to the functor -- typically `batch_axes`, which the body
    /// needs to isolate its own axes ( `coords.axes - batch_axes` ).
    auto run_parallel( auto &&func, auto &&items, auto &&args, auto &&extra ) {
        return sdot::run_parallel( *this, FORWARD( items ), FORWARD( func ),
                                   MutList(), FORWARD( args ), InpList(), FORWARD( extra ) );
    }

    cudaStream_t stream;
};

constexpr auto transfer_cost_per_byte( const CudaQueue &, CudaGlobalMemorySpace ) { return Ct<double,0.0>(); }

inline void cuda_check( cudaError_t err, const char *what ) {
    if ( err != cudaSuccess )
        throw std::runtime_error( std::string( "CUDA: " ) + what + ": " + cudaGetErrorString( err ) );
}

/// the attributes of the card, read ONCE. `nb_workers` is the number of threads that can be
/// resident at the same time ( SMs x threads per SM ): it is what a PER-THREAD scratch should
/// be sized on. `suggested_group` is the warp width, the natural split of a cooperative algorithm --
/// and not the `block = 128` of the flat launch, which is an implementation detail of this file.
inline const Machine &cuda_machine() {
    static const Machine m = [] {
        int dev = 0;
        cuda_check( cudaGetDevice( &dev ), "get device" );
        auto attr = [&]( cudaDeviceAttr a, const char *what ) {
            int v = 0;
            cuda_check( cudaDeviceGetAttribute( &v, a, dev ), what );
            return SI( v );
        };
        const SI warp = attr( cudaDevAttrWarpSize, "warp size" );
        return Machine{
            attr( cudaDevAttrMultiProcessorCount, "SM count" )
                * attr( cudaDevAttrMaxThreadsPerMultiProcessor, "threads per SM" ),
            warp,
            attr( cudaDevAttrMaxSharedMemoryPerBlock, "shared mem per block" ),
            warp,
        };
    }();
    return m;
}

inline Machine CudaQueue::machine() const { return cuda_machine(); }

// ── host re-read / write of an element in global memory (`Ptr::value` / `Ptr::set`) ──
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

// ── kernel-only timing ( `LOOM_KERNEL_TIMING=1` ) ──────────────────────────────────────────────
/// What a benchmark needs and the wall clock of a call cannot give: the time the CARD spent in our
/// kernels, without the host side of the call ( XLA dispatch, the handler, the allocations, the
/// copies ). Off by default: the switch is read ONCE, at the first launch of this library.
///
/// On, each launch is bracketed by two `cudaEvent`s recorded on the call's stream -- the time between
/// them is the kernel alone, even when the stream was busy before ( the first event fires when the
/// stream reaches it ). Nothing waits at launch: the pairs are kept PENDING and resolved when the
/// host reads the totals ( `loom_kernel_timing_read`, which synchronizes on them ). Each kernel
/// instantiation of this library has a SLOT, which also carries what the compiler made of it --
/// registers, local memory ( spills and stack ), the occupancy at the launch's block size --, read
/// once with `cudaFuncGetAttributes` / `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
///
/// The C functions at the end are looked up per library by `loom/devices/kernel_timing.py` ( one
/// generated library = one loom call; every library has its own copy of this registry ).
namespace detail::CudaQueueTiming {
    struct Slot {
        const char *kind;                    ///< "flat" or "grouped"
        double      ms           = 0;        ///< accumulated, resolved pairs only
        long long   count        = 0;        ///< resolved launches
        int         regs         = 0;        ///< registers per thread
        int         local_bytes  = 0;        ///< local memory per thread ( spills, stack arrays )
        int         static_shared= 0;        ///< static shared memory per block
        int         block        = 0;        ///< threads per block of the last launch
        int         grid         = 0;        ///< blocks of the last launch
        int         blocks_per_sm= 0;        ///< resident blocks per SM at that block size
        int         max_threads_per_sm = 0;  ///< the card's
        int         nb_sm        = 0;        ///< the card's
    };

    struct Pending {
        cudaEvent_t start, stop;
        int         slot;
    };

    inline bool enabled_by_env() {
        const char *v = std::getenv( "LOOM_KERNEL_TIMING" );
        return v && *v && std::strcmp( v, "0" ) && std::strcmp( v, "false" ) && std::strcmp( v, "no" ) && std::strcmp( v, "off" );
    }

    struct Registry {
        bool                       enabled = enabled_by_env();
        std::vector<Slot>          slots;
        std::vector<const void *>  kernels;   ///< the kernel of each slot
        std::vector<Pending>       pending;
        std::mutex                 mutex;

        void resolve() {
            for ( Pending &p : pending ) {
                float ms = 0;
                if ( cudaEventSynchronize( p.stop ) == cudaSuccess && cudaEventElapsedTime( &ms, p.start, p.stop ) == cudaSuccess ) {
                    slots[ p.slot ].ms += ms;
                    slots[ p.slot ].count += 1;
                }
                cudaEventDestroy( p.start );
                cudaEventDestroy( p.stop );
            }
            pending.clear();
        }
    };

    inline Registry &registry() {
        static Registry r;
        return r;
    }

    /// the slot of ONE kernel ( by its address ), its attributes read on the first call
    inline int slot_of( const void *kernel, const char *kind, int block, int dynamic_shared ) {
        Registry &r = registry();
        std::lock_guard<std::mutex> lock( r.mutex );
        int slot = -1;
        for ( int i = 0; i < int( r.kernels.size() ); ++i )
            if ( r.kernels[ i ] == kernel )
                slot = i;
        if ( slot < 0 ) {
            Slot s;
            s.kind = kind;
            cudaFuncAttributes attr;
            if ( cudaFuncGetAttributes( &attr, kernel ) == cudaSuccess ) {
                s.regs          = attr.numRegs;
                s.local_bytes   = int( attr.localSizeBytes );
                s.static_shared = int( attr.sharedSizeBytes );
            }
            int dev = 0;
            cudaGetDevice( &dev );
            cudaDeviceGetAttribute( &s.max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev );
            cudaDeviceGetAttribute( &s.nb_sm, cudaDevAttrMultiProcessorCount, dev );
            slot = int( r.slots.size() );
            r.slots.push_back( s );
            r.kernels.push_back( kernel );
        }
        Slot &s = r.slots[ slot ];
        if ( s.block != block ) {
            s.block = block;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor( &s.blocks_per_sm, kernel, block, dynamic_shared );
        }
        return slot;
    }

    /// bracket ONE launch: `launch()` between two events of `stream`
    template<class KernelPtr>
    void timed_launch( KernelPtr kernel, const char *kind, int grid, int block, int dynamic_shared, cudaStream_t stream, auto &&launch ) {
        Registry &r = registry();
        if ( ! r.enabled ) {
            launch();
            return;
        }
        const int slot = slot_of( ( const void * ) kernel, kind, block, dynamic_shared );
        Pending p{ nullptr, nullptr, slot };
        cuda_check( cudaEventCreate( &p.start ), "event (timing)" );
        cuda_check( cudaEventCreate( &p.stop ), "event (timing)" );
        cuda_check( cudaEventRecord( p.start, stream ), "event record (timing)" );
        launch();
        cuda_check( cudaEventRecord( p.stop, stream ), "event record (timing)" );
        std::lock_guard<std::mutex> lock( r.mutex );
        r.slots[ slot ].grid = grid;
        r.pending.push_back( p );
    }
}

/// the C side of the timing, per library ( see above ). `loom_kernel_timing_read( slot, ms, ints )`:
/// `ints` receives 10 values, `count, regs, local_bytes, static_shared, block, grid, blocks_per_sm,
/// max_threads_per_sm, nb_sm, is_grouped`; returns the number of slots ( call it with `slot = -1` to
/// only get that number ). It waits for the pending launches first.
extern "C" __attribute__(( used, visibility( "default" ) )) inline int loom_kernel_timing_enabled() {
    return detail::CudaQueueTiming::registry().enabled;
}

extern "C" __attribute__(( used, visibility( "default" ) )) inline int loom_kernel_timing_read( int slot, double *ms, long long *ints ) {
    auto &r = detail::CudaQueueTiming::registry();
    std::lock_guard<std::mutex> lock( r.mutex );
    r.resolve();
    if ( slot >= 0 && slot < int( r.slots.size() ) ) {
        const auto &s = r.slots[ slot ];
        *ms = s.ms;
        const long long v[] = { s.count, s.regs, s.local_bytes, s.static_shared, s.block, s.grid, s.blocks_per_sm,
                                s.max_threads_per_sm, s.nb_sm, std::strcmp( s.kind, "grouped" ) == 0 };
        for ( int i = 0; i < 10; ++i )
            ints[ i ] = v[ i ];
    }
    return int( r.slots.size() );
}

/// zeroes the totals ( after waiting for the pending launches, so that none lands in the next window )
extern "C" __attribute__(( used, visibility( "default" ) )) inline void loom_kernel_timing_reset() {
    auto &r = detail::CudaQueueTiming::registry();
    std::lock_guard<std::mutex> lock( r.mutex );
    r.resolve();
    for ( auto &s : r.slots ) {
        s.ms = 0;
        s.count = 0;
    }
}

// ── reductions: a per-thread accumulator in registers, combined atomically at the end ──────────
namespace detail::CudaQueueLaunch {
    /// atomic `target = op( target, value )`, by CAS on the 32 or 64-bit representation --
    /// generic over the operator, which `atomicAdd` alone does not give (`maximum` on a double).
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
            static_assert( sizeof( T ) == 8, "atomic_combine: 32 or 64 bits" );
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

    /// a device-side reduction target: the scalar that receives the contributions, and its
    /// re-read / release once the kernel is finished
    template<class Op,class T>
    struct DeviceTarget {
        Op  op;
        T  *host;
        T  *dev;
    };

    template<class Op,class T>
    DeviceTarget<Op,T> device_target_for( const ReductionTarget<Op,T> &target, cudaStream_t stream ) {
        T *dev;
        cuda_check( cudaMallocAsync( ( void ** ) &dev, sizeof( T ), stream ), "malloc (reduction)" );
        const T identity = Op::identity();
        cuda_check( cudaMemcpyAsync( dev, &identity, sizeof( T ), cudaMemcpyHostToDevice, stream ), "memcpy (identity)" );
        cuda_check( cudaStreamSynchronize( stream ), "sync (identity)" ); // `identity` is on the stack
        return { target.op, target.host, dev };
    }

    template<class Op,class T>
    __device__ CudaReducer<Op,T> reducer_for( const DeviceTarget<Op,T> &t ) {
        return { t.op, Op::identity(), t.dev };
    }

    template<class Func,class Item,class... Args>
    __device__ void call( Func &func, Item item, SI flat_index, int thread_index, int nb_threads, Args &...args ) {
        if constexpr ( requires { func( item, flat_index, thread_index, nb_threads, args... ); } )
            func( item, flat_index, thread_index, nb_threads, args... );
        else if constexpr ( requires { func( item, thread_index, nb_threads, args... ); } )
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
                    call( func, item_list[ index ], SI( index ), t, nb_threads, reds..., args... );
                ( reds.flush(), ... );
            }, reducers );
        }, targets );
    }

    /// the cooperative counterpart of `call`: the flat rank is added to it in the same way, and a functor
    /// that does not take it keeps working.
    template<class Func,class Item,class... Rest>
    __device__ void call_grouped( Func &func, Item item, SI flat_index, Rest &&...rest ) {
        if constexpr ( requires { func( item, flat_index, rest... ); } )
            func( item, flat_index, rest... );
        else
            func( item, rest... );
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
            call_grouped( func, item_list[ index ], SI( index ), g, lane, local_size, group, local_scratch, sub_group, args... );
    }
}

/// Flat launch: `nb_threads` threads, strided over the items; per-thread reductions in
/// registers, combined atomically into a device scalar re-read by the event's finalizer.
template<class Deps,class Func,class ItemList,class Targets,class... Args>
auto submit_kernel( const CudaQueue &queue, const Deps &deps, Func &&func, ItemList &&item_list,
                    int nb_items, int nb_threads, Targets reduction_targets, Args &&...args ) {
    using namespace detail::CudaQueueLaunch;
    deps.wait_all();
    cudaStream_t stream = queue.stream;

    auto targets = std::apply( [&]( auto &...t ) { return std::make_tuple( device_target_for( t, stream )... ); }, reduction_targets );

    const int block = 128, grid = ( nb_threads + block - 1 ) / block;
    // the instantiation `<<< >>>` would deduce, named so that the timing can ask for its attributes
    auto kernel = &flat_kernel<std::decay_t<Func>, std::decay_t<ItemList>, decltype( targets ), std::decay_t<Args>...>;
    detail::CudaQueueTiming::timed_launch( kernel, "flat", grid, block, 0, stream, [&] {
        kernel<<<grid, block, 0, stream>>>( func, item_list, nb_items, nb_threads, targets, args... );
    } );
    cuda_check( cudaGetLastError(), "launch" );

    QueueEvent ev( [stream]{ cuda_check( cudaStreamSynchronize( stream ), "sync" ); } );
    std::apply( [&]( auto &...t ) {
        ( ev.finalizers.push_back( [t]{
            cuda_check( cudaMemcpy( t.host, t.dev, sizeof( *t.host ), cudaMemcpyDeviceToHost ), "memcpy (reduction)" );
            cudaFree( t.dev );
        } ), ... );
    }, targets );
    return ev;
}

/// Cooperative launch: one BLOCK of `group_size` lanes per concurrent item, `local_elems` `int32`
/// words of shared memory (`local_scratch`), `group_barrier` on `__syncthreads`.
template<class Deps,class Func,class ItemList,class Targets,class... Args>
auto submit_kernel_grouped( const CudaQueue &queue, const Deps &deps, Func &&func, ItemList &&item_list,
                            int nb_items, int nb_groups, int group_size, int local_elems, Targets, Args &&...args ) {
    using namespace detail::CudaQueueLaunch;
    static_assert( std::tuple_size_v<Targets> == 0, "reductions are not supported in a group kernel" );
    deps.wait_all();
    cudaStream_t stream = queue.stream;
    const int shared = int( sizeof( std::int32_t ) * std::max( local_elems, 1 ) );
    auto kernel = &grouped_kernel<std::decay_t<Func>, std::decay_t<ItemList>, std::decay_t<Args>...>;
    detail::CudaQueueTiming::timed_launch( kernel, "grouped", nb_groups, group_size, shared, stream, [&] {
        kernel<<<nb_groups, group_size, shared, stream>>>( func, item_list, nb_items, nb_groups, args... );
    } );
    cuda_check( cudaGetLastError(), "launch (groups)" );
    return QueueEvent( [stream]{ cuda_check( cudaStreamSynchronize( stream ), "sync (groups)" ); } );
}

} // namespace sdot
