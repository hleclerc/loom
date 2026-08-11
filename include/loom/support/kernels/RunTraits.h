#pragma once

#include "../containers/Tuple.h"
#include "../common_macros.h"
#include "../Ct.h"
#include <thread>

namespace sdot {

namespace RunTraits {

// member-call probes for the optional hooks a functor may surdefine (see is_detected)
namespace detail {
    template<class F,class...A> using m_per_thread    = decltype( std::declval<F&>().per_thread( std::declval<A>()... ) );
    template<class F,class...A> using m_max_cpu       = decltype( std::declval<F&>().max_cpu_threads( std::declval<A>()... ) );
    template<class F,class...A> using m_local_gpu_mem = decltype( std::declval<F&>().local_gpu_memory_size( std::declval<A>()... ) );
    template<class F,class...A> using m_nb_gpu_regs   = decltype( std::declval<F&>().nb_gpu_register_per_thread( std::declval<A>()... ) );
    template<class F,class...A> using m_max_gpu       = decltype( std::declval<F&>().max_gpu_threads( std::declval<A>()... ) );
}

// ----------------- generic -----------------

/// add args computed for each thread
template<class A,class B,class C,class D,class... E>
HD void per_thread( A &func, B &&thread_info, C &&list, D &&cont, E &&...args ) {
    if constexpr( IS_DETECTED( detail::m_per_thread, A, B, C, D, E... ) )
        func.per_thread( FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... );
    else
        cont( FORWARD( args )... ); //
}

// ----------------- CPU -----------------

T_Tv HD int max_cpu_threads( T &&func, V &&...args ) {
    if constexpr( IS_DETECTED( detail::m_max_cpu, T, V... ) )
        return func.max_cpu_threads( args... );
    else
        return std::thread::hardware_concurrency();
}

// ----------------- GPU -----------------

T_Tv HD int local_gpu_memory_size( T &&func, V &&...args ) {
    if constexpr( IS_DETECTED( detail::m_local_gpu_mem, T, V... ) )
        return func.local_gpu_memory_size( args... );
    else
        return Ct<int,1>();
}

T_Tv HD int nb_gpu_register_per_thread( T &&func, V &&...args ) {
    if constexpr( IS_DETECTED( detail::m_nb_gpu_regs, T, V... ) )
        return func.nb_gpu_register_per_thread( args... );
    else
        return Ct<int,16>();
}

T_Tv HD int max_gpu_threads( T &&func, V &&...args ) {
    if constexpr( IS_DETECTED( detail::m_max_gpu, T, V... ) )
        return func.max_gpu_threads( args... );
    else
        // simple default to start; TODO: derive from nb_gpu_register_per_thread / local_gpu_memory_size
        return 1 << 20;
}

// ----------------- util -----------------

/// Wrap a func, allowing to surdefine methods
template<class Func>
struct RunFunctorWrapper {
    T_VT  HD int     nb_gpu_register_per_thread( T &&...args ) { return sdot::RunTraits::nb_gpu_register_per_thread( func, FORWARD( args )... ); }
    T_VT  HD int     local_gpu_memory_size     ( T &&...args ) { return sdot::RunTraits::local_gpu_memory_size( func, FORWARD( args )... ); }
    T_VT  HD int     max_gpu_threads           ( T &&...args ) { return sdot::RunTraits::max_gpu_threads( func, FORWARD( args )... ); }

    T_VT  HD int     max_cpu_threads           ( T &&...args ) { return sdot::RunTraits::max_cpu_threads( func, FORWARD( args )... ); }

    T_ABCV HD void   per_thread                ( A &&thread_info, B &&list, C &&cont, V &&...args ) { return sdot::RunTraits::per_thread( func, FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... ); }

    T_VT  HD auto    operator()                ( T &&...args ) const { func( FORWARD( args )... ); }

    Func             func;
};

} // RunTraits

} // namespace sdot
