#pragma once

#include <type_traits> // IWYU pragma: export

#define T_ABCV template<class A,class B,class C,class... V>
#define T_TdA  template<class T,int d,class A>
#define T_TAv  template<class T,class A,class... V>
#define T_TAB  template<class T,class A,class B>
#define T_VT   template<class... T>
#define T_VA   template<class... A>
#define T_Up   template<class U,std::size_t p>
#define T_Uu   template<class U,U u>
#define T_TA   template<class T,class A>
#define T_Tv   template<class T,class... V>
#define T_Td   template<class T,int d>
#define T_T    template<class T>
#define T_U    template<class U>
#define T_d    template<int d>
#define T_p    template<PI p>

#define SCInt static constexpr int

// `HD` marks what must exist on both sides, host and device: empty everywhere except under nvcc, where
// it is `__host__ __device__`. This is the ONLY place the attribute is spelled -- application code
// writes `HD`, never `__device__`.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#define HD_INLINE HD inline

// `LOOM_CONSTANT( declaration )`: a global constant usable on both sides -- a quadrature table
// read with a loop index, an axis label on which one calls
// `num_vertex = i`. A global-scope `constexpr` is not usable in device code as soon as
// it is ODR-used (non-constant index, `this` of a method); nvcc compiles the same source
// twice, and the device pass wants a `__device__` -- which is what we give it, there and only there.
//     LOOM_CONSTANT( double gl8_x[ 4 ] ) = { ... };
//     LOOM_TAG( _num_vertex, num_vertex );
#ifdef __CUDA_ARCH__
#define LOOM_CONSTANT( ... ) static __device__ const __VA_ARGS__
#else
#define LOOM_CONSTANT( ... ) inline constexpr __VA_ARGS__
#endif
#define LOOM_TAG( Type, name ) LOOM_CONSTANT( Type name ){}

// `LOOM_EXPORT`: a symbol a library PUBLISHES (everything is hidden by default,
// `-fvisibility=hidden`) -- a kernel's entry point, the runtime's thread pool.
#if defined( _WIN32 )
#define LOOM_EXPORT __declspec( dllexport )
#else
#define LOOM_EXPORT __attribute__(( visibility( "default" ) ))
#endif

#define ASSERTED_EQUAL( A, B ) ( []( auto a, auto b ) { if ( a != b ) throw std::runtime_error( #A " and " #B " are not equal" ); return a; } )( A, B )
#define DECAYED_TYPE_OF( v )   std::decay_t<decltype( v )>
#define IS_BASE_OF( A, V )     std::is_base_of_v<A,std::decay_t<V>>
#define CT_VALUE( v )          std::decay_t<decltype( v )>::value
#define FORWARD( v )           std::forward<decltype( v )>( v )

// Detection idiom helpers — C++14-compatible replacement for requires{} expressions.
// All trait structs live in sdot::detail to avoid global-namespace pollution.
// The macros expose them at any call site, including inside other namespaces.
namespace sdot { namespace detail {
    template<class...> using void_t = void;  // C++17 std::void_t, portable for Metal

    template<class T,class=void> struct has_static_value : std::false_type {};
    T_T struct has_static_value<T,void_t<decltype(T::value)>> : std::true_type {};

    template<class T,class=void> struct has_ct_rank : std::false_type {};
    T_T struct has_ct_rank<T,void_t<decltype(T::ct_rank)>> : std::true_type {};

    template<class T,class=void> struct has_size_method : std::false_type {};
    T_T struct has_size_method<T,void_t<decltype(std::declval<T>().size())>> : std::true_type {};

    template<class T,bool=has_size_method<T>::value> struct has_constexpr_size : std::false_type {};
    T_T struct has_constexpr_size<T,true> : has_static_value<DECAYED_TYPE_OF( std::declval<T>().size() )> {};

    template<class R=void> struct AnyFunc { T_VT HD R operator()( T&&...) const { if constexpr ( ! std::is_void_v<R> ) return *reinterpret_cast<R *>( 0ul ); } };

    // generic detection idiom (Library Fundamentals TS): is Op<A...> well-formed?
    // Op is an alias template wrapping the probed expression (e.g. a member call).
    template<class AlwaysVoid,template<class...>class Op,class...A> struct detector              : std::false_type {};
    template<template<class...>class Op,class...A> struct detector<void_t<Op<A...>>,Op,A...>     : std::true_type  {};
    template<template<class...>class Op,class...A> using is_detected = detector<void,Op,A...>;
} }

#define HAS_CONSTEXPR_SIZE( expr ) ::sdot::detail::has_constexpr_size<DECAYED_TYPE_OF( expr )>::value
#define HAS_STATIC_VALUE( expr )   ::sdot::detail::has_static_value<DECAYED_TYPE_OF( expr )>::value
#define IS_DETECTED( Op, ... )     ::sdot::detail::is_detected<Op,__VA_ARGS__>::value
#define HAS_CT_RANK( T )           ::sdot::detail::has_ct_rank<T>::value

// ---- OPTIONAL bounds checking (`-DLOOM_BOUNDS_CHECK`) ------------------------------------------
//
// `TensorView::squeeze` is the SINGLE point every indexing goes through: bounding the index there
// catches any overrun, in any kernel, without a sanitizer.
//
// Why it is not always on: it is one test per access, on the hottest path of the
// code. Why it exists: an out-of-bounds index does NOT always fault -- it often lands
// in neighbouring mapped memory, and the computation carries on wrong, or faults minutes later
// in an innocent kernel. `compute-sanitizer` sees it, but it is slow and changes the timing enough
// to make race bugs disappear. This guard, for its part, is deterministic and names the line.
//
// Usage: `LOOM_BOUNDS_CHECK=1 ./run test ...`.
#ifdef LOOM_BOUNDS_CHECK
#include <cassert>
#define LOOM_CHECK_INDEX( index, extent ) \
    assert( SI( index ) >= 0 && SI( index ) < SI( extent ) && "TensorView: index out of bounds" )
#else
#define LOOM_CHECK_INDEX( index, extent ) ( (void) 0 )
#endif
