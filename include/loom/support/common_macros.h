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

    template<class R=void> struct AnyFunc { T_VT R operator()( T&&...) const { if constexpr ( ! std::is_void_v<R> ) return *reinterpret_cast<R *>( 0ul ); } };

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
