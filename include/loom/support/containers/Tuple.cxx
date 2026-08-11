#pragma once

#include "../util/TypePromote.h"
#include "Tuple.h"
// #include <utility>

namespace sdot {

namespace TupleDetail {
    // Functor replacements for the pack-capturing generic lambdas used by Tuple's methods.
    // nvcc forbids generic / by-reference extended __host__ __device__ lambdas, so these have
    // to be functors for the methods to be callable from device code.

    /// calls cb( head, tail... ) — prepends `head` when unwinding apply_values
    template<class CB,class Head>
    struct PrependApply {
        CB   cb;
        Head head;
        T_VT auto operator()( T &&...tail ) const { return cb( head, FORWARD( tail )... ); }
    };

    /// calls tuple( values..., extra ) — used by with_appended_value
    template<class Extra>
    struct AppendValue {
        Extra extra;
        T_VT auto operator()( T &&...values ) const { return tuple( FORWARD( values )..., extra ); }
    };
}

// ---- runtime front value -----------------------------------------------------------
#define UTP template<class Head,class... Tail>
#define DTP Tuple<Head,Tail...>

UTP T_TA DTP::Tuple( Function, T &&func, A index ) : head( func( index ) ), tail( Function(), FORWARD( func ), index + Ct<int,1>() ) {
}

UTP T_T DTP::Tuple( Function, T &&func ) : Tuple( Function(), func, Ct<int,0>() ) {
}

UTP T_VT DTP::Tuple( const Tuple<T...> &that ) : head( that.head ), tail( that.tail ) {
}

UTP T_Tv DTP::Tuple( Values, T head, V... tail ) : head( head ), tail( Values(), tail... ) {
}

UTP T_T void DTP::for_each_item( T &&cb ) const {
     cb( head );
     tail.for_each_item( FORWARD( cb ) );
}

UTP T_T void DTP::for_each_item( T &&cb ) {
     cb( head );
     tail.for_each_item( FORWARD( cb ) );
}

UTP T_T auto DTP::apply_values( T &&cb ) const {
    return tail.apply_values( TupleDetail::PrependApply<DECAYED_TYPE_OF( cb ),Head>{ FORWARD( cb ), head } );
}

UTP T_T auto DTP::operator[]( T &&index ) const {
    if constexpr ( HAS_STATIC_VALUE( index ) ) { //  requires { DECAYED_TYPE_OF( index )::value; }
        if constexpr ( DECAYED_TYPE_OF( index )::value )
            return tail[ index - Ct<int,1>() ];
        else
            return head;
    } else {
        using TR = typename TypePromote<Head,Tail...>::type;
        if constexpr ( sizeof...( Tail ) == 0 )
            return TR( head );
        else if ( index == 0 )
            return TR( head );
        else
            return TR( tail[ index - 1_c ] );
    }
}

UTP T_TA void DTP::set( T &&index, A &&value ) {
    if constexpr ( HAS_STATIC_VALUE( index ) ) { // requires { DECAYED_TYPE_OF( index )::value; }
        if constexpr ( DECAYED_TYPE_OF( index )::value )
            tail[ index - Ct<int,1>() ].set( FORWARD( value ) );
        else
            head = FORWARD( value );
    } else {
        if constexpr ( sizeof...( Tail ) == 0 )
            head = FORWARD( value );
        else if ( index == 0 )
            head = FORWARD( value );
        else
            tail[ index - 1_c ].set( value );
    }
}

UTP T_T auto DTP::operator==( const T &that ) const {
    return apply_values( [&]( const auto &...a ) {
        return that.apply_values( [&]( const auto &...b ) {
            if constexpr ( sizeof...( a ) == sizeof...( b ) )
                return ( ( a == b ) && ... && Ct<bool,true>() );
            else
                return Ct<bool,false>();
        } );
    } );
}

UTP auto DTP::size() {
    return Ct<int,1 + sizeof...( Tail )>();
}

UTP T_T auto DTP::with_appended_value( T &&new_value ) const {
    return apply_values( TupleDetail::AppendValue<DECAYED_TYPE_OF( new_value )>{ FORWARD( new_value ) } );
}

UTP T_T auto DTP::without_index( T index ) const {
    if constexpr ( HAS_STATIC_VALUE( index ) ) {
        if constexpr ( DECAYED_TYPE_OF( index )::value )
            return concat( tuple( head ), tail.without_index( index - Ct<int,1>() ) );
        else
            return tail;
    } else {
        using TR = typename TypePromote<Head,Tail...>::type;
        return apply_values( [&]( auto ...vals ) {
            TR rt_vals[ sizeof...( vals ) ] = { TR( vals )... };
            return Tuple<std::conditional_t<1,TR,Tail>...>( Function(), [&]( auto i ) {
                return rt_vals[ i + ( i >= index ) ];
            } );
        } );
    }
}

#undef UTP
#undef DTP

// ---- rank 0 ------------------------------------------------------------------------
#define UTP // template<>
#define DTP Tuple<>

UTP T_TA DTP::Tuple( Function, T &&/*func*/, A /*index*/ ) {
}

UTP T_T DTP::Tuple( Function, T &&/*func*/ ) {
}

UTP DTP::Tuple( Values ) {
}

UTP T_T void DTP::for_each_item( T &&/* cb */ ) const {
}

UTP T_T auto DTP::apply_values( T &&cb ) const {
    return cb();
}

UTP T_T Void DTP::operator[]( T ) const {
    return {};
}

UTP T_TA void DTP::set( T &&/* index */, A &&/* value */ ) {
}

UTP T_T auto DTP::operator==( const T &that ) const {
    return that.size() == 0_c;
}

UTP T_T auto DTP::with_appended_value( T &&new_value ) const {
    return tuple( new_value );
}

UTP auto DTP::size() {
    return Ct<int,0>();
}

#undef UTP
#undef DTP

} // namespace sdot
