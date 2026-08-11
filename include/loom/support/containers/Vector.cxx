#pragma once

#include <algorithm>
#include <utility>

#include "../algorithms/for_each_item.h"
#include "Vector.h"

namespace sdot {

#define UTP template<class T,int ct_size>
#define DTP Vector<T,ct_size>

UTP DTP::Vector( auto &&values ) {
    PI i = 0;
    sdot::for_each_item( values, [&]( const auto &item ) {
        if ( i < ct_size )
            new ( data() + i++ ) T( item );
    } );
    for( ; i < ct_size; ++i )
        new ( data() + i ) T;
}

UTP DTP::Vector( FillWith, auto &&...ctor_args ) {
    for( auto &v : *this )
        new ( &v ) T( ctor_args... );
}

UTP DTP::Vector( Function, auto &&func ) {
    for( PI i = 0; i < ct_size; ++i )
        new ( &operator[]( i ) ) T( func( i ) );
}

UTP DTP::Vector( Reserved ) {
}

UTP DTP::Vector() {
    for( auto &v : *this )
        new ( &v ) T;
}

UTP T_VA DTP::Vector( Values, A &&...values ) : Vector( Reserved() ) {
    PI i = 0;
    auto append = [&]( auto &&value ) {
        new ( data() + i++ ) T( FORWARD( value ) );
    };
    ( append( FORWARD( values ) ), ... );
}

UTP DTP::Vector( const Vector &that ) : Vector( Reserved() ) {
    for ( PI i = 0; i < that.size(); ++i )
        new ( data() + i ) T( that[ i ] );
}

UTP DTP::Vector( Vector &&that ) noexcept {
    for ( PI i = 0; i < PI( ct_size ); ++i )
        new ( data() + i ) T( std::move( that[ i ] ) );
}

UTP Vector<T,ct_size>& DTP::operator=( const Vector &that ) {
    if ( this != &that )
        for ( PI i = 0; i < size(); ++i )
            operator[]( i ) = that[ i ];
    return *this;
}

UTP DTP& DTP::operator=( Vector &&that ) noexcept {
    if ( this != &that )
        for ( PI i = 0; i < PI( ct_size ); ++i )
            operator[]( i ) = std::move( that[ i ] );
    return *this;
}

UTP DTP::~Vector() {
    for ( PI i = 0; i < size(); ++i )
        data()[ i ].~T();
}

UTP const T& DTP::operator[]( PI index ) const {
    return data()[ index ];
}

UTP T& DTP::operator[]( PI index ) {
    return data()[ index ];
}

// UTP bool DTP::operator<( const Vector &that ) const {
//     return std::ranges::lexicographical_compare( operator std::span<T>(), that.operator std::span<T>() );
// }

UTP T_U DTP DTP::with_func( U &&func ) {
    Vector res;
    for ( PI i = 0; i < ct_size; ++i )
        new ( res.data() + i ) T( func( i ) );
    return res;
}

UTP DTP DTP::zeros() {
    return with_func( [] ( PI ) { return T( 0 ); } );
}

UTP DTP DTP::ones() {
    return with_func( [] ( PI ) { return T( 1 ); } );
}

UTP DTP DTP::with_value_at( PI index, T value ) {
    return with_func( [=] ( PI i ) { return i == index ? value : T( 0 ); } );
}

UTP auto DTP::with_pushed_value( T value ) const {
    Vector<T,ct_size+1> res( Reserved{} );
    for( PI i = 0; i < size(); ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    new ( res.data() + size() ) T( value );
    return res;
}

UTP auto DTP::without_index( PI ind_to_remove ) const {
    Vector<T,ct_size-1> res( Reserved{} );
    for( PI i = 0; i < ind_to_remove; ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        new ( res.data() + i - 1 ) T( operator[]( i ) );
    return res;
}

UTP const T* DTP::data() const {
    return reinterpret_cast<const T *>( this->_storage );
}

UTP T* DTP::data() {
    return reinterpret_cast<T *>( this->_storage );
}

UTP auto DTP::begin() const {
    return data();
}

UTP auto DTP::begin() {
    return data();
}

UTP auto DTP::end() const {
    return data() + size();
}

UTP auto DTP::end() {
    return data() + size();
}

UTP PI DTP::arg_max() const {
    PI res = 0;
    for( PI i = 1; i < size(); ++i )
        if ( operator[]( res ) < operator[]( i ) )
            res = i;
    return res;
}

UTP T DTP::max() const {
    T res = operator[]( 0 );
    for( PI i = 1; i < size(); ++i )
        if ( res < operator[]( i ) )
            res = operator[]( i );
    return res;
}

#undef UTP
#undef UTP
#undef DTP

} // namespace sdot
