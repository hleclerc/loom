#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// static size vector (like a std::array)
template<class T,int ct_size>
class alignas( T ) Vector {
public:
    using             value_type               = T;

    /**/              Vector                   ( const Vector &that );
    /**/              Vector                   ( Vector &&that ) noexcept;
    /**/              Vector                   ( auto &&that );

    /**/              Vector                   ( FillWith, auto &&...ctor_args );
    /**/              Vector                   ( Function, auto &&func );
    /**/              Vector                   ( Values, auto &&...values );
    /**/              Vector                   ( Reserved ); // do not call new on items
    /**/              Vector                   ();


    /**/             ~Vector                   ();

    Vector&           operator=                ( const Vector &that );
    Vector&           operator=                ( Vector &&that ) noexcept;

    const T&          operator[]               ( PI index ) const;
    T&                operator[]               ( PI index );

    const T&          operator()               ( PI index ) const { return operator[]( index ); }
    T&                operator()               ( PI index ) { return operator[]( index ); }

    // bool           operator<                ( const Vector &that ) const;

    static Vector     with_value_at            ( PI index, T value ); ///< 0 ... 0 value 0 ... 0. `value` is positionned at `index`
    static Vector     with_func                ( auto &&func );
    static Vector     zeros                    ( );
    static Vector     ones                     ( );

    auto              with_pushed_value        ( T value ) const;
    auto              without_index            ( PI ind_to_remove ) const;
    // T_d auto       from                     () const { return Vector<T,Arch,ct_size-d>( std::span( begin() + d, end() ) ); }

    constexpr auto    size                     () const { return Ct<int,ct_size>(); }

    const T*          data                     () const;
    T*                data                     ();

    auto              begin                    () const;
    auto              begin                    ();
    auto              end                      () const;
    auto              end                      ();

    PI                arg_max                  () const;
    T                 max                      () const;

    auto              is_valid                 () const { return Ct<bool,true>(); }

    friend Vector     normalized               ( const Vector &a ) { return a / norm_2( a ); }
    friend T          norm_2_p2                ( const Vector &a ) { return dot( a, a ); }
    friend T          norm_2                   ( const Vector &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

    friend Vector     operator+                ( const Vector &a ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
    friend Vector     operator-                ( const Vector &a ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
    friend Vector     floor                    ( const Vector &a ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = floor( a[ i ] ); return res; }
    friend Vector     ceil                     ( const Vector &a ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = ceil ( a[ i ] ); return res; }

    friend Vector     operator+                ( const Vector &a, const Vector &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
    friend Vector     operator-                ( const Vector &a, const Vector &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }
    friend Vector     operator*                ( const T &a, const Vector &b ) { Vector res; for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
    friend Vector     operator/                ( const Vector &a, const T &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }
    friend Vector     max                      ( const Vector &a, const Vector &b ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::max( a[ i ], b[ i ] ); return res; }
    friend Vector     min                      ( const Vector &a, const Vector &b ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::min( a[ i ], b[ i ] ); return res; }
    friend T          dot                      ( const Vector &a, const Vector &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

    friend void       operator+=               ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
    friend void       operator-=               ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
    friend void       operator/=               ( Vector &a, const auto &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

    friend void       _for_each_in_range       ( const Vector &beg, const Vector &end, Vector &cur, int i, const auto &func ) { if ( i == beg.size() ) { func( cur ); return; } for( T v = beg[ i ]; v < end[ i ]; ++v ) { cur[ i ] = v; _for_each_in_range( beg, end, cur, i + 1, func ); } }
    friend void       for_each_in_range        ( const Vector &beg, const Vector &end, auto &&func ) { Vector cur = beg; _for_each_in_range( beg, end, cur, 0, func ); }
    friend void       for_each_in_range        ( const Vector &end, auto &&func ) { Vector beg( Size(), end.size(), 0 ); for_each_in_range( beg, end, func ); }

    char              _storage                 [ sizeof( T ) * ct_size ];
};


} // namespace sdot

#include "Vector.cxx" // IWYU pragma: export
