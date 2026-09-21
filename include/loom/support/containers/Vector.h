#pragma once

#include <loom/support/common_macros.h> // HD

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// static size vector (like a std::array)
template<class T,int ct_size>
class alignas( T ) Vector {
public:
    using             value_type               = T;

    /**/              HD Vector                ( const Vector &that );
    /**/              HD Vector                ( Vector &&that ) noexcept;
    /**/              HD Vector                ( auto &&that );

    /**/              HD Vector                ( FillWith, auto &&...ctor_args );
    /**/              HD Vector                ( Function, auto &&func );
    /**/              HD Vector                ( Values, auto &&...values );
    /**/              HD Vector                ( Reserved ); // do not call new on items
    /**/              HD Vector                ();


    /**/             HD ~Vector                ();

    HD Vector&        operator=                ( const Vector &that );
    HD Vector&        operator=                ( Vector &&that ) noexcept;

    HD const T&       operator[]               ( PI index ) const;
    HD T&             operator[]               ( PI index );

    HD const T&       operator()               ( PI index ) const { return operator[]( index ); }
    HD T&             operator()               ( PI index ) { return operator[]( index ); }

    // bool           operator<                ( const Vector &that ) const;

    HD static Vector  with_value_at            ( PI index, T value ); ///< 0 ... 0 value 0 ... 0. `value` is positionned at `index`
    HD static Vector  with_func                ( auto &&func );
    HD static Vector  zeros                    ( );
    HD static Vector  ones                     ( );

    HD auto           with_pushed_value        ( T value ) const;
    HD auto           without_index            ( PI ind_to_remove ) const;
    // T_d auto       from                     () const { return Vector<T,Arch,ct_size-d>( std::span( begin() + d, end() ) ); }

    HD constexpr auto size                     () const { return Ct<int,ct_size>(); }

    HD const T*       data                     () const;
    HD T*             data                     ();

    HD auto           begin                    () const;
    HD auto           begin                    ();
    HD auto           end                      () const;
    HD auto           end                      ();

    HD PI             arg_max                  () const;
    HD T              max                      () const;

    HD auto           is_valid                 () const { return Ct<bool,true>(); }

    friend HD Vector  normalized               ( const Vector &a ) { return a / norm_2( a ); }
    friend HD T       norm_2_p2                ( const Vector &a ) { return dot( a, a ); }
    friend HD T       norm_2                   ( const Vector &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

    friend HD Vector  operator+                ( const Vector &a ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
    friend HD Vector  operator-                ( const Vector &a ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
    friend HD Vector  floor                    ( const Vector &a ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = floor( a[ i ] ); return res; }
    friend HD Vector  ceil                     ( const Vector &a ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = ceil ( a[ i ] ); return res; }

    friend HD Vector  operator+                ( const Vector &a, const Vector &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
    friend HD Vector  operator-                ( const Vector &a, const Vector &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }
    friend HD Vector  operator*                ( const T &a, const Vector &b ) { Vector res; for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
    friend HD Vector  operator/                ( const Vector &a, const T &b ) { Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }
    friend HD Vector  max                      ( const Vector &a, const Vector &b ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::max( a[ i ], b[ i ] ); return res; }
    friend HD Vector  min                      ( const Vector &a, const Vector &b ) { using namespace std; Vector res; for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::min( a[ i ], b[ i ] ); return res; }
    friend HD T       dot                      ( const Vector &a, const Vector &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

    friend HD void    operator+=               ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
    friend HD void    operator-=               ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
    // `template<class B>` épelé, et non `const auto &b` : un ami template abrégé défini dans une
    // classe template fait perdre à nvcc (EDG) les membres qui suivent (« has no member _storage »)
    template<class B> friend HD void operator/=( Vector &a, const B &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

    template<class F> friend HD void _for_each_in_range( const Vector &beg, const Vector &end, Vector &cur, int i, const F &func ) { if ( i == beg.size() ) { func( cur ); return; } for( T v = beg[ i ]; v < end[ i ]; ++v ) { cur[ i ] = v; _for_each_in_range( beg, end, cur, i + 1, func ); } }
    template<class F> friend HD void for_each_in_range( const Vector &beg, const Vector &end, F &&func ) { Vector cur = beg; _for_each_in_range( beg, end, cur, 0, func ); }
    template<class F> friend HD void for_each_in_range( const Vector &end, F &&func ) { Vector beg( Size(), end.size(), 0 ); for_each_in_range( beg, end, func ); }

    char              _storage                 [ sizeof( T ) * ct_size ];
};


} // namespace sdot

#include "Vector.cxx" // IWYU pragma: export
