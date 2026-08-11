#pragma once

#include "util/TypePromote.h"
#include "algorithms/min.h" // IWYU pragma: export
#include "common_macros.h"
#include <type_traits>

namespace sdot {

template<class T,T i>
struct Ct {
    static constexpr T value = i;

    /**/ constexpr Ct( T value ) {
        ASSERT( value == i );
    }

    /**/ constexpr Ct() {
    }

    constexpr operator T() const {
        return i;
    }

    // selecting axes on a value that has none: there is nothing to select, so it gives back
    // itself. That is what lets a batch index be applied to a whole aggregate
    // (`cell( batch_index )`) whose members are not all mapped -- or not tensors at all.
    constexpr auto operator()( auto &&.../*index*/ ) const { return *this; }

    // as a `run_parallel` argument: la valeur est dans le type, il n'y a rien en mémoire à
    // rendre accessible -- elle traverse le kernel telle quelle, à coût nul.
    constexpr auto transfer_cost ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto make_available( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    T_U void display( U &os ) const {
        os << "Ct(" << i  << ")";
    }
};

// 5_c will produce a Ct<int,5>()
template<char... Digits>
constexpr auto operator""_c() {
    constexpr int v = [] {
        char ds[] = { Digits... };
        int r = 0;
        for ( char c : ds )
            r = r * 10 + ( c - '0' );
        return r;
    }();
    return sdot::Ct<int,v>{};
}

// 1_c will produce a Ct<bool,1>()
template<char... Digits>
constexpr auto operator""_b() {
    constexpr int v = [] {
        char ds[] = { Digits... };
        int r = 0;
        for ( char c : ds )
            r = r * 10 + ( c - '0' );
        return r;
    }();
    return sdot::Ct<bool,bool( v )>{};
}

template<class A,A i,class B,B j>
struct TypePromote<Ct<A,i>,Ct<B,j>> { static_assert( i == j ); using type = Ct<typename TypePromote<A,B>::type,i>; };

template<class A,A i,class B>
struct TypePromote<Ct<A,i>,B> { using type = typename TypePromote<A,B>::type; };

template<class A,class B,B j>
struct TypePromote<A,Ct<B,j>> { using type = typename TypePromote<A,B>::type; };


// Ct Ct
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator+ ( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 + v1 ); return Ct<TR,v0 + v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator- ( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 - v1 ); return Ct<TR,v0 - v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator* ( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 * v1 ); return Ct<TR,v0 * v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator/ ( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 / v1 ); return Ct<TR,v0 / v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator% ( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 % v1 ); return Ct<TR,v0 % v1>(); }

template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator&&( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 && v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator||( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 || v1)>(); }

template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator==( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 == v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator!=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 != v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator<=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 <= v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator>=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 >= v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator< ( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 <  v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator> ( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 >  v1)>(); }

template<class T0,T0 v0,class T1,T1 v1> constexpr auto min( Ct<T0,v0>, Ct<T1,v1> ) { using TR = TypePromote<T0,T1>::type; return Ct<TR,(v0 <= v1 ? TR( v0 ) : TR( v1 ) )>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto max( Ct<T0,v0>, Ct<T1,v1> ) { using TR = TypePromote<T0,T1>::type; return Ct<TR,(v0 >= v1 ? TR( v0 ) : TR( v1 ) )>(); }

// Ct T1
template<class T0,T0 v0,class T1> constexpr auto operator+ ( Ct<T0,v0>, T1 v1 ) { return v0 + v1; }
template<class T0,T0 v0,class T1> constexpr auto operator- ( Ct<T0,v0>, T1 v1 ) { return v0 - v1; }
template<class T0,T0 v0,class T1> constexpr auto operator* ( Ct<T0,v0>, T1 v1 ) { return v0 * v1; }
template<class T0,T0 v0,class T1> constexpr auto operator/ ( Ct<T0,v0>, T1 v1 ) { return v0 / v1; }
template<class T0,T0 v0,class T1> constexpr auto operator% ( Ct<T0,v0>, T1 v1 ) { return v0 % v1; }

template<class T0,T0 v0,class T1> constexpr auto operator&&( Ct<T0,v0>, T1 v1 ) { return v0 && v1; }
template<class T0,T0 v0,class T1> constexpr auto operator||( Ct<T0,v0>, T1 v1 ) { return v0 || v1; }

template<class T0,T0 v0,class T1> constexpr auto operator==( Ct<T0,v0>, T1 v1 ) { return v0 == v1; }
template<class T0,T0 v0,class T1> constexpr auto operator!=( Ct<T0,v0>, T1 v1 ) { return v0 != v1; }
template<class T0,T0 v0,class T1> constexpr auto operator<=( Ct<T0,v0>, T1 v1 ) { return v0 <= v1; }
template<class T0,T0 v0,class T1> constexpr auto operator>=( Ct<T0,v0>, T1 v1 ) { return v0 >= v1; }
template<class T0,T0 v0,class T1> constexpr auto operator< ( Ct<T0,v0>, T1 v1 ) { return v0 <  v1; }
template<class T0,T0 v0,class T1> constexpr auto operator> ( Ct<T0,v0>, T1 v1 ) { return v0 >  v1; }

template<class T0,T0 v0,class T1> constexpr auto min( Ct<T0,v0>, T1 v1 ) { using TR = TypePromote<T0,T1>::type; return v0 <= v1 ? TR( v0 ) : TR( v1 ); }
template<class T0,T0 v0,class T1> constexpr auto max( Ct<T0,v0>, T1 v1 ) { using TR = TypePromote<T0,T1>::type; return v0 >= v1 ? TR( v0 ) : TR( v1 ); }

// T0 Ct
template<class T0,class T1,T1 v1> constexpr auto operator+( T0 v0, Ct<T1,v1> ) { return v0 + v1; }
template<class T0,class T1,T1 v1> constexpr auto operator-( T0 v0, Ct<T1,v1> ) { return v0 - v1; }
template<class T0,class T1,T1 v1> constexpr auto operator*( T0 v0, Ct<T1,v1> ) { return v0 * v1; }
template<class T0,class T1,T1 v1> constexpr auto operator/( T0 v0, Ct<T1,v1> ) { return v0 / v1; }
template<class T0,class T1,T1 v1> constexpr auto operator%( T0 v0, Ct<T1,v1> ) { return v0 % v1; }

template<class T0,class T1,T1 v1> constexpr auto operator&&( T0 v0, Ct<T1,v1> ) { return v0 && v1; }
template<class T0,class T1,T1 v1> constexpr auto operator||( T0 v0, Ct<T1,v1> ) { return v0 || v1; }
template<class T0,class T1,T1 v1> constexpr auto operator==( T0 v0, Ct<T1,v1> ) { return v0 == v1; }
template<class T0,class T1,T1 v1> constexpr auto operator!=( T0 v0, Ct<T1,v1> ) { return v0 != v1; }
template<class T0,class T1,T1 v1> constexpr auto operator<=( T0 v0, Ct<T1,v1> ) { return v0 <= v1; }
template<class T0,class T1,T1 v1> constexpr auto operator>=( T0 v0, Ct<T1,v1> ) { return v0 >= v1; }
template<class T0,class T1,T1 v1> constexpr auto operator< ( T0 v0, Ct<T1,v1> ) { return v0 <  v1; }
template<class T0,class T1,T1 v1> constexpr auto operator> ( T0 v0, Ct<T1,v1> ) { return v0 >  v1; }

template<class T0,class T1,T1 v1> constexpr auto min( T0 v0, Ct<T1,v1> ) { using TR = TypePromote<T0,T1>::type; return v0 <= v1 ? TR( v0 ) : TR( v1 ); }
template<class T0,class T1,T1 v1> constexpr auto max( T0 v0, Ct<T1,v1> ) { using TR = TypePromote<T0,T1>::type; return v0 >= v1 ? TR( v0 ) : TR( v1 ); }

} // namespace sdot
