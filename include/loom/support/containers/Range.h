#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI
#include "../Ct.h"

namespace sdot {

template<class TI,class TC=PI>
struct Range {
    T_U  void for_each_item_split( PI rel, PI mod, U &&func ) const { for( TC i = rel; i < TC( end ); i += mod ) func( TI( i ) ); }
    auto      make_available     ( auto &&queue, auto &&io_category, auto &&cont ) const { return cont( *this ); }
    // Une plage ne designe RIEN en memoire : son cout de transfert est nul, quel que soit le type
    // de sa borne. Le dire ici plutot que de compter sur `std::is_trivial_v` dans `transfer_cost`,
    // car une borne compile-time (`Range<Ct<SI,2>>`, ce que rend `range( nb_items() )` sur un
    // tenseur dont tous les extents sont dans le type) n'est PAS triviale -- `Ct` a un
    // constructeur -- et tombait alors sur la branche d'erreur.
    constexpr auto transfer_cost ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    T_U  void for_each_item      ( U &&func ) const { for( TC i = 0; i < TC( end ); ++i ) func( i ); }
    T_U  TC   operator[]         ( U index ) const { return index; } ///< the index-th item (a Range yields its own index)
    TI        size               () const { return end; }

    TI        end;               ///<
};

T_T constexpr auto range( T &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

} // namespace sdot
