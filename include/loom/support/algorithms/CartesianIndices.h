#pragma once

#include "../containers/Tuple.h" // tuple, product, with_appended_value, without_index, apply_values, 1_c
#include "../containers/AxisNames.h" // UnnamedAxis, optional_axis_index, unnamed_axes
#include "../common_macros.h"
#include "min.h"

namespace sdot {

namespace detail {
    // unravel d'un index plat en multi-indice (ordre colonne) pour `shape`
    HD auto unravel_index( auto flat, auto &&res_so_far, auto &&shape ) {
        auto coeff = shape.apply_values( []( auto &&...values ) { return ( 1_c * ... * values ); } );
        auto res   = res_so_far.with_appended_value( flat / coeff );
        if constexpr ( DECAYED_TYPE_OF( shape )::ct_size )
            return unravel_index( flat % coeff, res, shape.without_index( 0_c ) );
        else
            return res;
    }

    // attach its axis name to each coordinate. An unnamed axis keeps a bare index (positional,
    // as `indices_of` produces for a plain shape); a named one becomes `name = coordinate`, and
    // an OPTIONAL one at that: an argument not mapped along that axis lets it through untouched
    // (see AxisNames.h), which is what lets a batched and an unbatched call share one body.
    HD auto attach_axis_names( auto &&raw, auto &&names, auto &&res ) {
        if constexpr ( DECAYED_TYPE_OF( raw )::ct_size == 0 )
            return res;
        else {
            auto value = raw[ 0_c ];
            auto name  = names[ 0_c ];
            auto named = [&] {
                if constexpr ( std::is_same_v<DECAYED_TYPE_OF( name ),UnnamedAxis> )
                    return value;
                else
                    return optional_axis_index( name, value );
            };
            return attach_axis_names( raw.without_index( 0_c ), names.without_index( 0_c ),
                                      res.with_appended_value( named() ) );
        }
    }
}

/// Ensemble des multi-indices d'une forme (cf. `CartesianIndices` en Julia).
/// Utilisée comme item_list de `run_parallel` : le kernel reçoit `item_list[flat]`, c.-à-d. le
/// multi-indice correspondant. Trivialement copiable (ne porte que la forme) -> capturable kernel.
/// Le cas de rang 0 (`CartesianIndices<Tuple<>>`) est légitime et fréquent : un seul item, le
/// multi-indice vide -- « une passe, sans axe de batch » (c'est le `global_batch_indices` par
/// défaut d'un kernel généré ; un `vmap` lui ajoute des axes).
///
/// The axes may be NAMED (`CartesianIndices<Tuple<SI>,Tuple<_vmap_0>>`), and a generated kernel's
/// batch indices are: the multi-index then holds `vmap_0 = i` rather than a bare `i`, so an
/// argument consumes it BY NAME -- `cell.vertex_positions( batch_index, dim = 0 )` -- and one that
/// is not mapped along that axis ignores it. Same body, batched or not: with no `vmap` the
/// multi-index is empty and indexing by it is a no-op.
template<class Shape, class AxisNames = DECAYED_TYPE_OF( unnamed_axes( Shape{} ) )>
struct CartesianIndices {
    HD auto size       () const { return product( shape ); }
    HD auto operator[] ( auto flat ) const {
        if constexpr ( Shape::ct_size == 0 )
            return tuple();
        else {
            auto raw = detail::unravel_index( flat, tuple(), shape.without_index( 0_c ) );
            return detail::attach_axis_names( raw, AxisNames{}, tuple() );
        }
    }
       auto kernel_form   ( auto &&/*queue*/, auto /*io_category*/ ) const { return *this; }

    /// intersection des parcours : min terme à terme des formes (mêmes rangs).
    HD auto intersection  ( const auto &other ) const {
        auto s = shape.apply_values( [&]( auto &&...as ) {
            return other.shape.apply_values( [&]( auto &&...bs ) {
                return tuple( min( as, bs )... );
            } );
        } );
        return CartesianIndices<DECAYED_TYPE_OF( s )>{ s };
    }

    Shape shape;
};

/// concaténation de deux `Tuple` au niveau des TYPES ( pour les noms d'axes d'un domaine composé ).
template<class,class> struct TupleCat;
template<class... A,class... B> struct TupleCat<Tuple<A...>,Tuple<B...>> { using type = Tuple<A...,B...>; };

/// COMPOSER DEUX DOMAINES : `batch_axes + args.suivant.axes()`.
///
/// C'est la piece qui rend la forme generale aussi capable que l'echafaudage : un corps qui lance
/// lui-meme n'ignore plus les axes de batch de l'appel, il les AJOUTE aux siens. Et comme les
/// indices de batch sont optionnels ( voir `AxisNames.h` ), chaque tenseur ne consomme que les
/// coordonnees qu'il a -- un scalaire de rang 0 les ignore toutes. Un seul corps, batche ou non.
template<class S1,class N1,class S2,class N2>
HD auto operator+( const CartesianIndices<S1,N1> &a, const CartesianIndices<S2,N2> &b ) {
    auto shape = a.shape.apply_values( [&]( auto &&...as ) {
        return b.shape.apply_values( [&]( auto &&...bs ) { return tuple( as..., bs... ); } );
    } );
    return CartesianIndices<DECAYED_TYPE_OF( shape ),typename TupleCat<N1,N2>::type>{ shape };
}

/// UN DOMAINE DE LANCEMENT QU'ON CHOISIT : un item par point de la forme donnée.
///
/// C'est ce qui manquait quand `run_parallel` était engendré pour vous : le seul domaine possible
/// était `global_batch_indices`, qui ne se remplit que des axes de `vmap`. Un noyau dont le
/// parallélisme n'est PAS un axe de `vmap` -- une grille cartésienne, parcourue en (j, i) -- devait
/// donc fabriquer un axe de batch plat, matérialiser le rang dans un tampon, et le redécouper en
/// C++. Voir l'historique d'`examples/diffusion`, qui faisait exactement ça.
///
/// Les extents GARDENT LEUR TYPE : un `Ct<SI,N>` reste connu à la compilation, donc le corps peut
/// dérouler dessus. Les items sont des multi-indices NUS ( `item[ 0_c ]`, `item[ 1_c ]` ) -- ce que
/// veut un en-tête écrit à la main, dont les fonctions prennent des entiers.
HD auto indices_over( auto &&...extents ) {
    auto shape = tuple( FORWARD( extents )... );
    return CartesianIndices<DECAYED_TYPE_OF( shape )>{ shape };
}

/// MANQUE ENCORE, et c'est ce qu'il faudra pour composer avec `vmap` : un domaine NOMMÉ ( dont les
/// items portent `y = j, x = i`, donc consommables par nom ) et un moyen de le concaténer avec
/// `global_batch_indices`. Un corps qui lance lui-même ignore aujourd'hui les axes de batch de
/// l'appel, donc il ne se `vmap` pas tout seul.

} // namespace sdot
