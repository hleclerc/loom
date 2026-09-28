#pragma once

#include "../containers/Tuple.h" // tuple, product, with_appended_value, without_index, apply_values, 1_c
#include "../containers/AxisNames.h" // UnnamedAxis, optional_axis_index, unnamed_axes
#include "../containers/Coords.h" // Coords, AxisSet
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

    /// L'ITEM : des coordonnees NOMMEES ( voir `Coords.h` ), pas un tuple nu. Un tenseur accepte les
    /// deux, mais un corps qui veut un entier le demande par nom -- `coords[ y ]` -- au lieu de
    /// compter les positions.
    HD auto operator[] ( auto flat ) const {
        if constexpr ( Shape::ct_size == 0 )
            return coords_of( tuple() );
        else {
            auto raw = detail::unravel_index( flat, tuple(), shape.without_index( 0_c ) );
            return coords_of( detail::attach_axis_names( raw, AxisNames{}, tuple() ) );
        }
    }

    /// les axes de ce domaine, comme un ENSEMBLE ( pour `coords.axes - batch_axes` ).
    static constexpr typename detail::AxesOfTuple<AxisNames>::type axes = {};
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

template<class> struct TupleHead;
template<class H,class... T> struct TupleHead<Tuple<H,T...>> { using type = H; };

/// COMPOSER DEUX DOMAINES : `batch_axes + args.suivant.domain()`.
///
/// C'est la piece qui rend la forme generale aussi capable que l'echafaudage : un corps qui lance
/// lui-meme n'ignore plus les axes de batch de l'appel, il les AJOUTE aux siens. Et comme les
/// indices de batch sont optionnels ( voir `AxisNames.h` ), chaque tenseur ne consomme que les
/// coordonnees qu'il a -- un scalaire de rang 0 les ignore toutes. Un seul corps, batche ou non.
///
/// L'UNION, ET PAS LA CONCATENATION : un tenseur batche porte DEJA l'axe de batch dans ses propres
/// axes, donc `batch_axes + args.suivant.domain()` le nommerait deux fois -- le domaine aurait alors
/// nb_batch fois trop d'items, et chaque element serait ecrit plusieurs fois. Un axe NOMME deja
/// present est donc saute ( on garde l'extent du premier ). Les axes anonymes, eux, ne se
/// dedupliquent pas : deux dimensions sans nom sont deux dimensions.
namespace detail {
    template<class SA,class NA,class SB,class NB>
    HD auto merge_domains( const CartesianIndices<SA,NA> &a, const CartesianIndices<SB,NB> &b ) {
        if constexpr ( SB::ct_size == 0 )
            return a;
        else {
            using Head = typename TupleHead<NB>::type;
            auto reste_shape = b.shape.without_index( Ct<int,0>() );
            CartesianIndices<DECAYED_TYPE_OF( reste_shape ),typename NB::Next> reste{ reste_shape };

            constexpr bool deja = ! std::is_same_v<Head,UnnamedAxis>
                               && axis_in_set<Head,typename AxesOfTuple<NA>::type>;
            if constexpr ( deja )
                return merge_domains( a, reste );
            else {
                auto s = a.shape.with_appended_value( b.shape[ Ct<int,0>() ] );
                CartesianIndices<DECAYED_TYPE_OF( s ),typename TupleCat<NA,Tuple<Head>>::type> plus{ s };
                return merge_domains( plus, reste );
            }
        }
    }
}

template<class S1,class N1,class S2,class N2>
HD auto operator+( const CartesianIndices<S1,N1> &a, const CartesianIndices<S2,N2> &b ) {
    return detail::merge_domains( a, b );
}

/// la difference directement contre un DOMAINE : `coords.axes - batch_axes` sans avoir a ecrire
/// `batch_axes.axes`. C'est la forme qui se lit.
template<class... A,class S,class N>
HD constexpr auto operator-( AxisSet<A...> a, const CartesianIndices<S,N> & ) {
    return a - typename detail::AxesOfTuple<N>::type{};
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
