#pragma once

#include "../containers/Tuple.h" // tuple, product, with_appended_value, without_index, apply_values, 1_c
#include "../containers/AxisNames.h" // UnnamedAxis, optional_axis_index, unnamed_axes
#include "../common_macros.h"
#include "min.h"

namespace sdot {

namespace detail {
    // unravel d'un index plat en multi-indice (ordre colonne) pour `shape`
    auto unravel_index( auto flat, auto &&res_so_far, auto &&shape ) {
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
    auto attach_axis_names( auto &&raw, auto &&names, auto &&res ) {
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
    auto size          () const { return product( shape ); }
    auto operator[]    ( auto flat ) const {
        if constexpr ( Shape::ct_size == 0 )
            return tuple();
        else {
            auto raw = detail::unravel_index( flat, tuple(), shape.without_index( 0_c ) );
            return detail::attach_axis_names( raw, AxisNames{}, tuple() );
        }
    }
    auto make_available( auto &&/*queue*/, auto &&/*io_category*/, auto &&cont ) const { return cont( *this ); }

    /// intersection des parcours : min terme à terme des formes (mêmes rangs).
    auto intersection  ( const auto &other ) const {
        auto s = shape.apply_values( [&]( auto &&...as ) {
            return other.shape.apply_values( [&]( auto &&...bs ) {
                return tuple( min( as, bs )... );
            } );
        } );
        return CartesianIndices<DECAYED_TYPE_OF( s )>{ s };
    }

    Shape shape;
};

} // namespace sdot
