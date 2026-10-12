#pragma once

#include "../containers/Tuple.h" // tuple, product, with_appended_value, without_index, apply_values, 1_c
#include "../containers/AxisNames.h" // UnnamedAxis, optional_axis_index, unnamed_axes
#include "../containers/Coords.h" // Coords, AxisSet
#include "../common_macros.h"
#include "min.h"

namespace sdot {

namespace detail {
    // unravel a flat index into a multi-index (column order) for `shape`
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

/// Set of the multi-indices of a shape (cf. `CartesianIndices` in Julia).
/// Used as the item_list of `run_parallel`: the kernel receives `item_list[flat]`, i.e. the
/// corresponding multi-index. Trivially copyable (carries only the shape) -> capturable by a kernel.
/// The rank-0 case (`CartesianIndices<Tuple<>>`) is legitimate and frequent: a single item, the
/// empty multi-index -- "one pass, no batch axis" (it is the default `global_batch_indices` of a
/// generated kernel; a `vmap` adds axes to it).
///
/// The axes may be NAMED (`CartesianIndices<Tuple<SI>,Tuple<_vmap_0>>`), and a generated kernel's
/// batch indices are: the multi-index then holds `vmap_0 = i` rather than a bare `i`, so an
/// argument consumes it BY NAME -- `cell.vertex_positions( batch_index, dim = 0 )` -- and one that
/// is not mapped along that axis ignores it. Same body, batched or not: with no `vmap` the
/// multi-index is empty and indexing by it is a no-op.
template<class Shape, class AxisNames = DECAYED_TYPE_OF( unnamed_axes( Shape{} ) )>
struct CartesianIndices {
    HD auto size       () const { return product( shape ); }

    /// THE ITEM: NAMED coordinates ( see `Coords.h` ), not a bare tuple. A tensor accepts
    /// both, but a body that wants an integer asks for it by name -- `coords[ y ]` -- instead of
    /// counting positions.
    HD auto operator[] ( auto flat ) const {
        if constexpr ( Shape::ct_size == 0 )
            return coords_of( tuple() );
        else {
            auto raw = detail::unravel_index( flat, tuple(), shape.without_index( 0_c ) );
            return coords_of( detail::attach_axis_names( raw, AxisNames{}, tuple() ) );
        }
    }

    /// the axes of this domain, as a SET ( for `coords.axes - batch_axes` ).
    static constexpr typename detail::AxesOfTuple<AxisNames>::type axes = {};
       auto kernel_form   ( auto &&/*queue*/, auto /*io_category*/ ) const { return *this; }

    /// intersection of the traversals: term-by-term min of the shapes (same ranks).
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

/// concatenation of two `Tuple`s at the TYPE level ( for the axis names of a composed domain ).
template<class,class> struct TupleCat;
template<class... A,class... B> struct TupleCat<Tuple<A...>,Tuple<B...>> { using type = Tuple<A...,B...>; };

template<class> struct TupleHead;
template<class H,class... T> struct TupleHead<Tuple<H,T...>> { using type = H; };

/// COMPOSING TWO DOMAINS: `batch_axes + args.next.domain()`.
///
/// This is the piece that makes the general form as capable as the scaffolding: a body that launches
/// by itself no longer ignores the call's batch axes, it ADDS them to its own. And since the
/// batch indices are optional ( see `AxisNames.h` ), each tensor only consumes the
/// coordinates it has -- a rank-0 scalar ignores them all. A single body, batched or not.
///
/// THE UNION, NOT THE CONCATENATION: a batched tensor ALREADY carries the batch axis in its own
/// axes, so `batch_axes + args.next.domain()` would name it twice -- the domain would then have
/// nb_batch times too many items, and each element would be written several times. A NAMED axis already
/// present is therefore skipped ( we keep the extent of the first one ). Anonymous axes, on the other hand,
/// are not deduplicated: two unnamed dimensions are two dimensions.
namespace detail {
    template<class SA,class NA,class SB,class NB>
    HD auto merge_domains( const CartesianIndices<SA,NA> &a, const CartesianIndices<SB,NB> &b ) {
        if constexpr ( SB::ct_size == 0 )
            return a;
        else {
            using Head = typename TupleHead<NB>::type;
            auto rest_shape = b.shape.without_index( Ct<int,0>() );
            CartesianIndices<DECAYED_TYPE_OF( rest_shape ),typename NB::Next> rest{ rest_shape };

            constexpr bool already = ! std::is_same_v<Head,UnnamedAxis>
                               && axis_in_set<Head,typename AxesOfTuple<NA>::type>;
            if constexpr ( already )
                return merge_domains( a, rest );
            else {
                auto s = a.shape.with_appended_value( b.shape[ Ct<int,0>() ] );
                CartesianIndices<DECAYED_TYPE_OF( s ),typename TupleCat<NA,Tuple<Head>>::type> extended{ s };
                return merge_domains( extended, rest );
            }
        }
    }
}

template<class S1,class N1,class S2,class N2>
HD auto operator+( const CartesianIndices<S1,N1> &a, const CartesianIndices<S2,N2> &b ) {
    return detail::merge_domains( a, b );
}

/// the difference directly against a DOMAIN: `coords.axes - batch_axes` without having to write
/// `batch_axes.axes`. It is the form that reads well.
template<class... A,class S,class N>
HD constexpr auto operator-( AxisSet<A...> a, const CartesianIndices<S,N> & ) {
    return a - typename detail::AxesOfTuple<N>::type{};
}

/// A LAUNCH DOMAIN OF YOUR CHOOSING: one item per point of the given shape.
///
/// This is what was missing when `run_parallel` was generated for you: the only possible domain
/// was `global_batch_indices`, which only gets filled with `vmap` axes. A kernel whose
/// parallelism is NOT a `vmap` axis -- a Cartesian grid, traversed in (j, i) -- thus had to
/// fabricate a flat batch axis, materialize the rank in a buffer, and slice it back up in
/// C++. See the history of `examples/diffusion`, which did exactly that.
///
/// The extents KEEP THEIR TYPE: a `Ct<SI,N>` stays known at compile time, so the body can
/// unroll over it. The items are BARE multi-indices ( `item[ 0_c ]`, `item[ 1_c ]` ) -- what
/// a hand-written header wants, whose functions take integers.
HD auto indices_over( auto &&...extents ) {
    auto shape = tuple( FORWARD( extents )... );
    return CartesianIndices<DECAYED_TYPE_OF( shape )>{ shape };
}

/// WHAT IS STILL MISSING: CHOSEN extents and NAMED axes at the same time. Here the axes are
/// anonymous, so the items are bare multi-indices -- which does not compose with the call's
/// batch axes, and cannot be consumed by name.
///
/// The case that mattered is already covered from the other end: `TensorView::domain( num_y, num_x )`,
/// a named SUB-DOMAIN of a tensor, which concatenates with `global_batch_indices` and whose
/// coordinates are read by name. It is what `examples/splats` launches. What remains is knowing how to name a
/// domain that is the shadow of no tensor.

} // namespace sdot
