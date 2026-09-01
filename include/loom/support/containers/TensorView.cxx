#pragma once

// #include "../algorithms/apply_values.h"
#include "../kernels/IoCategory.h" // UndefList/InpList/OutList/MutList
#include "../common_macros.h" // IWYU pragma: export  (LOOM_CHECK_INDEX)
#include "TensorView.h"
#include "Range.h"

// #include "CartesianProduct.h"
// #include "Range.h"

#define UTP template<class TF,class Shape,class MemorySpace,class AxisNames,class Strides>
#define DTP TensorView<TF,Shape,MemorySpace,AxisNames,Strides>

namespace sdot {

template <typename T> struct Is_TensorView : std::false_type {};
UTP struct Is_TensorView<DTP> : std::true_type {};

UTP DTP::TensorView( DataPtr data, Shape shape, Strides strides ) :
        _strides( strides ), _shape( shape ), _data( reinterpret_cast<RawByte *>( data.raw ), data.memory_space ) {
}

UTP auto DTP::make_available( auto &&queue, auto io_category, auto &&cont ) const {
    using KMS = typename DECAYED_TYPE_OF( queue )::DefaultKernelMemorySpace;

    // un argument doit être catégorisé (Inp/Out/Mut) ; sinon l'utilisateur a oublié un tag
    static_assert( ! std::is_same_v<DECAYED_TYPE_OF( io_category ), UndefList>,
                   "argument passe a run_parallel sans categorie Inp/Out/Mut" );

    if constexpr ( DECAYED_TYPE_OF( transfer_cost_per_byte( queue, _data.memory_space ) )::value == 0 ) {
        // coût nul -> donnée déjà accessible depuis le contexte cible -> on retype le Ptr
        using KTensor = TensorView<TF,Shape,KMS,AxisNames,Strides>;
        return cont( KTensor( typename KTensor::DataPtr( _data.template as<TF>() ), _shape, _strides ) );
    } else {
        TODO; // chemin avec transfert (alloc USM + copy selon io_category) -> point SYCL
    }
}

UTP auto DTP::operator()( const auto &index, auto ...rem ) const {
    using I = DECAYED_TYPE_OF( index );
    if constexpr ( IsAxisIndex<I>::value )
        // index = (nom = valeur) -> squeeze de l'axe nommé, puis on continue
        return squeeze( index )( rem... );
    else if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
        // index = multi-indice (taille connue à la compilation) -> on déplie ses composantes
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return operator()( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return operator()( rem... );
    } else
        // index = position scalaire -> squeeze de l'axe 0
        return row( index )( rem... );
}

// 2 arguments : sélecteur (position `Ct<int,N>` ou nom d'axe) + valeur (`index`, possiblement un Ct)
UTP auto DTP::squeeze( auto axis, auto index ) const {
    using A = DECAYED_TYPE_OF( axis );
    if constexpr ( is_axis<A> ) {
        // axis = nom d'axe -> on résout sa position puis on squeeze positionnellement
        constexpr int pos = AxisPos<A, AxisNames>::value;
        static_assert( pos >= 0, "nom d'axe inconnu pour ce tenseur" );
        return squeeze( Ct<int,pos>(), index );
    } else {
        // axis = position : on retire cet axe (shape/strides/noms) et on avance le Ptr
        auto new_shape   = _shape.without_index( axis );
        auto new_strides = _strides.without_index( axis );
        auto new_names   = AxisNames{}.without_index( axis );
        LOOM_CHECK_INDEX( index, _shape[ axis ] );
        SI   off         = _strides[ axis ] * index;         // offset en octets (Ct ou runtime -> SI)
        auto ptr         = DataPtr( ( _data + off ).template as<TF>(), _data.memory_space ); // typé, pour le ctor
        using R = TensorView<TF,DECAYED_TYPE_OF( new_shape ),MemorySpace,DECAYED_TYPE_OF( new_names ),DECAYED_TYPE_OF( new_strides )>;
        return R( ptr, new_shape, new_strides );
    }
}

// 1 argument : un indice nommé `dim = i` -> on en extrait nom + valeur
UTP auto DTP::squeeze( auto axis_index ) const {
    using A = DECAYED_TYPE_OF( axis_index );
    static_assert( IsAxisIndex<A>::value, "squeeze a 1 argument attend un indice nomme (nom = valeur)" );
    constexpr int pos = AxisPos<typename A::axis_type, AxisNames>::value;
    if constexpr ( pos < 0 ) {
        // an axis we do not have. A strict index (`dim = 0`) makes this a typo; an OPTIONAL one
        // (a batch index, see AxisNames.h) is meant for whoever carries that axis -- we are not
        // mapped along it, so we simply let it through.
        static_assert( A::optional, "nom d'axe inconnu pour ce tenseur" );
        return *this;
    } else
        return squeeze( Ct<int,pos>(), axis_index.index );
}

UTP auto DTP::row( auto index ) const {
    return squeeze( Ct<int,0>(), index );
}

UTP auto DTP::offset( const auto &index, auto ...rem ) const {
    if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
        // `index` est un multi-indice (taille connue à la compilation) -> on déplie ses composantes
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return offset( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return offset( rem... );
    } else {
        // décale l'axe 0 de `index` éléments (sous-vue) et avance le pointeur d'autant (strides en octets)
        TensorView res = *this;
        res._shape.set( 0_c, res._shape[ 0_c ] - index );
        res._data.raw += _strides[ 0_c ] * index;
        return res.offset( rem... );
    }
}



// // `index` is a full multi-index, so a[index] / b[index] are rank-0 views.
// struct AddTensorItemElementwise {              ///< same-shape operand: a[i] += b[i]
//     T_TAB void operator()( T index, A a, B b ) const { a[ index ] += b[ index ]; }
// };
// struct AddTensorItemBroadcast {                ///< scalar / rank-0 operand: a[i] += b
//     T_TAB void operator()( T index, A a, B b ) const { a[ index ] += b; }
// };

// /// true iff `B` is a tensor operand with the same rank as the destination (-> element-wise);
// /// anything else (a scalar, or a lower-rank view) is broadcast.
// template<class B,int dst_rank>
// constexpr bool add_is_elementwise() {
//     if constexpr ( HAS_CT_RANK( B ) )
//         return int( B::ct_rank ) == dst_rank;
//     else
//         return false;
// }

// UTP T_T void DTP::operator+=( const T &that ) {
//     if constexpr ( ct_rank == 0 ) {
//         // base case: accumulate in place (also what stops the per-item recursion above)
//         if constexpr ( DECAYED_TYPE_OF( accessible_from( current_execution_context(), *this ) )::value ) {
//             ref() += TF( that );
//         } else {
//             TF cur = value();
//             cur += TF( that );
//             operator=( cur );
//         }
//     } else if constexpr ( add_is_elementwise<DECAYED_TYPE_OF( that ),ct_rank>() ) {
//         run_parallel( cartesian_product_ranges( _shape ), AddTensorItemElementwise(), inout( *this ), that );
//     } else {
//         run_parallel( cartesian_product_ranges( _shape ), AddTensorItemBroadcast(), inout( *this ), that );
//     }
// }

// UTP T_T void DTP::operator-=( const T &that ) {
//     if constexpr ( ct_rank == 0 ) {
//         if constexpr ( DECAYED_TYPE_OF( accessible_from( current_execution_context(), *this ) )::value ) {
//             ref() -= TF( that );
//         } else {
//             TF cur = value();
//             cur -= TF( that );
//             operator=( cur );
//         }
//     } else if constexpr ( add_is_elementwise<DECAYED_TYPE_OF( that ),ct_rank>() ) {
//         TODO; // run_parallel( cartesian_product_ranges( _shape ), AddTensorItemElementwise(), inout( *this ), that );
//     } else {
//         TODO; // run_parallel( cartesian_product_ranges( _shape ), AddTensorItemBroadcast(), inout( *this ), that );
//     }
// }

// UTP T_T void DTP::operator*=( const T & ) {
//     TODO;
// }

// UTP T_T void DTP::operator/=( const T & ) {
//     TODO;
// }

UTP void DTP::operator=( const TensorView &that ) {
    copy_elements_from( that );
}

UTP void DTP::operator=( const auto &that ) {
    copy_elements_from( that );
}

// UTP template<class... ExtraTags> auto DTP::with_tags() const {
//     // same data, tag pack extended with ExtraTags... (appended verbatim, no axis transform)
//     return TensorView<TF,MemorySpace,Shape,Strides,Tags...,ExtraTags...>( data().raw, _shape, _strides, _memory_space );
// }

UTP auto DTP::data() const {
    return DataPtr( _data.template as<TF>(), _data.memory_space );
}

UTP TF DTP::value() const {
    static_assert( ct_rank == 0 );
    return data().value();
}

UTP TF &DTP::ref() const {
    static_assert( ct_rank == 0 );
    return *data();
}

UTP void DTP::for_each_scalar( auto &&func ) const {
    if constexpr ( ct_rank == 0 )
        func( *this );
    else
        for ( TI i = 0; i < TI( shape( Ct<int,0>() ) ); ++i )
            operator[]( i ).for_each_scalar( func );
}

UTP auto DTP::nb_items() const {
    return product( _shape );
}

// coût (secondes) pour rendre cette vue accessible depuis `queue` = coût/octet * nb octets
UTP auto DTP::transfer_cost( const auto &queue, auto /*io_category*/ ) const {
    return transfer_cost_per_byte( queue, memory_space() ) * ( nb_items() * Ct<int,sizeof( TF )>() );
}

// variante « boucle simple » : nécessite que la zone soit accessible depuis l'hôte
// (sinon, passer un tuple de contextes d'exécution -> surcharge run_parallel ci-dessous)
UTP void DTP::fill_with( TF value ) {
    static_assert(
        MemorySpace::directly_accessible,
        "fill_with sans contexte : zone non accessible depuis l'hote ; passez un tuple de contextes d'execution"
    );
    for_each_scalar( [&]( auto v ) { v.ref() = value; } );
}

// variante avec contextes d'exécution : dispatch via run_parallel (choix du meilleur contexte)
// Choix de l'item_list + du kernel selon la forme. `run` effectue l'appel run_parallel (avec ou
// sans dépendances) -> on ne nomme jamais SYCL/Dependencies ici (TensorView reste sans SYCL).
UTP auto DTP::_fill_with( TF value, auto &&run ) {
    if constexpr ( ct_rank == 0 )
        return run( range( 1 ), []( auto, auto out, auto v ) { out.ref() = v; },
                    OutList(), *this, InpList(), value );
    else if ( items_are_contiguous() )
        return run( range( nb_items() ), []( auto id, auto out, auto v ) { out._data.template as<TF>()[ id ] = v; },
                    OutList(), *this, InpList(), value );
    else
        return run( range( nb_items() ), []( auto id, auto out, auto v ) { out( out.indices_col_ordering( id ) ) = v; },
                    OutList(), *this, InpList(), value );
}

UTP auto DTP::fill_with( auto &&queue_list, TF value ) {
    return _fill_with( value, [&]( auto &&...a ) { return run_parallel( FORWARD( queue_list ), FORWARD( a )... ); } );
}

UTP auto DTP::fill_with( auto &&queue_list, auto &&deps, TF value ) {
    return _fill_with( value, [&]( auto &&...a ) { return run_parallel( FORWARD( queue_list ), FORWARD( deps ), FORWARD( a )... ); } );
}

// UTP void DTP::display( std::ostream &os ) const {
//     if constexpr ( DECAYED_TYPE_OF( transfer_cost( ExecutionContext_Cpu{} ) )::value ) {
//         make_accessible( ExecutionContext_Cpu{}, *this, 1_b, 0_b, [&]( auto &&tensor ) {
//             tensor.display( os );
//         } );
//     } else if constexpr ( ct_rank == 0 ) {
//         os << value();
//     } else if constexpr ( ct_rank == 1 ) {
//         for( TI i = 0; i < shape( 0_c ); ++i )
//             sdot::display( os << ( i ? ", " : "" ), operator[]( i ) );
//     } else {
//         for( std::size_t i = 0; i < shape( 0_c ); ++i )
//             sdot::display( os << "\n  ", operator[]( i ) );
//     }
// }

// UTP auto DTP::nb_items() const {
//     return product( _shape );
// }

// Primitive boucle simple (hôte) : applique op( ref_scalaire_de_this, scalaire_de_that ) sur chaque
// élément. `that` de même rang -> élémentaire ; tenseur rang 0 ou scalaire -> broadcast.
UTP void DTP::_zip_apply( auto op, const auto &that ) const {
    static_assert( MemorySpace::directly_accessible,
                   "operation sans contexte : zone non accessible depuis l'hote ; passez un tuple de contextes d'execution" );
    using That = DECAYED_TYPE_OF( that );
    if constexpr ( ct_rank == 0 ) {
        if constexpr ( Is_TensorView<That>::value )
            op( ref(), that.value() );
        else
            op( ref(), that );
    } else {
        for ( TI i = 0; i < TI( shape( Ct<int,0>() ) ); ++i ) {
            if constexpr ( Is_TensorView<That>::value ) {
                if constexpr ( int( That::ct_rank ) == int( ct_rank ) )
                    operator[]( i )._zip_apply( op, that[ i ] );    // même rang -> élémentaire
                else
                    operator[]( i )._zip_apply( op, that );         // broadcast (rang différent)
            } else
                operator[]( i )._zip_apply( op, that );             // broadcast (scalaire)
        }
    }
}

UTP void DTP::copy_elements_from( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a  = b; }, that ); }
UTP void DTP::operator+=        ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a += b; }, that ); }
UTP void DTP::operator-=        ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a -= b; }, that ); }
UTP void DTP::operator*=        ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a *= b; }, that ); }
UTP void DTP::operator/=        ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a /= b; }, that ); }

namespace detail {
    auto indices_rec( auto index, auto &&res_so_far, auto &&shape ) {
        auto coeff = shape.apply_values( []( auto&&...values ) { return ( 1_c * ... * values ); } );
        auto res = res_so_far.with_appended_value( index / coeff );
        if constexpr ( DECAYED_TYPE_OF( shape )::ct_size )
            return indices_rec( index % coeff, res, shape.without_index( 0_c ) );
        else
            return res;
    };
}

UTP auto DTP::indices_col_ordering( auto index ) const {
    return detail::indices_rec( index, tuple(), _shape.without_index( 0_c ) );
}

UTP auto DTP::items_are_contiguous() const {
    // TODO: sort items
    return _strides == contiguous_strides<TF>( _shape );
}

// UTP T_T void DTP::for_each_index( T &&func ) const {
//     cartesian_product( map( _shape, range<PI> ) ).for_each_item( FORWARD( func ) );
// }

// UTP T_T void DTP::for_each_item( T &&func ) const {
//     for_each_index( [&]( auto &index ) {
//         func( operator()( index ) );
//     } );
// }

UTP auto DTP::size() const {
    static_assert( ct_rank == 1, "..." );
    return shape( Ct<int,0>() );
}

// UTP auto DTP::empty() const {
//     if constexpr ( ct_rank == 0 )
//         return 0_b;
//     return _shape.apply_values( [&]( auto ...values ) {
//         return ( ( values == 0_c ) || ... || 0_b );
//     } );
// }

// namespace details::TensorView {
//     // Namespace-scope functors for arch-aware element-wise ops.
//     // Lambda bodies inside HD/GD template methods from .cxx files cause issues with
//     // some nvcc versions when the lambda references class-level template params (TF).
//     // Using concrete struct operator() avoids the problem.
//     template<class DstTV, class SrcTV, class BI>
//     struct TensorCopyFunctor {
//         DstTV dst;
//         SrcTV src;
//         GD void operator()( BI bi ) const {
//             dst( bi ).item() = src( bi ).item();
//         }
//     };

//     struct TensorFillFunctor {
//         T_TAB GD void operator()( T index, A dst, B value ) const {
//             dst( index ) = value;
//         }
//     };
// } // namespace details::TensorView

// UTP auto DTP::all_indices() const {
//     return cartesian_product_ranges( _shape );
// }

// UTP void DTP::fill_with( TF value ) {
//     run_parallel( all_indices(), details::TensorView::TensorFillFunctor(), Out(), *this, value );
// }

// // transfer_cost for TensorView: accessible without transfer → cost 0, else 1
// UTP T_T auto DTP::transfer_cost( const T &ec ) const {
//     return sdot::transfer_cost_per_byte( ec, _memory_space );
// }

// // UTP CPU_ONLY void DTP::get_data_from( const auto &that ) {
// //     // contiguous -> copy works for all the cases
// //     if ( is_contiguous() && that.is_contiguous() ) {
// //         copy( data(), that.data(), nb_items() );
// //         return;
// //     }

// //     // same memory space -> for each on indices
// //     run_sequential( _shape.all_indices(), details::TensorView::TensorCopyFunctor( *this, that ) );

// //     // else
// //     TODO;
// // }


// // UTP void DTP::with_same_shape( const auto &arch, auto &&func ) const {
// //     arch.template with_reservation<TF>( nb_items(), [&]( auto buf ) {
// //         auto new_strides = contiguous_strides<TF>( _shape );
// //         using NewStrides = DECAYED_TYPE_OF( new_strides );
// //         using BufMS      = typename DECAYED_TYPE_OF( buf )::MemorySpace;
// //         TensorView<TF,Shape,NewStrides,BufMS> res( buf.raw, _shape, new_strides, buf.memory_space );
// //         func( res );
// //     } );
// // }

// // UTP void DTP::spill_to( TensorView &that ) {
// //     that.get_data_from( *this );
// //     _raw_ptr = that._raw_ptr;
// // }


// UTP Strides DTP::strides() const {
//     return _strides;
// }

// UTP T_T auto DTP::stride( T d ) const {
//     return _strides[ d ];
// }

// // UTP PI DTP::nb_items() const {
// //     PI res = 1;
// //     for( PI d = 0; d < rank(); ++d )
// //         res *= shape( d );
// //     return res;
// // }


// UTP bool DTP::surely_null() const {
//     return false; // TensorView is always a real, non-null tensor in FFI bindings
//     // if ( is_invalid() )
//     //     return true;

//     // /* Version using lambdas and Ct<> (causes nvcc to crash in some cases)
//     // // empty tensor (any dimension == 0)
//     // if ( _shape.has_value( []( auto size ) -> bool { return size < Ct<int,1>(); } ) )
//     //     return true;
//     // // all strides zero (rank > 0) → surely-null by construction: all elements alias data()[0] == 0
//     // if ( rank() > 0 && ! _strides.has_value( []( auto size ) -> bool { return size != Ct<int,0>(); } ) )
//     //     return true;
//     // // single scalar: check value
//     // if ( ! _shape.has_value( []( auto size ) -> bool { return size > Ct<int,1>(); } ) )
//     //     return *data() == 0;
//     // */

//     // // empty tensor (any dimension == 0)
//     // for ( PI i = 0; i < ct_rank; ++i )
//     //     if ( _shape[ i ] == 0 )
//     //         return true;

//     // // all strides zero (rank > 0) → surely-null if *data()
//     // bool all_strides_zero = true;
//     // for ( PI i = 0; i < ct_rank; ++i ) {
//     //     if ( _strides[ i ] && _shape[ i ] > 1 ) {
//     //         all_strides_zero = false;
//     //         break;
//     //     }
//     // }
//     // if ( all_strides_zero )
//     //     return *data() == 0;

//     // return false;
// }

// UTP bool DTP::is_invalid() const {
//     return _raw_ptr == _sentinel();
// }

// UTP bool DTP::is_valid() const {
//     return _raw_ptr != _sentinel();
// }

// // UTP auto DTP::rank() const {
// //     return Ct<int,ct_rank>();
// // }


UTP auto DTP::begin() const {
    // `data()` is a `Ptr<TF>` advancing by ELEMENTS, so the iterator stride must be in elements:
    // byte stride / sizeof(TF).
    if constexpr ( ct_rank == 0 ) {
        return StridedIterator<TF, MemorySpace>( data(), 1 );
    } else if constexpr ( ct_rank == 1 ) {
        return StridedIterator<TF, MemorySpace>( data(), _strides[ 0_c ] / SI( sizeof( TF ) ) );
    } else {
        TODO; // multi-dim: need nested/cartesian product iterator for row-major traversal
    }
}

UTP auto DTP::end() const {
    if constexpr ( ct_rank == 0 ) {
        return StridedIterator<TF, MemorySpace>( data() + 1, 1 );
    } else if constexpr ( ct_rank == 1 ) {
        const SI elem_stride = _strides[ 0_c ] / SI( sizeof( TF ) );
        return StridedIterator<TF, MemorySpace>( data() + size() * elem_stride, elem_stride );
    } else {
        TODO; // multi-dim: need nested/cartesian product iterator for row-major traversal
    }
}


// // UTP auto DTP::unsqueeze( auto axis ) const {
// //     // Append a dimension of size 1.
// //     // The stride for the new axis is sizeof(T): matches contiguous layout when the source is contiguous.
// //     // For a size-1 axis the stride value is irrelevant for correctness, but we set it consistently.
// //     // constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank + 1 : -1;
// //     // return TensorView<T,new_ct_rank,Arch>( data(), _shape.with_pushed_value( PI( 1 ) ), _strides.with_pushed_value( SI( sizeof( T ) ) ) );
// //     TODO;
// // }


// // UTP void DTP::for_each_index( auto &&func ) const {
// //     IndexRange<Shape>{ _shape }.for_each_item( FORWARD( func ) );
// // }

// // // #ifdef __CUDACC__
// // // // Functor at namespace scope so CUDA/cudafe can properly mangle the type in Thrust typedefs
// // // // (local structs with __device__ members cause "template argument N is invalid" in cudafe)
// // // template<class T, int ct_rank>
// // // struct TensorViewIndexer {
// // //     __device__ const T &operator()( sdot::PI i ) const {
// // //         sdot::SI off = 0;
// // //         for ( int d = ct_rank - 1; d >= 0; --d ) {
// // //             off += sdot::SI( i % ext[ d ] ) * str[ d ];
// // //             i /= ext[ d ];
// // //         }
// // //         return *reinterpret_cast<const T *>( src + off );
// // //     }

// // //     const std::byte *src;
// // //     sdot::PI ext[ ct_rank ];
// // //     sdot::SI str[ ct_rank ];
// // // };


// // NB: the cross-space transfer for TensorView used to live here as a make_accessible overload, but it
// // made make_accessible (a fully generic policy over scalars/aggregates/views) leak a TensorView-specific
// // overload. The accessible case (func(value)) is already handled generically in hardware/make_accessible.h.
// // When real strided cross-space transfer is implemented, expose it as a customization point on the data
// // type that the generic else-branch delegates to — not as a competing overload.

#undef UTP
#undef DTP

} // namespace sdot
