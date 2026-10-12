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

UTP HD DTP::TensorView( DataPtr data, Shape shape, Strides strides ) :
        _strides( strides ), _shape( shape ), _data( reinterpret_cast<RawByte *>( data.raw ), data.memory_space ) {
}

UTP HD auto DTP::size( auto axis ) const {
    constexpr int pos = AxisPos<DECAYED_TYPE_OF( axis ),AxisNames>::value;
    static_assert( pos >= 0, "TensorView::size : this tensor has no such axis" );
    return _shape[ Ct<int,pos>() ];
}

UTP HD auto DTP::domain() const {
    return CartesianIndices<Shape,AxisNames>{ _shape };
}

UTP    auto DTP::kernel_form( auto &&queue, auto io_category ) const {
    using KMS = typename DECAYED_TYPE_OF( queue )::DefaultKernelMemorySpace;

    // an argument must be categorized (Inp/Out/Mut); otherwise the user forgot a tag
    static_assert( ! std::is_same_v<DECAYED_TYPE_OF( io_category ), UndefList>,
                   "argument passed to run_parallel without an Inp/Out/Mut category" );

    // zero cost -> data already accessible from the target context -> we retype the Ptr. A device
    // that would have to transfer does not exist yet (see `kernel_form` in make_avaiable.h)
    static_assert( DECAYED_TYPE_OF( transfer_cost_per_byte( queue, _data.memory_space ) )::value == 0,
                   "TensorView::kernel_form : this queue cannot see this memory area, and the transfer is not written" );
    using KTensor = TensorView<TF,Shape,KMS,AxisNames,Strides>;
    return KTensor( typename KTensor::DataPtr( _data.template as<TF>() ), _shape, _strides );
}

UTP HD auto DTP::operator()( const auto &index, auto ...rem ) const {
    using I = DECAYED_TYPE_OF( index );
    if constexpr ( IsCoords<I>::value )
        // NAMED coordinates: we consume the tuple they carry. The bare `Tuple` is still
        // accepted just below -- both work, as agreed.
        return operator()( index.values, rem... );
    else if constexpr ( IsAxisIndex<I>::value )
        // index = (name = value) -> squeeze the named axis, then carry on
        return squeeze( index )( rem... );
    else if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
        // index = multi-index (size known at compile time) -> we unfold its components
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return operator()( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return operator()( rem... );
    } else
        // index = scalar position -> squeeze axis 0
        return row( index )( rem... );
}

// 2 arguments: selector (position `Ct<int,N>` or axis name) + value (`index`, possibly a Ct)
UTP HD auto DTP::squeeze( auto axis, auto index ) const {
    using A = DECAYED_TYPE_OF( axis );
    if constexpr ( is_axis<A> ) {
        // axis = axis name -> we resolve its position then squeeze positionally
        constexpr int pos = AxisPos<A, AxisNames>::value;
        static_assert( pos >= 0, "unknown axis name for this tensor" );
        return squeeze( Ct<int,pos>(), index );
    } else {
        // axis = position: we remove this axis (shape/strides/names) and advance the Ptr
        auto new_shape   = _shape.without_index( axis );
        auto new_strides = _strides.without_index( axis );
        auto new_names   = AxisNames{}.without_index( axis );
        LOOM_CHECK_INDEX( index, _shape[ axis ] );
        SI   off         = _strides[ axis ] * index;         // offset in bytes (Ct or runtime -> SI)
        auto ptr         = DataPtr( ( _data + off ).template as<TF>(), _data.memory_space ); // typed, for the ctor
        using R = TensorView<TF,DECAYED_TYPE_OF( new_shape ),MemorySpace,DECAYED_TYPE_OF( new_names ),DECAYED_TYPE_OF( new_strides )>;
        return R( ptr, new_shape, new_strides );
    }
}

// 1 argument: a named index `dim = i` -> we extract name + value
UTP HD auto DTP::squeeze( auto axis_index ) const {
    using A = DECAYED_TYPE_OF( axis_index );
    static_assert( IsAxisIndex<A>::value, "squeeze with 1 argument expects a named index (name = value)" );
    constexpr int pos = AxisPos<typename A::axis_type, AxisNames>::value;
    if constexpr ( pos < 0 ) {
        // an axis we do not have. A strict index (`dim = 0`) makes this a typo; an OPTIONAL one
        // (a batch index, see AxisNames.h) is meant for whoever carries that axis -- we are not
        // mapped along it, so we simply let it through.
        static_assert( A::optional, "unknown axis name for this tensor" );
        return *this;
    } else
        return squeeze( Ct<int,pos>(), axis_index.index );
}

UTP HD auto DTP::row( auto index ) const {
    return squeeze( Ct<int,0>(), index );
}

UTP HD auto DTP::offset( const auto &index, auto ...rem ) const {
    if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
        // `index` is a multi-index (size known at compile time) -> we unfold its components
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return offset( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return offset( rem... );
    } else {
        // shifts axis 0 by `index` elements (sub-view) and advances the pointer accordingly (strides in bytes)
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

UTP HD void DTP::operator=( const TensorView &that ) {
    copy_elements_from( that );
}

UTP HD void DTP::operator=( const auto &that ) {
    copy_elements_from( that );
}

// UTP template<class... ExtraTags> auto DTP::with_tags() const {
//     // same data, tag pack extended with ExtraTags... (appended verbatim, no axis transform)
//     return TensorView<TF,MemorySpace,Shape,Strides,Tags...,ExtraTags...>( data().raw, _shape, _strides, _memory_space );
// }

UTP HD auto DTP::data() const {
    return DataPtr( _data.template as<TF>(), _data.memory_space );
}

UTP HD TF DTP::value() const {
    static_assert( ct_rank == 0 );
    return data().value();
}

UTP HD TF &DTP::ref() const {
    static_assert( ct_rank == 0 );
    return *data();
}

UTP HD void DTP::for_each_scalar( auto &&func ) const {
    if constexpr ( ct_rank == 0 )
        func( *this );
    else
        for ( TI i = 0; i < TI( shape( Ct<int,0>() ) ); ++i )
            operator[]( i ).for_each_scalar( func );
}

UTP HD auto DTP::nb_items() const {
    return product( _shape );
}

// cost (seconds) to make this view accessible from `queue` = cost/byte * nb bytes
UTP    auto DTP::transfer_cost( const auto &queue, auto /*io_category*/ ) const {
    return transfer_cost_per_byte( queue, memory_space() ) * ( nb_items() * Ct<int,sizeof( TF )>() );
}

// "simple loop" variant: requires the area to be accessible from the host
// (otherwise, pass a tuple of execution contexts -> run_parallel overload below)
UTP    void DTP::fill_with( TF value ) {
    static_assert(
        MemorySpace::directly_accessible,
        "fill_with without context: area not accessible from the host; pass a tuple of execution contexts"
    );
    for_each_scalar( [&]( auto v ) { v.ref() = value; } );
}

// variant with execution contexts: dispatch via run_parallel (choice of the best context)
// Choice of the item_list + the kernel according to the shape. `run` performs the run_parallel call (with or
// without dependencies) -> we never name Dependencies here.
// the bodies of `fill_with` are named FUNCTORS and not lambdas: a lambda defined on the host side cannot
// be the kernel of a device launch (nvcc), whereas a namespace-scope struct can.
namespace detail::TensorViewFill {
    struct Scalar     { template<class I,class Out,class V> HD void operator()( I, Out out, V v ) const { out.ref() = v; } };
    struct Contiguous { template<class I,class Out,class V> HD void operator()( I id, Out out, V v ) const { out._data.template as<typename Out::TF>()[ id ] = v; } };
    struct Strided    { template<class I,class Out,class V> HD void operator()( I id, Out out, V v ) const { out( out.indices_col_ordering( id ) ) = v; } };
}

UTP    auto DTP::_fill_with( TF value, auto &&run ) {
    if constexpr ( ct_rank == 0 )
        return run( range( 1 ), detail::TensorViewFill::Scalar{}, OutList(), *this, InpList(), value );
    else if ( items_are_contiguous() )
        return run( range( nb_items() ), detail::TensorViewFill::Contiguous{}, OutList(), *this, InpList(), value );
    else
        return run( range( nb_items() ), detail::TensorViewFill::Strided{}, OutList(), *this, InpList(), value );
}

UTP    auto DTP::fill_with( auto &&queue_list, TF value ) {
    return _fill_with( value, [&]( auto &&...a ) { return run_parallel( FORWARD( queue_list ), FORWARD( a )... ); } );
}

UTP    auto DTP::fill_with( auto &&queue_list, auto &&deps, TF value ) {
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

// Simple loop primitive (host): applies op( scalar_ref_of_this, scalar_of_that ) on each
// element. `that` of same rank -> element-wise; rank-0 tensor or scalar -> broadcast.
UTP HD void DTP::_zip_apply( auto op, const auto &that ) const {
    static_assert( MemorySpace::directly_accessible,
                   "operation without context: area not accessible from the host; pass a tuple of execution contexts" );
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
                    operator[]( i )._zip_apply( op, that[ i ] );    // same rank -> element-wise
                else
                    operator[]( i )._zip_apply( op, that );         // broadcast (different rank)
            } else
                operator[]( i )._zip_apply( op, that );             // broadcast (scalar)
        }
    }
}

UTP HD void DTP::copy_elements_from( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a  = b; }, that ); }
UTP HD void DTP::operator+=     ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a += b; }, that ); }
UTP HD void DTP::operator-=     ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a -= b; }, that ); }
UTP HD void DTP::operator*=     ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a *= b; }, that ); }
UTP HD void DTP::operator/=     ( const auto &that ) { _zip_apply( []( auto &a, auto b ) { a /= b; }, that ); }

namespace detail {
    HD auto indices_rec( auto index, auto &&res_so_far, auto &&shape ) {
        auto coeff = shape.apply_values( []( auto&&...values ) { return ( 1_c * ... * values ); } );
        auto res = res_so_far.with_appended_value( index / coeff );
        if constexpr ( DECAYED_TYPE_OF( shape )::ct_size )
            return indices_rec( index % coeff, res, shape.without_index( 0_c ) );
        else
            return res;
    };
}

UTP HD auto DTP::indices_col_ordering( auto index ) const {
    return detail::indices_rec( index, tuple(), _shape.without_index( 0_c ) );
}

UTP HD auto DTP::items_are_contiguous() const {
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

UTP HD auto DTP::size() const {
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
