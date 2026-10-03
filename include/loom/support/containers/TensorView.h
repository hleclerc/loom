#pragma once

#include <loom/support/common_macros.h> // HD

// #include "../hardware/Ptr.h"

#include "internal/contiguous_strides.h" // IWYU pragma: export
#include "../kernels/CpuHostMemorySpace.h"
#include "../kernels/Ptr.h" // IWYU pragma: export
#include "AxisNames.h" // IWYU pragma: export
#include "../algorithms/CartesianIndices.h" // IWYU pragma: export  (axes())
#include "TupleRep.h" // IWYU pragma: export
#include "StridedIterator.h" // IWYU pragma: export
#include <type_traits>
// #include "container_tags.h" // IWYU pragma: export
// #include "AxisNames.h" // IWYU pragma: export
// #include "Vector.h" // IWYU pragma: export

namespace sdot {


/// view on strided data (strides in bytes, handles non-contiguous arrays)
///
///
///   TF -> the scalar type
///   Shape ->  size for each axis
///   AxisNames -> name of each axis (UnnamedAxis if not named), see AxisNames.h / DEFINE_AXIS
template<
    class _TF,
    class _Shape,
    class _MemorySpace,
    class _AxisNames = DECAYED_TYPE_OF( unnamed_axes( _Shape{} ) ),
    class _Strides = DECAYED_TYPE_OF( contiguous_strides<_TF>( _Shape{} ) )
>
class TensorView {
public:
    using            MemorySpace            = _MemorySpace;
    using            AxisNames              = _AxisNames;
    using            Strides                = _Strides;
    using            Shape                  = _Shape;
    using            TF                     = _TF;
    using            TI                     = SI;

    using            value_type             = TF;

    using            RawByte                = std::conditional_t<std::is_const_v<TF>,const std::byte,std::byte>; ///< byte, const or not depending on the constness of TF (strides are in bytes)
    using            BytePtr                = Ptr<RawByte,MemorySpace>; ///< byte pointer + memory space (what we store)
    using            DataPtr                = Ptr<TF,MemorySpace>;      ///< typed pointer returned by data()
    SCInt            ct_rank                = Shape::ct_size;

    /* */            HD TensorView          ( DataPtr data, Shape shape, Strides strides ); ///< typed pointer; stored internally as a BytePtr (strides in bytes)
    /* */               TensorView          ( const TensorView & ) = default; ///< Eigen-like view semantics: copy-construction shares the data (shallow), while operator= copies the elements (deep). The defaulted copy-ctor also silences -Wdeprecated-copy.
    /* */               TensorView          () = default;

    // generic info. A view HAS data, and says so in its type: the answer is a `Ct`, known at
    // compile time, so a kernel branches with `if constexpr` and never tests a pointer. The
    // storageless cases are distinct TYPES (see NoneTensor.h -- unbound, and ZeroTensor.h --
    // symbolically zero), not a TensorView in a degenerate state.
    static constexpr bool is_valid = true;
    static constexpr bool surely_null = false;

    HD MemorySpace   memory_space           () const { return _data.memory_space; }
    // (no display member: the generic display() from display.h handles TensorView via shape()/value()/operator[])

    //
    HD Strides       strides                () const;
    HD auto          stride                 ( auto d ) const;
    HD auto          rank                   () const;

    // shape
    // for_each_index / for_each_item : to be migrated (depend on cartesian_product/range)
    HD auto          indices_col_ordering   ( auto index ) const;
    HD auto          items_are_contiguous   () const; ///<
    HD void          for_each_scalar        ( auto &&func ) const; ///< calls func( rank_0_view ) for each element (simple recursive loop)
    HD auto          all_indices            () const;
    HD auto          nb_items               () const;
    HD auto          shape                  ( auto d ) const { return _shape[ d ]; }
    HD Shape         shape                  () const { return _shape; }
    HD auto          empty                  () const;
    HD auto          size                   () const;

    // content
    HD auto          data                   () const;

    HD auto          begin                  () const;
    HD auto          end                    () const;

    // operator() and operator[] produce a new tensor
    HD auto          operator()             ( const auto &index, auto ...rem ) const;
    HD auto          operator[]             ( const auto &index ) const { return operator()( index ); }
    HD auto          operator()             () const { return *this; }

    HD auto          offset                 ( const auto &index, auto ...rem ) const;
    HD auto          offset                 () const { return *this; }

    // scalar value/reference for a rank 1 tensor
    /* */            HD operator TF        () const { return value(); }
    HD TF            value                 () const;
    HD TF&           ref                   () const;

    // reassign
    HD void          _zip_apply             ( auto op, const auto &that ) const; ///< op( scalar_ref, scalar_of_that ) on each element (same rank -> elementwise; rank 0/scalar -> broadcast)
    HD void          copy_elements_from     ( const auto &that );
    HD void          operator-=             ( const auto &that );
    HD void          operator+=             ( const auto &that );
    HD void          operator*=             ( const auto &that );
    HD void          operator/=             ( const auto &that );
    HD void          operator=              ( const auto &that );
    HD void          operator=              ( const TensorView &that );
    HD void          spill_to               ( TensorView &that ); ///< copy data of *this to that, and use data from that

    // data copy / transfer
       auto          transfer_cost          ( const auto &queue, auto io_category ) const;

    // makes the view accessible from the execution context `queue`: if the transfer cost
    // is zero we simply retype the Ptr towards the target kernel space, otherwise we transfer
    // (alloc + copy according to io_category). Then calls cont( kernel_view ).
       auto          kernel_form            ( auto &&queue, auto io_category ) const;

    /// OUR DOMAIN: one item per element, each coordinate carrying the name of its axis. This is
    /// what we pass to `run_parallel`, and it is well defined -- unlike the axes of an AGGREGATE,
    /// which often has some that we do not iterate over ( `Splats` has `splat`, but also `rgb` ).
    ///
    /// It carries the EXTENTS, which are runtime data: that is why this is a
    /// method, whereas `axes` below is a compile-time constant.
    HD auto          domain                 () const;

    /// A SUB-DOMAIN, by NAME: `image.domain( num_y, num_x )` -- the pixels, not the channels.
    ///
    /// This is what was missing for a body that itself launches to stay tensorial. `domain()` takes
    /// ALL the axes, which is only the right traversal when the tensor has nothing but axes to
    /// iterate over; an image `( y, x, rgb )` is not in that case -- one item per channel would
    /// make it recompute the same gaussian three times. The workaround was to build a flat rank and
    /// to split it back with `/` and `%`, that is, to lose the axes at the very moment we iterate
    /// over them.
    ///
    /// The extents KEEP THEIR TYPE ( `size( axis )` returns a `Ct<SI,N>` when it is one ), so
    /// an extent known at compile time stays so in the domain.
    template<class A,class... B> requires ( is_axis<A> )
    HD auto domain( A, B... ) const {
        auto shape = tuple( size( A{} ), size( B{} )... );
        return CartesianIndices<DECAYED_TYPE_OF( shape ),Tuple<A,B...>>{ shape };
    }

    /// THE SAME, FROM A SET: `image.domain( image.axes - num_channel )`.
    ///
    /// This is the form a body writes when it does NOT know its dimension: it only names the
    /// axis to remove, and the others -- two in 2D, three in 3D -- follow without being spelled out.
    template<class... A>
    HD auto domain( AxisSet<A...> ) const {
        return domain( A{}... );
    }

    /// OUR AXES, as a set ( see `Coords.h` ): something to subtract from and iterate over, without
    /// extents. Purely types, hence free -- and the same name as `coords.axes`.
    static constexpr typename detail::AxesOfTuple<_AxisNames>::type axes = {};

    /// our size along a NAMED axis ( `args.next.size( y )` ).
    HD auto          size                   ( auto axis ) const;

       auto          fill_with              ( auto &&queue_list, auto &&deps, TF value ); ///< with dependencies (after(...)) -> QueueEvent
       auto          fill_with              ( auto &&queue_list, TF value );              ///< -> QueueEvent (RAII: synchronous by default, async if managed)
       void          fill_with              ( TF value );                                 ///< simple host-side loop (guarded by directly_accessible)
       auto          _fill_with             ( TF value, auto &&run );                     ///< shared impl: item_list/kernel choice; `run` = run_parallel call (with or without deps)

    //
    HD auto          unsqueeze              ( auto axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
    HD auto          squeeze                ( auto axis, auto index ) const; ///< axis = position (Ct) or axis name, + value
    HD auto          squeeze                ( auto axis_index ) const;       ///< axis_index = (name = value)
    HD auto          row                    ( auto index ) const;

    Strides          _strides;              ///< strides in bytes
    Shape            _shape;                ///<
    BytePtr          _data;                 ///< byte pointer + memory space, aggregated
};

// View on data whose address we know. The memory space is a TYPE parameter (it is
// part of the view's type, as of the Ptr's): host RAM by default, but the kernel
// generated by a `driver.call` on GPU builds it on `CudaGlobalMemorySpace`, since that is where
// XLA hands it its buffers.
template<class MemorySpace = CpuHostMemorySpace>
HD auto tensor_view( auto *ptr, auto &&shape, auto &&axis_names, auto &&strides ) {
    using TF = DECAYED_TYPE_OF( *ptr );
    return TensorView<TF,DECAYED_TYPE_OF( shape ),MemorySpace,DECAYED_TYPE_OF( axis_names ),DECAYED_TYPE_OF( strides )>(
        Ptr<TF,MemorySpace>( ptr ), FORWARD( shape ), FORWARD( strides )
    );
}
template<class MemorySpace = CpuHostMemorySpace>
HD auto tensor_view( auto *ptr, auto &&shape, auto &&axis_names ) { return tensor_view<MemorySpace>( ptr, shape, FORWARD( axis_names ), contiguous_strides<DECAYED_TYPE_OF( *ptr )>( shape ) ); }
template<class MemorySpace = CpuHostMemorySpace>
HD auto tensor_view( auto *ptr, auto &&shape ) { return tensor_view<MemorySpace>( ptr, FORWARD( shape ), unnamed_axes( shape ) ); }



// #undef SDOT_DATA_ACCESSOR

} // namespace sdot

#include "TensorView.cxx" // IWYU pragma: export
