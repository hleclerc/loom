#pragma once

#include "../kernels/make_avaiable.h"
#include "../kernels/transfer_cost.h"
#include "../common_types.h"
#include "../Ct.h"
#include <SYCL/sycl.hpp>
#include <cstdint>

namespace sdot {

/// What went wrong. The KIND is what tells the host how to read the two numbers that come with a
/// record: for a capacity overflow, `id` is the ShapeVar and `value` the count it asked for.
struct ErrorKind {
    static constexpr std::int32_t capacity_overflow = 1;
};

/// Where device code says that something went wrong.
///
/// Nothing here is about shapes: a capacity that did not fit is one kind of error, and whatever
/// comes next (a degenerate cell, a division by zero, ...) records the same way. The host reads
/// the buffer after the call and decides -- for a capacity, reserve more and run again.
///
/// A C++ GLOBAL would be the natural spelling of "accessible everywhere", but there is no such
/// thing on a device: a kernel only reaches memory it was HANDED. So this is a buffer, allocated
/// by the call, and the values that can fail carry this view -- two words each, all pointing at
/// the one buffer.
///
/// Layout: `[ nb_records, kind, id, value, kind, id, value, ... ]`. `nb_records` is bumped
/// atomically, so any number of batch items (or of kernels) may record at once; records past
/// `max_records` are DROPPED -- the host still learns that something failed, and what it does
/// about it converges anyway (it doubles).
template<class View>
struct ErrorBuffer {
    View view;         ///< rank 1, `std::int32_t`
    SI   max_records;

    void record( std::int32_t kind, std::int32_t id, SI value ) const {
        sycl::atomic_ref<
            std::int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::generic_space
        > nb_records( view( 0 ).ref() );

        const std::int32_t num = nb_records.fetch_add( 1 );
        if ( num >= max_records )
            return;

        view( 1 + 3 * num + 0 ).ref() = kind;
        view( 1 + 3 * num + 1 ).ref() = id;
        view( 1 + 3 * num + 2 ).ref() = std::int32_t( value );
    }

    void fill_with( auto &&queue, auto v ) { view.fill_with( FORWARD( queue ), v ); }

    auto transfer_cost( const auto &queue, auto io_category ) const {
        return sdot::transfer_cost( queue, io_category, view );
    }

    auto make_available( auto &&queue, auto io_category, auto &&cont ) const {
        return sdot::make_available( queue, io_category, view, [&]( auto &&kernel_view ) {
            return cont( ErrorBuffer<DECAYED_TYPE_OF( kernel_view )>{ FORWARD( kernel_view ), max_records } );
        } );
    }
};

/// The error buffer of something that cannot fail: recording is a no-op, and nothing crosses into
/// the kernel. A TYPE, so the whole thing compiles away -- as `NoneTensor` does for a tensor that
/// is not there.
struct NoErrorBuffer {
    constexpr void record        ( std::int32_t /*kind*/, std::int32_t /*id*/, SI /*value*/ ) const {}
    constexpr auto transfer_cost ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto make_available( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }
};

template<class View>
auto make_error_buffer( View view, SI max_records ) { return ErrorBuffer<View>{ view, max_records }; }

} // namespace sdot
