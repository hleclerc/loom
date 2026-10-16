#pragma once

#include "../common_types.h"
#include <algorithm>
#include <initializer_list>
#include <cstdint>
#include <vector>

namespace sdot {

/// A BUFFER OF A `loom.scratch` ARGUMENT, taken by the generated handler from the call's allocator ( `Scratch.h` ) -- not an FFI
/// result that Python allocated before the call.
///
/// Why: a scratch is the call's own business ( the cells of a sweep, a sort buffer ): nothing reads it once the call returns. As
/// an FFI result it was allocated by the framework at a capacity Python guessed, kept alive by the write-back, kept again as a
/// residual for the adjoint -- and, above all, it could not GROW inside the call: a count that did not fit sent the whole call
/// back to Python, which cannot run it again under a `jit`. Taken here, a scratch that is too small is replaced by a larger one
/// and the body runs again, in the same call ( `grow_scratch_capacities` ), under a `jit` as eagerly.
///
/// Spelled like an FFI result buffer ( `typed_data()`, `dimensions()`, `->` ), so the views of the generated source are built
/// on it exactly as on the buffers XLA hands over.
template<class T,int R>
struct ScratchBuffer {
    T           *data = nullptr;
    std::int64_t dims[ R > 0 ? R : 1 ] = {};

    T                   *typed_data () const { return data; }
    const std::int64_t  *dimensions () const { return dims; }
    ScratchBuffer       *operator-> ()       { return this; }
    const ScratchBuffer *operator-> () const { return this; }
};

/// the buffer of `dims` elements from `alloc`, after giving back what `prev` held ( a retry: the call does not hold both )
template<class T,int R>
ScratchBuffer<T,R> scratch_buffer( auto &alloc, ScratchBuffer<T,R> &prev, std::initializer_list<std::int64_t> dims ) {
    alloc.release( prev.data );
    prev.data = nullptr;
    ScratchBuffer<T,R> res;
    SI n = 1;
    int d = 0;
    for ( std::int64_t e : dims ) {
        if ( d < R )
            res.dims[ d++ ] = e;
        n *= SI( e );
    }
    res.data = alloc.template raw<T>( n );
    return res;
}

/// AFTER A RUN OF THE BODY: the capacity overflows the error buffer recorded for the counts of the scratch ( `ids`, their
/// capacities in `caps` ), read back to the host. Each count that overflowed grows ONCE, to `max( wanted, 2 * capacity )`,
/// `wanted` being the largest it was asked for -- every work-item that ran out records its own overflow, and they are the same
/// shortage ( as `CallArg_Errors.capacity_overflows` reads them ). `true` when at least one grew: the body is then run again,
/// on larger buffers. The other records ( an output's capacity, a failure ) are left as they are: Python reads them once the
/// call returns.
inline bool grow_scratch_capacities( const auto &queue, const std::int32_t *errors, SI nb_words, const SI *ids, SI *caps, int nb_ids ) {
    std::vector<std::int32_t> h( std::size_t( nb_words ), 0 );
    read_to_host( queue, h.data(), errors, SI( sizeof( std::int32_t ) ) * nb_words );
    const SI nb_records = std::min<SI>( SI( h[ 0 ] ), ( nb_words - 1 ) / 3 );
    bool grown = false;
    for ( int i = 0; i < nb_ids; ++i ) {
        SI wanted = -1;
        for ( SI r = 0; r < nb_records; ++r )
            if ( h[ 1 + 3 * r ] == 1 && h[ 2 + 3 * r ] == ids[ i ] )   // ( `ErrorKind::capacity_overflow` )
                wanted = std::max<SI>( wanted, SI( h[ 3 + 3 * r ] ) );
        if ( wanted < 0 )
            continue;
        caps[ i ] = std::max<SI>( wanted, 2 * caps[ i ] );
        grown = true;
    }
    return grown;
}

} // namespace sdot
