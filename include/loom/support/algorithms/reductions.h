#pragma once

#include <loom/support/common_macros.h> // HD

#include "../kernels/run_parallel.h" // run_parallel, RedList, InpList
#include "indices_of.h"
#include "../kernels/Reducer.h"
#include <limits>

namespace sdot {

/// Reductions built *on top of* `run_parallel`: `RedList(op)` asks the queue for a private
/// per-thread accumulator (`Reducer`), combined into the host target once the kernel is done.
/// The temporary `QueueEvent` returned by `run_parallel` is destroyed at the end of the expression: it waits for
/// the kernel to finish then runs its finalizers -> `res` is ready at the `return`.

HD auto sum( auto &&queue_list, auto &&a ) {
    using TF = typename DECAYED_TYPE_OF( a )::TF;
    TF res = 0;
    run_parallel( FORWARD( queue_list ), indices_of( a ),
        []( auto idx, auto &r, auto a ) { r.combine( a[ idx ].value() ); },
        RedList( plus<TF>() ), res, InpList(), a );
    return res;
}

HD auto max( auto &&queue_list, auto &&a ) {
    using TF = typename DECAYED_TYPE_OF( a )::TF;
    TF res = std::numeric_limits<TF>::lowest();
    run_parallel( FORWARD( queue_list ), indices_of( a ),
        []( auto idx, auto &r, auto a ) { r.combine( a[ idx ].value() ); },
        RedList( maximum<TF>() ), res, InpList(), a );
    return res;
}

} // namespace sdot
