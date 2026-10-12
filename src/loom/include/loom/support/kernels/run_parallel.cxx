#pragma once

#include "../algorithms/min.h"
#include <loom/support/common_macros.h>
#include "make_avaiable.h"
#include "run_parallel.h"
#include "QueueEvent.h"
#include "IoCategory.h"
#include "../Ct.h"
#include <tuple>

namespace sdot {

namespace detail::RunParallel {
    /// The kernel form of each argument, in a `std::tuple`: `( io, a, io, b, c, ... )` ->
    /// `( map( io, a ), map( io, b ), map( io, c ) )`, a category applying to what follows it.
    /// A fold over the list, with no counter or continuation -- `kernel_form` returns a VALUE, and the
    /// original form (`Ct<int,n>` counter + continuations, values rotated to the end of the
    /// list) made the nvcc 13.4 frontend collapse as soon as the counter was dependent.
    auto _map_args( const auto &/*map*/, auto /*io_category*/ ) {
        return std::tuple<>();
    }

    auto _map_args( const auto &map, auto io_category, auto &&head, auto &&...tail ) {
        if constexpr ( is_io_category<DECAYED_TYPE_OF( head )> )
            return _map_args( map, head, FORWARD( tail )... );
        else
            return std::tuple_cat( std::make_tuple( map( io_category, FORWARD( head ) ) ), _map_args( map, io_category, FORWARD( tail )... ) );
    }

    /// Peels the `ReductionTarget`s (necessarily at the head of the args: the accumulators follow
    /// the item immediately in the body's signature) into `reduction_targets`, a `std::tuple`
    /// that the queue receives as is -- it is the one that knows how to accumulate (one line per thread on
    /// CPU) and copy back into the host target.
    auto _submit_kernel( auto &queue, const auto &deps, auto &&func, auto &&item_list,
                         int nb_items, int nb_threads, auto reduction_targets, auto &&head, auto &&...tail ) {
        if constexpr ( is_reduction_target<DECAYED_TYPE_OF( head )> )
            return _submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads,
                                   std::tuple_cat( reduction_targets, std::make_tuple( head ) ), FORWARD( tail )... );
        else
            return submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, reduction_targets, FORWARD( head ), FORWARD( tail )... );
    }

    /// base case: no argument left (only reductions, or empty list).
    auto _submit_kernel( auto &queue, const auto &deps, auto &&func, auto &&item_list,
                         int nb_items, int nb_threads, auto reduction_targets ) {
        return submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, reduction_targets );
    }

    auto _submit_kernel_grouped( auto &queue, const auto &deps, auto &&func, auto &&item_list,
                                 int nb_items, int nb_groups, int group_size, int local_elems, auto reduction_targets, auto &&head, auto &&...tail ) {
        if constexpr ( is_reduction_target<DECAYED_TYPE_OF( head )> )
            return _submit_kernel_grouped( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems,
                                           std::tuple_cat( reduction_targets, std::make_tuple( head ) ), FORWARD( tail )... );
        else
            return submit_kernel_grouped( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems, reduction_targets, FORWARD( head ), FORWARD( tail )... );
    }

    auto _submit_kernel_grouped( auto &queue, const auto &deps, auto &&func, auto &&item_list,
                                 int nb_items, int nb_groups, int group_size, int local_elems, auto reduction_targets ) {
        return submit_kernel_grouped( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems, reduction_targets );
    }

    QueueEvent _run_kernel( auto &&queue, auto &&deps, auto &&func, auto &&item_list, auto &&...args ) {
        const int nb_items = item_list.size();
        int max_nb_threads = nb_items;
        if constexpr ( requires { func.max_nb_threads( args... ); } )
            max_nb_threads = func.max_nb_threads( args... );

        // launch at most `max_nb_threads` work items, each handling a slice of the items
        const int nb_threads = min( nb_items, max_nb_threads );
        if ( nb_threads <= 0 )
            return {};

        // cooperative path: opt-in via `func.group_size(...)`/`local_mem_elems(...)`
        // (`with_group_kernel`, see run_parallel.h), same opt-in mechanism as `max_nb_threads`
        // above -- when absent (any kernel that does not ask for a group, e.g. `Cell.measure`), we
        // keep the flat path, below.
        if constexpr ( requires { func.group_size( args... ); func.local_mem_elems( args... ); } ) {
            const int group_size  = func.group_size( args... );
            const int local_elems = func.local_mem_elems( args... );
            return _submit_kernel_grouped( queue, deps, FORWARD( func ), FORWARD( item_list ),
                                           nb_items, nb_threads, group_size, local_elems, std::tuple<>(), FORWARD( args )... );
        } else {
            return _submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ),
                                   nb_items, nb_threads, std::tuple<>(), FORWARD( args )... );
        }
    }

    // body of run_parallel, with explicit dependencies `deps` (may be Dependencies<0>).
    //
    // ONE queue. The original form took a LIST of queues and chose the cheapest one
    // (transfers included) in a `for_each_item` loop with a generic lambda -- a choice that nothing
    // exercises (a call has ONE context, that of its device), and a construction on which the
    // nvcc 13.4 (EDG) frontend collapses ("Segmentation fault" on any kernel, whereas 13.3
    // was fine). The day two contexts compete for a call, the selection will be made HERE, before
    // the chain of continuations, not inside it.
    template<class Queue,class Deps,class ItemList,class Func,class... Args>
    auto _run_parallel( Queue &&queue, Deps &&deps, ItemList &&item_list, Func &&func, Args &&...args ) {
        auto mapped = _map_args( [&]( auto io_category, auto &&arg ) {
            // a reduction target is not "made available" (it is a host scalar):
            // we turn it into a `ReductionTarget` (op + host pointer), handled by `_submit_kernel`.
            if constexpr ( is_red_list<DECAYED_TYPE_OF( io_category )> )
                return ReductionTarget{ io_category.op, &arg };
            else
                return kernel_form( queue, io_category, FORWARD( arg ) );
        }, InpList(), item_list, UndefList(), args... );
        return std::apply( [&]( auto &&...m ) {
            return _run_kernel( queue, deps, FORWARD( func ), FORWARD( m )... );
        }, std::move( mapped ) );
    }
}

// `second` = either a Dependencies (explicit deps via after(...)), or the item_list (no deps).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest ) {
    // a list of ONE queue is worth the queue (the old context-choosing form, see `_run_parallel`)
    if constexpr ( ! requires { typename DECAYED_TYPE_OF( queue_list )::DefaultKernelMemorySpace; } ) {
        static_assert( DECAYED_TYPE_OF( queue_list )::ct_size == 1, "run_parallel: a single queue only" );
        return run_parallel( queue_list[ Ct<int,0>() ], FORWARD( second ), FORWARD( rest )... );
    } else if constexpr ( is_dependencies<DECAYED_TYPE_OF( second )> )
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), FORWARD( second ), FORWARD( rest )... );
    else
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), Dependencies<0>{}, FORWARD( second ), FORWARD( rest )... );
}

} // namespace sdot
