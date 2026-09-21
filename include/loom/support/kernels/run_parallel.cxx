#pragma once

#include "../algorithms/for_each_item.h"
#include "../algorithms/min.h"
#include <loom/support/common_macros.h>
#include "transfer_cost.h"
#include "make_avaiable.h"
#include "run_parallel.h"
#include "QueueEvent.h"
#include "kernel_cost.h"
#include "IoCategory.h"
#include "../Ct.h"
#include <tuple>

namespace sdot {

namespace detail::RunParallel {
    template<int nb_args>
    auto _map_reduce_run_arg( const auto &map, const auto &reduce, auto io_category, Ct<int,nb_args>, auto &&head, auto &&...tail ) {
        if constexpr ( nb_args == 0 )
            return reduce( FORWARD( head ), FORWARD( tail )... );
        else if constexpr ( is_io_category<DECAYED_TYPE_OF( head )> )
            return _map_reduce_run_arg( map, reduce, head, Ct<int,nb_args-1>(), FORWARD( tail )... );
        else {
            return map( io_category, FORWARD( head ), [&]( auto &&mapped ) {
                return _map_reduce_run_arg( map, reduce, io_category, Ct<int,nb_args-1>(), FORWARD( tail )..., FORWARD( mapped ) );
            } );
        }
    }

    /// Pèle les `ReductionTarget` (forcément en tête des args : les accumulateurs suivent
    /// immédiatement l'item dans la signature du corps) dans `reduction_targets`, un `std::tuple`
    /// que la queue reçoit tel quel -- c'est elle qui sait comment accumuler (une ligne par fil sur
    /// CPU) et recopier dans la cible hôte.
    auto _submit_kernel( auto &queue, const auto &deps, auto &&func, auto &&item_list,
                         int nb_items, int nb_threads, auto reduction_targets, auto &&head, auto &&...tail ) {
        if constexpr ( is_reduction_target<DECAYED_TYPE_OF( head )> )
            return _submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads,
                                   std::tuple_cat( reduction_targets, std::make_tuple( head ) ), FORWARD( tail )... );
        else
            return submit_kernel( queue, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, reduction_targets, FORWARD( head ), FORWARD( tail )... );
    }

    /// cas de base : plus aucun argument (que des réductions, ou liste vide).
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

        // chemin coopératif : opt-in via `func.group_size(...)`/`local_mem_elems(...)`
        // (`with_group_kernel`, cf run_parallel.h), même mécanisme opt-in que `max_nb_threads`
        // ci-dessus -- absent (tout kernel qui ne demande pas de groupe, ex `Cell.measure`), on
        // garde le chemin plat, ci-dessous.
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

    // corps de run_parallel, avec dépendances explicites `deps` (peut être Dependencies<0>)
    auto _run_parallel( auto &&queue_list, auto &&deps, auto &&item_list, auto &&func, auto &&...args ) {
        // costs
        auto costs = apply_values( FORWARD( queue_list ), [&]( auto &&...queues ) {
            auto cost_for = [&]( auto &&queue ) {
                return _map_reduce_run_arg( [&]( auto io_category, const auto &arg, auto &&cont ) {
                    return cont( transfer_cost( queue, io_category, arg ) );
                }, [&]( auto &&...map_out ) {
                    return ( map_out + ... + kernel_cost( func, queue, item_list, args... ) );
                }, InpList(), Ct<int,sizeof...(args)+2>(), item_list, UndefList(), args... );
            };
            return tuple( cost_for( queues )... );
        } );

        // first cost == min cost
        double min_cost = costs.apply_values( []( auto...values ) { return min( values... ); } );
        int index_in_queue = 0;
        bool done = false;
        QueueEvent result; ///< event du contexte choisi (RAII : wait à la destruction si non consommé)
        for_each_item( queue_list, [&]( auto &&queue ) {
            double cost = costs[ index_in_queue++ ];
            if ( done || cost > min_cost )
                return;
            done = true;

            result = _map_reduce_run_arg( [&]( auto io_category, auto &&arg, auto &&cont ) {
                // une cible de réduction n'est pas « rendue disponible » (c'est un scalaire hôte) :
                // on la transforme en `ReductionTarget` (op + pointeur hôte), traitée par `_submit_kernel`.
                if constexpr ( is_red_list<DECAYED_TYPE_OF( io_category )> )
                    return cont( ReductionTarget{ io_category.op, &arg } );
                else
                    return cont( kernel_form( queue, io_category, FORWARD( arg ) ) );
            }, [&]( auto &&...args ) {
                return _run_kernel( queue, deps, FORWARD( func ), FORWARD( args )... );
            }, InpList(), Ct<int,sizeof...(args)+2>(), item_list, UndefList(), args... );
        } );
        return result;
    }
}

// `second` = soit un Dependencies (déps explicites via after(...)), soit l'item_list (pas de déps).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest ) {
    // une queue seule vaut une liste d'une queue (une queue est un contexte d'exécution, pas un Tuple)
    if constexpr ( requires { typename DECAYED_TYPE_OF( queue_list )::DefaultKernelMemorySpace; } )
        return run_parallel( tuple( FORWARD( queue_list ) ), FORWARD( second ), FORWARD( rest )... );
    else if constexpr ( is_dependencies<DECAYED_TYPE_OF( second )> )
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), FORWARD( second ), FORWARD( rest )... );
    else
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), Dependencies<0>{}, FORWARD( second ), FORWARD( rest )... );
}

} // namespace sdot
