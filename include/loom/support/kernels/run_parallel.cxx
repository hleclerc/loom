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
    /// La forme noyau de chaque argument, dans un `std::tuple` : `( io, a, io, b, c, ... )` ->
    /// `( map( io, a ), map( io, b ), map( io, c ) )`, une catégorie valant pour ce qui la suit.
    /// Un pli sur la liste, sans compteur ni continuation -- `kernel_form` rend une VALEUR, et la
    /// forme d'origine (compteur `Ct<int,n>` + continuations, les valeurs tournées en fin de
    /// liste) faisait s'effondrer le frontend de nvcc 13.4 dès que le compteur était dépendant.
    auto _map_args( const auto &/*map*/, auto /*io_category*/ ) {
        return std::tuple<>();
    }

    auto _map_args( const auto &map, auto io_category, auto &&head, auto &&...tail ) {
        if constexpr ( is_io_category<DECAYED_TYPE_OF( head )> )
            return _map_args( map, head, FORWARD( tail )... );
        else
            return std::tuple_cat( std::make_tuple( map( io_category, FORWARD( head ) ) ), _map_args( map, io_category, FORWARD( tail )... ) );
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

    // corps de run_parallel, avec dépendances explicites `deps` (peut être Dependencies<0>).
    //
    // UNE queue. La forme d'origine prenait une LISTE de queues et choisissait la moins coûteuse
    // (transferts compris) dans une boucle `for_each_item` à lambda générique -- un choix que rien
    // n'exerce (un appel a UN contexte, celui de son device), et une construction sur laquelle le
    // frontend de nvcc 13.4 (EDG) s'effondre (« Segmentation fault » sur tout noyau, quand 13.3
    // passait). Le jour où deux contextes se disputent un appel, la sélection se fera ICI, avant
    // la chaîne de continuations, pas dedans.
    template<class Queue,class Deps,class ItemList,class Func,class... Args>
    auto _run_parallel( Queue &&queue, Deps &&deps, ItemList &&item_list, Func &&func, Args &&...args ) {
        auto mapped = _map_args( [&]( auto io_category, auto &&arg ) {
            // une cible de réduction n'est pas « rendue disponible » (c'est un scalaire hôte) :
            // on la transforme en `ReductionTarget` (op + pointeur hôte), traitée par `_submit_kernel`.
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

// `second` = soit un Dependencies (déps explicites via after(...)), soit l'item_list (pas de déps).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest ) {
    // une liste d'UNE queue vaut la queue (l'ancienne forme à choix de contexte, voir `_run_parallel`)
    if constexpr ( ! requires { typename DECAYED_TYPE_OF( queue_list )::DefaultKernelMemorySpace; } ) {
        static_assert( DECAYED_TYPE_OF( queue_list )::ct_size == 1, "run_parallel : une seule queue" );
        return run_parallel( queue_list[ Ct<int,0>() ], FORWARD( second ), FORWARD( rest )... );
    } else if constexpr ( is_dependencies<DECAYED_TYPE_OF( second )> )
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), FORWARD( second ), FORWARD( rest )... );
    else
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), Dependencies<0>{}, FORWARD( second ), FORWARD( rest )... );
}

} // namespace sdot
