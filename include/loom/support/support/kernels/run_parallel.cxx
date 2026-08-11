#pragma once

#include "../algorithms/for_each_item.h"
#include "../algorithms/min.h"
#include "sdot/support/common_macros.h"
#include "../algorithms/min.h"
#include "transfer_cost.h"
#include "make_avaiable.h"
#include "run_parallel.h"
#include "QueueEvent.h"
#include "kernel_cost.h"
#include "IoCategory.h"
#include "../Ct.h"

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

    /// Soumet effectivement le `parallel_for` : plus aucune réduction en tête, `args...` sont les
    /// arguments « normaux » du kernel. `reductions` est un tuple de `(usm, identity, op)` ; les
    /// `sycl::reduction` sont créées ici (temporaires) et passées à `parallel_for` avant le kernel.
    sycl::event _do_submit( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                            int nb_items, int nb_threads, auto reductions, auto &&...args ) {
        return reductions.apply_values( [&]( auto... infos ) {
            return q.submit( [&]( sycl::handler &h ) {
                for ( const auto &e : deps.events )
                    h.depends_on( e );
                h.parallel_for( nb_threads, sycl::reduction( infos[ 0_c ], infos[ 1_c ], infos[ 2_c ] )...,
                                [=]( sycl::id<1> thread_id, auto &...reducers ) {
                    // l'item passé au kernel est `item_list[index]` : un Range renvoie l'index plat,
                    // `indices_of(t)` renvoie le multi-indice correspondant. Les reducers (s'il y en a)
                    // précèdent les arguments normaux, comme dans la signature de `func`.
                    //
                    // Un corps peut OPTIONNELLEMENT recevoir, juste après l'item, le NUMÉRO de work-item
                    // `int( thread_id )` (0..nb_threads-1, unique et stable sur toute la boucle striée)
                    // et leur COMPTE `nb_threads` : de quoi indexer un scratch PER-THREAD race-free
                    // (`scratch( thread_index )`), dont la taille ne dépend PAS du nombre d'items. Les
                    // corps générés par `FfiCodeParallel` déclarent ces deux params ; les foncteurs
                    // internes (`ErrorBuffer::fill_with`, ...) ne les prennent pas -> on choisit la forme
                    // d'appel au compile-time selon ce que `func` accepte.
                    for ( int index = thread_id; index < nb_items; index += nb_threads ) {
                        if constexpr ( requires { func( item_list[ index ], int( thread_id ), nb_threads, reducers..., args... ); } )
                            func( item_list[ index ], int( thread_id ), nb_threads, reducers..., args... );
                        else
                            func( item_list[ index ], reducers..., args... );
                    }
                } );
            } );
        } );
    }

    /// Variante coopérative de `_do_submit` : un `nd_range` (au lieu d'un `range` plat), un work-GROUPE
    /// par item (au lieu d'un work-item), `group_size` work-items coopérant dedans via la mémoire
    /// locale (`local_scratch`, un `int32` `local_accessor` construit ici depuis le handler -- sa
    /// taille est un ENTIER RUNTIME, jamais figée dans le type : `generic`/SSCP compile une fois et
    /// JIT-exécute par machine, une taille figée casserait ça). `nb_threads` devient `nb_groups`
    /// (même budget RAM qu'avant : coopérer ne change PAS le scratch O(n) par item concurrent, ça
    /// change juste combien de work-items l'attaquent). Un `group_id` est UNIFORME sur tous les
    /// work-items d'un même groupe (cf commentaire dans le kernel) -- condition nécessaire pour que
    /// les `group_barrier` du corps ne soient pas UB.
    sycl::event _do_submit_grouped( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                            int nb_items, int nb_groups, int group_size, int local_elems, auto reductions, auto &&...args ) {
        return reductions.apply_values( [&]( auto... infos ) {
            return q.submit( [&]( sycl::handler &h ) {
                for ( const auto &e : deps.events )
                    h.depends_on( e );
                sycl::local_accessor<std::int32_t,1> local_scratch( sycl::range<1>( local_elems ), h );
                h.parallel_for( sycl::nd_range<1>( sycl::range<1>( nb_groups * group_size ), sycl::range<1>( group_size ) ),
                                sycl::reduction( infos[ 0_c ], infos[ 1_c ], infos[ 2_c ] )...,
                                [=]( sycl::nd_item<1> nd_item, auto &...reducers ) {
                    const int group_id    = int( nd_item.get_group_linear_id() );
                    const int local_index = int( nd_item.get_local_linear_id() );
                    const int local_size  = int( nd_item.get_local_range( 0 ) );
                    auto group = nd_item.get_group();
                    // sub-group handle: only reachable off `nd_item`, never off `group` (not a
                    // standard SYCL `group` method) -- hence extracted here and threaded through
                    // positionally like `group`, for a body that wants WARP-granularity cooperation
                    // (a shared row per sub-group instead of per work-item, see `OtPlan1d::sort_diracs`).
                    auto sub_group = nd_item.get_sub_group();
                    // `group_id` est le même pour tous les work-items du groupe (uniforme) -> valide
                    // vis-à-vis des `group_barrier`/`local_scratch` que le corps peut appeler dedans.
                    // STABLE sur toute la boucle striée (comme `thread_index` dans `_do_submit`) : le
                    // corps s'en sert pour indexer SA propre ligne de scratch per-groupe
                    // (`scratch( group_index )`), la même à chaque angle traité par ce groupe.
                    for ( int index = group_id; index < nb_items; index += nb_groups )
                        func( item_list[ index ], group_id, local_index, local_size, group, local_scratch, sub_group, reducers..., args... );
                } );
            } );
        } );
    }

    sycl::event _submit_kernel_grouped( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                                int nb_items, int nb_groups, int group_size, int local_elems,
                                std::vector<std::function<void()>> &finalizers,
                                auto reductions, auto &&head, auto &&...tail ) {
        if constexpr ( is_reduction_target<DECAYED_TYPE_OF( head )> ) {
            using T = DECAYED_TYPE_OF( *head.host );
            T  identity = sycl::known_identity_v<DECAYED_TYPE_OF( head.op ),T>;
            T *usm      = sycl::malloc_shared<T>( 1, q );
            *usm = identity;
            finalizers.push_back( [usm,host=head.host,q]() { *host = *usm; sycl::free( usm, q ); } );
            return _submit_kernel_grouped( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems,
                                   finalizers, reductions.with_appended_value( tuple( usm, identity, head.op ) ), FORWARD( tail )... );
        } else
            return _do_submit_grouped( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems, reductions, FORWARD( head ), FORWARD( tail )... );
    }

    sycl::event _submit_kernel_grouped( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                                int nb_items, int nb_groups, int group_size, int local_elems,
                                std::vector<std::function<void()>> &/*finalizers*/,
                                auto reductions ) {
        return _do_submit_grouped( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_groups, group_size, local_elems, reductions );
    }

    /// Pèle les `ReductionTarget` (forcément en tête des args, contrainte SYCL : les reducers suivent
    /// immédiatement l'item). Pour chacun on alloue l'USM, on l'initialise à l'identité, on enregistre
    /// un finalizer (recopie USM -> hôte puis free), et on l'accumule dans `reductions`.
    sycl::event _submit_kernel( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                                int nb_items, int nb_threads, std::vector<std::function<void()>> &finalizers,
                                auto reductions, auto &&head, auto &&...tail ) {
        if constexpr ( is_reduction_target<DECAYED_TYPE_OF( head )> ) {
            using T = DECAYED_TYPE_OF( *head.host );
            T  identity = sycl::known_identity_v<DECAYED_TYPE_OF( head.op ),T>;
            T *usm      = sycl::malloc_shared<T>( 1, q );
            *usm = identity;
            finalizers.push_back( [usm,host=head.host,q]() { *host = *usm; sycl::free( usm, q ); } );
            return _submit_kernel( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, finalizers,
                                   reductions.with_appended_value( tuple( usm, identity, head.op ) ), FORWARD( tail )... );
        } else
            return _do_submit( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, reductions, FORWARD( head ), FORWARD( tail )... );
    }

    /// cas de base : plus aucun argument (que des réductions, ou liste vide).
    sycl::event _submit_kernel( sycl::queue q, const auto &deps, auto &&func, auto &&item_list,
                                int nb_items, int nb_threads, std::vector<std::function<void()>> &/*finalizers*/,
                                auto reductions ) {
        return _do_submit( q, deps, FORWARD( func ), FORWARD( item_list ), nb_items, nb_threads, reductions );
    }

    QueueEvent _run_kernel( auto &&queue, auto &&deps, auto &&func, auto &&item_list, auto &&...args ) {
        const int nb_items = item_list.size();
        int max_nb_threads = nb_items;
        if constexpr ( requires { func.max_nb_threads( args... ); } )
            max_nb_threads = func.max_nb_threads( args... );

        // launch at most `max_nb_threads` work items, each handling a strided slice of the items
        const int nb_threads = min( nb_items, max_nb_threads );
        if ( nb_threads <= 0 )
            return {};

        // submit (et pas parallel_for direct) pour pouvoir déclarer les dépendances `deps`.
        // `submit` n'est pas const -> on passe par une copie de la queue (même queue sous-jacente).
        // capture PAR VALEUR du kernel : il est asynchrone (on ne wait pas ici), donc autonome.
        sycl::queue q = queue.queue;
        std::vector<std::function<void()>> finalizers;

        // chemin coopératif (nd_range) : opt-in via `func.group_size(...)`/`local_mem_elems(...)`
        // (`with_group_kernel`, cf run_parallel.h), même mécanisme opt-in que `max_nb_threads`
        // ci-dessus -- absent (tout kernel qui ne demande pas de groupe, ex `Cell.measure`), on
        // garde le chemin `range` plat inchangé, ci-dessous.
        if constexpr ( requires { func.group_size( args... ); func.local_mem_elems( args... ); } ) {
            const int group_size  = func.group_size( args... );
            const int local_elems = func.local_mem_elems( args... );
            sycl::event ev = _submit_kernel_grouped( q, deps, FORWARD( func ), FORWARD( item_list ),
                                             nb_items, nb_threads, group_size, local_elems, finalizers, tuple(), FORWARD( args )... );
            QueueEvent qe( ev );
            qe.finalizers = std::move( finalizers );
            return qe;
        } else {
            sycl::event ev = _submit_kernel( q, deps, FORWARD( func ), FORWARD( item_list ),
                                             nb_items, nb_threads, finalizers, tuple(), FORWARD( args )... );
            QueueEvent qe( ev );
            qe.finalizers = std::move( finalizers );
            return qe;
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
                    return make_available( queue, io_category, FORWARD( arg ), FORWARD( cont ) );
            }, [&]( auto &&...args ) {
                return _run_kernel( queue, deps, FORWARD( func ), FORWARD( args )... );
            }, InpList(), Ct<int,sizeof...(args)+2>(), item_list, UndefList(), args... );
        } );
        return result;
    }
}

// `second` = soit un Dependencies (déps explicites via after(...)), soit l'item_list (pas de déps).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest ) {
    // une queue seule vaut une liste d'une queue (une queue porte une `sycl::queue`, pas un Tuple)
    if constexpr ( requires { queue_list.queue; } )
        return run_parallel( tuple( FORWARD( queue_list ) ), FORWARD( second ), FORWARD( rest )... );
    else if constexpr ( is_dependencies<DECAYED_TYPE_OF( second )> )
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), FORWARD( second ), FORWARD( rest )... );
    else
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), Dependencies<0>{}, FORWARD( second ), FORWARD( rest )... );
}

// namespace RunDetails {
//     // force max_cpu_threads to 1
//     template<class Func>
//     struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
//         T_VA int max_cpu_threads( A &&.../* args */ ) { return 1_c; }
//         T_VA int max_gpu_threads( A &&.../* args */ ) { return 1_c; }
//     };

//     // decl
//     template<class ES,class F,class L,class... A>        void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail );

//     // end
//     template<class ES,class F,class L,class... A> void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args ) {
//         execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
//     }

//     // Inp
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 1_b, 0_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // Out
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 0_b, 1_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // Mut
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 1_b, 1_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // raw
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 0_b, 0_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 1_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }
// } // namespace RunDetails

// template<class L,class F,class... A> void run_parallel( L &&list, F &&func, A &&...args ) {
//     // statically chosen from the args memory spaces (single type -> only this branch compiles)
//     auto execution_space = execution_space_for( args... );

//     //
//     RunDetails::run_parallel_from( execution_space, Ct<int,2+sizeof...( args )>(), func, Inp(), FORWARD( list ), FORWARD( args )... );

//     // make every arg accessible from that space (pass-through or transfer), then run
//     // RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )...,
//     //     RunDetails::LaunchOn<DECAYED_TYPE_OF( execution_space ),DECAYED_TYPE_OF( func )>{ execution_space, func }
//     // );
// }

// template<class L,class F,class... A> void run_sequential( L &&list, F &&func, A &&...args ) {
//     run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
// }

} // namespace sdot
