#pragma once

#include "CpuKernelMemorySpace.h"
#include "CpuHostMemorySpace.h"
#include "../common_macros.h"
#include "../common_types.h"
#include "CpuThreadPool.h"
#include "QueueEvent.h"
#include "IoCategory.h"
#include "Reducer.h"
#include "Group.h"
#include "../Ct.h"

#include <functional>
#include <barrier>
#include <thread>
#include <vector>
#include <memory>
#include <tuple>

namespace sdot {

/// Le contexte d'exécution CPU : la file de threads du processus (`cpu_thread_pool()`, dans la
/// bibliothèque runtime), et les deux formes de lancement qu'un kernel peut demander
/// (`submit_kernel`, `submit_kernel_grouped`).
///
/// C'est ICI que vit ce qui est propre au CPU -- la boucle sur les items, la répartition entre
/// fils, les réductions par fil. `run_parallel` ne sait rien du device : il pèle les catégories
/// d'entrée/sortie, rend les arguments disponibles, et appelle `submit_kernel( queue, ... )` --
/// une surcharge par type de queue, trouvée par ADL. Ajouter un device = écrire sa queue et ses
/// deux `submit_kernel`, pas un `if` dans `run_parallel`.
///
/// Chaque « fil virtuel » `t` de `[ 0, nb_threads )` traite une tranche CONTIGUË des items
/// (`[ t n / T, ( t + 1 ) n / T )`), voir `CpuThreadPool`. Le contrat du corps ne change pas :
/// `thread_index` est unique et STABLE pour tout ce qu'un fil traite (une ligne de scratch par
/// fil), et `nb_threads` est leur compte.
///
/// Une poignée trivialement copiable, comme l'était la queue SYCL.
struct CpuQueue {
    /// zone mémoire par défaut vue par les kernels lancés sur cette queue (un contexte
    /// d'exécution peut exposer plusieurs zones ; celle-ci est celle utilisée par défaut)
    using DefaultKernelMemorySpace = CpuKernelMemorySpace;

    CpuQueue() : pool( &cpu_thread_pool() ) {}

    void run_threads( int nb_threads, const std::function<void( int )> &job ) const { pool->run_threads( nb_threads, job ); }
    int  nb_workers () const { return pool->nb_workers(); }

    CpuThreadPool *pool;
};

/// Coût de transfert (secondes par octet) pour rendre une zone source accessible depuis ce
/// contexte d'exécution. Un contexte connaît les zones qu'il peut atteindre (l'inverse non),
/// donc ces surcharges vivent près du contexte.
///   CPU -> CPU : la donnée est déjà en RAM hôte, rien à transférer.
constexpr auto transfer_cost_per_byte( const CpuQueue &, CpuHostMemorySpace ) { return Ct<double,0.0>(); }

namespace detail::CpuQueueLaunch {
    /// appel du corps pour un item, avec ou sans les infos de fil selon ce que `func` accepte
    /// (les corps générés par `FfiCodeParallel` les prennent ; `TensorView::fill_with` non).
    void call( auto &&func, auto &&item, int thread_index, int nb_threads, auto &...reducers_and_args ) {
        if constexpr ( requires { func( item, thread_index, nb_threads, reducers_and_args... ); } )
            func( item, thread_index, nb_threads, reducers_and_args... );
        else
            func( item, reducers_and_args... );
    }

    /// une ligne de réduction par fil virtuel, initialisée à l'identité
    template<class Op,class T>
    auto partials_for( const ReductionTarget<Op,T> &target, int nb_threads ) {
        return std::vector<Reducer<Op,T>>( nb_threads, Reducer<Op,T>{ target.op, Reducer<Op,T>::identity( target.op ) } );
    }
}

/// Lancement plat : `nb_threads` fils virtuels, chacun sa tranche d'items. `reduction_targets`
/// est un `std::tuple` de `ReductionTarget` (op + pointeur hôte), pelés par `run_parallel` en
/// tête des arguments ; chaque fil accumule dans son propre `Reducer`, combinés à la fin.
/// Synchrone (la file rend la main quand tout est fait) : l'événement rendu est déjà complet.
/// Les dépendances `deps` sont attendues avant de commencer (gratuit si elles le sont déjà).
auto submit_kernel( const CpuQueue &queue, const auto &deps, auto &&func, auto &&item_list,
                    int nb_items, int nb_threads, auto reduction_targets, auto &&...args ) {
    deps.wait_all();
    return std::apply( [&]( auto &...targets ) {
        auto partials = std::make_tuple( detail::CpuQueueLaunch::partials_for( targets, nb_threads )... );
        std::apply( [&]( auto &...vecs ) {
            queue.run_threads( nb_threads, [&]( int t ) {
                const int b = int( ( long long ) t * nb_items / nb_threads );
                const int e = int( ( long long ) ( t + 1 ) * nb_items / nb_threads );
                for ( int index = b; index < e; ++index )
                    detail::CpuQueueLaunch::call( func, item_list[ index ], t, nb_threads, vecs[ t ]..., args... );
            } );
            // combinaison des lignes dans la cible hôte
            ( [&]( auto &target, auto &vec ) {
                for ( auto &r : vec )
                    *target.host = target.op( *target.host, r.value );
            }( targets, vecs ), ... );
        }, partials );
        return QueueEvent{};
    }, reduction_targets );
}

/// Lancement coopératif : un GROUPE de `group_size` voies par item concurrent (`nb_groups` groupes
/// en vol), qui partagent `local_scratch` (`local_elems` mots `int32`) et se synchronisent par
/// `group_barrier( group )`.
///
/// `group_size == 1` (la production sur CPU, voir `Cpu.group_size` côté python) se réduit au
/// lancement plat : une voie, pas de barrière, un scratch par fil. Au-delà, chaque groupe est
/// `group_size` fils système autour d'un `std::barrier` -- correct, pas rapide : c'est le chemin
/// qui fait tourner sur CPU un kernel écrit pour les groupes d'un GPU, à fins de test.
auto submit_kernel_grouped( const CpuQueue &queue, const auto &deps, auto &&func, auto &&item_list,
                            int nb_items, int nb_groups, int group_size, int local_elems, auto reduction_targets, auto &&...args ) {
    deps.wait_all();
    static_assert( std::tuple_size_v<DECAYED_TYPE_OF( reduction_targets )> == 0,
                   "les réductions ne sont pas supportées dans un kernel de groupe" );
    const int scratch_stride = std::max( local_elems, 1 );
    std::vector<std::int32_t> scratch( std::size_t( scratch_stride ) * std::size_t( nb_groups ) );

    if ( group_size <= 1 ) {
        queue.run_threads( nb_groups, [&]( int g ) {
            CpuGroup    group{ nullptr, 1 };
            CpuSubGroup sub_group{ 0, 1 };
            std::int32_t *local_scratch = scratch.data() + std::size_t( g ) * scratch_stride;
            for ( int index = g; index < nb_items; index += nb_groups )
                func( item_list[ index ], g, 0, 1, group, local_scratch, sub_group, args... );
        } );
        return QueueEvent{};
    }

    // un groupe = `group_size` fils système et une barrière ; les groupes tournent en parallèle
    std::vector<std::thread> lanes;
    lanes.reserve( std::size_t( nb_groups ) * group_size );
    std::vector<std::unique_ptr<std::barrier<>>> barriers;
    barriers.reserve( nb_groups );
    for ( int g = 0; g < nb_groups; ++g ) {
        barriers.emplace_back( std::make_unique<std::barrier<>>( group_size ) );
        for ( int lane = 0; lane < group_size; ++lane ) {
            lanes.emplace_back( [&,g,lane] {
                CpuGroup    group{ barriers[ g ].get(), group_size };
                CpuSubGroup sub_group{ lane, 1 };
                std::int32_t *local_scratch = scratch.data() + std::size_t( g ) * scratch_stride;
                for ( int index = g; index < nb_items; index += nb_groups )
                    func( item_list[ index ], g, lane, group_size, group, local_scratch, sub_group, args... );
            } );
        }
    }
    for ( auto &th : lanes )
        th.join();
    return QueueEvent{};
}

} // namespace sdot
