#pragma once

#include "CpuKernelMemorySpace.h"
#include "CpuHostMemorySpace.h"
#include "../common_macros.h"
#include "../common_types.h"
#include "QueueEvent.h"
#include "IoCategory.h"
#include "Reducer.h"
#include "Group.h"
#include "../Ct.h"

#include <condition_variable>
#include <functional>
#include <cstdlib>
#include <barrier>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <tuple>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace sdot {

/// Le contexte d'exécution CPU : une file de threads PERSISTANTE, et les deux formes de
/// lancement qu'un kernel peut demander (`submit_kernel`, `submit_kernel_grouped`).
///
/// C'est ICI que vit ce qui est propre au CPU -- la boucle sur les items, la répartition entre
/// fils, les réductions par fil. `run_parallel` ne sait rien du device : il pèle les catégories
/// d'entrée/sortie, rend les arguments disponibles, et appelle `submit_kernel( queue, ... )` --
/// une surcharge par type de queue, trouvée par ADL. Ajouter un device = écrire sa queue et ses
/// deux `submit_kernel`, pas un `if` dans `run_parallel`.
///
/// Répartition : chaque « fil virtuel » `t` de `[ 0, nb_threads )` traite une tranche CONTIGUË des
/// items (`[ t n / T, ( t + 1 ) n / T )`), et les fils virtuels sont eux-mêmes répartis en
/// tranches contiguës sur les workers. C'est le découpage qui a gagné dans le banc
/// (`solvers_des_familles/src/util/parallel.h`, « blocks contre strided ») : un fil reste dans
/// SA région de l'espace, ce qui compte quand les items sont en ordre d'arbre. Le contrat du
/// corps ne change pas : `thread_index` est unique et STABLE pour tout ce qu'un fil traite (une
/// ligne de scratch par fil), et `nb_threads` est leur compte.
///
/// Les workers sont créés au premier lancement et dorment sur une variable de condition entre
/// deux : coût nul au repos, ~10 µs par réveil. Le fil appelant fait office de worker 0 (pas de
/// réveil pour lui). `SDOT_NB_THREADS` fixe leur nombre (défaut : `hardware_concurrency`),
/// `SDOT_PIN_THREADS=1` épingle le worker `w` sur le CPU `w`.
///
/// Une queue par bibliothèque générée pour l'instant (chaque `.so` a sa statique) : le runtime
/// partagé, avec une seule file par processus, est l'étape « couches » de la refonte.
///
/// La file elle-même (`CpuThreadPool`) n'est pas copiable ; `CpuQueue` est la POIGNÉE qu'on
/// passe partout, copiable à volonté (un `shared_ptr` sur la file), comme l'était la queue SYCL.
struct CpuThreadPool {
    CpuThreadPool() {
        nb_workers = 0;
        if ( const char *env = std::getenv( "SDOT_NB_THREADS" ) )
            nb_workers = std::atoi( env );
        if ( nb_workers <= 0 )
            nb_workers = int( std::thread::hardware_concurrency() );
        if ( nb_workers <= 0 )
            nb_workers = 1;
        if ( const char *env = std::getenv( "SDOT_PIN_THREADS" ) )
            pin = std::atoi( env ) != 0;
    }

    ~CpuThreadPool() {
        {
            std::lock_guard<std::mutex> lock( mutex );
            stop = true;
            ++generation;
        }
        cv_job.notify_all();
        for ( auto &th : workers )
            th.join();
    }

    CpuThreadPool( const CpuThreadPool & ) = delete;
    CpuThreadPool &operator=( const CpuThreadPool & ) = delete;

    /// `job( t )` pour chaque fil virtuel `t` de `[ 0, nb_threads )`, réparti par tranches
    /// contiguës sur les workers. Rend la main quand tout est fait.
    void run_threads( int nb_threads, const std::function<void( int )> &job ) {
        if ( nb_threads <= 0 )
            return;
        const int W = std::min( nb_threads, nb_workers );
        if ( W == 1 ) {
            for ( int t = 0; t < nb_threads; ++t )
                job( t );
            return;
        }
        _ensure_workers();

        {
            std::lock_guard<std::mutex> lock( mutex );
            current_job        = &job;
            current_nb_threads = nb_threads;
            current_nb_workers = W;
            nb_remaining       = W - 1;
            ++generation;
        }
        cv_job.notify_all();

        _run_slice( 0, W, nb_threads, job );

        std::unique_lock<std::mutex> lock( mutex );
        cv_done.wait( lock, [&] { return nb_remaining == 0; } );
        current_job = nullptr;
    }

    int  nb_workers;   ///< fils disponibles, fil appelant compris
    bool pin = false;  ///< épingler le worker `w` sur le CPU `w`

private:
    static void _run_slice( int w, int W, int nb_threads, const std::function<void( int )> &job ) {
        const int b = int( ( long long ) w * nb_threads / W );
        const int e = int( ( long long ) ( w + 1 ) * nb_threads / W );
        for ( int t = b; t < e; ++t )
            job( t );
    }

    void _ensure_workers() {
        if ( ! workers.empty() )
            return;
        workers.reserve( nb_workers - 1 );
        for ( int w = 1; w < nb_workers; ++w )
            workers.emplace_back( [this,w] { _worker_loop( w ); } );
    }

    void _worker_loop( int w ) {
        if ( pin ) {
#ifdef __linux__
            cpu_set_t set;
            CPU_ZERO( &set );
            CPU_SET( w, &set );
            pthread_setaffinity_np( pthread_self(), sizeof( set ), &set );
#endif
        }
        long seen = 0;
        for ( ;; ) {
            const std::function<void( int )> *job;
            int nb_threads, W;
            {
                std::unique_lock<std::mutex> lock( mutex );
                cv_job.wait( lock, [&] { return generation != seen; } );
                seen = generation;
                if ( stop )
                    return;
                job = current_job; nb_threads = current_nb_threads; W = current_nb_workers;
            }
            if ( w < W )
                _run_slice( w, W, nb_threads, *job );
            {
                std::lock_guard<std::mutex> lock( mutex );
                if ( w < W && --nb_remaining == 0 )
                    cv_done.notify_one();
            }
        }
    }

    std::vector<std::thread>          workers;
    std::mutex                        mutex;
    std::condition_variable           cv_job, cv_done;
    const std::function<void( int )> *current_job        = nullptr;
    int                               current_nb_threads = 0;
    int                               current_nb_workers = 0;
    int                               nb_remaining       = 0;
    long                              generation         = 0;
    bool                              stop               = false;
};

struct CpuQueue {
    /// zone mémoire par défaut vue par les kernels lancés sur cette queue (un contexte
    /// d'exécution peut exposer plusieurs zones ; celle-ci est celle utilisée par défaut)
    using DefaultKernelMemorySpace = CpuKernelMemorySpace;

    CpuQueue() : pool( std::make_shared<CpuThreadPool>() ) {}

    void run_threads( int nb_threads, const std::function<void( int )> &job ) const { pool->run_threads( nb_threads, job ); }
    int  nb_workers () const { return pool->nb_workers; }

    std::shared_ptr<CpuThreadPool> pool;
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
