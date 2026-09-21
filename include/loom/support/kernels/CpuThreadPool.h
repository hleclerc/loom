#pragma once

#include "../common_macros.h" // LOOM_EXPORT
#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>

namespace sdot {

/// LA file de threads du processus : une seule, pour tous les noyaux (`cpu_thread_pool()`), dans
/// la bibliothèque runtime `libloom_runtime` que chaque bibliothèque générée lie. Les workers sont
/// créés au premier lancement et dorment sur une variable de condition entre deux : coût nul au
/// repos, ~10 µs par réveil. Le fil appelant fait office de worker 0 (pas de réveil pour lui).
///
/// `SDOT_NB_THREADS` fixe leur nombre (défaut : `hardware_concurrency`), `SDOT_PIN_THREADS=1`
/// épingle le worker `w` sur le CPU `w`.
///
/// Répartition : `run_threads( T, job )` appelle `job( t )` pour `t` dans `[ 0, T )`, les fils
/// virtuels étant répartis en tranches CONTIGUËS sur les workers. C'est le découpage qui a gagné
/// dans le banc (`solvers_des_familles/src/util/parallel.h`, « blocks contre strided ») : un fil
/// reste dans SA région de l'espace, ce qui compte quand les items sont en ordre d'arbre.
///
/// L'implémentation est dans `loom/cpp/runtime/cpu_thread_pool.cpp` : rien de tout cela n'a à
/// être recompilé avec chaque noyau.
class LOOM_EXPORT CpuThreadPool {
public:
    CpuThreadPool();
    ~CpuThreadPool();

    CpuThreadPool( const CpuThreadPool & ) = delete;
    CpuThreadPool &operator=( const CpuThreadPool & ) = delete;

    /// `job( t )` pour chaque fil virtuel `t` de `[ 0, nb_threads )`. Rend la main quand tout est
    /// fait. Non réentrant (un `job` ne relance pas la file).
    void run_threads( int nb_threads, const std::function<void( int )> &job );

    int  nb_workers() const { return _nb_workers; }
    bool pinned    () const { return _pin; }

private:
    static void _run_slice( int w, int W, int nb_threads, const std::function<void( int )> &job );
    void        _ensure_workers();
    void        _worker_loop( int w );

    std::vector<std::thread>          _workers;
    std::mutex                        _mutex;
    std::condition_variable           _cv_job, _cv_done;
    const std::function<void( int )> *_current_job        = nullptr;
    int                               _current_nb_threads = 0;
    int                               _current_nb_workers = 0;
    int                               _nb_remaining       = 0;
    long                              _generation         = 0;
    int                               _nb_workers         = 1;
    bool                              _pin                = false;
    bool                              _stop               = false;
};

/// La file du processus, créée au premier appel, jamais détruite (elle peut encore posséder des
/// fils quand le processus démonte ses bibliothèques).
LOOM_EXPORT CpuThreadPool &cpu_thread_pool();

} // namespace sdot
