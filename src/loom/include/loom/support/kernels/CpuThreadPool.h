#pragma once

#include "../common_macros.h" // LOOM_EXPORT
#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>

namespace sdot {

/// THE process-wide thread pool: a single one, for all the kernels (`cpu_thread_pool()`), in
/// the runtime library `libloom_runtime` that every generated library links. The workers are
/// created at the first launch and sleep on a condition variable between two: zero cost at
/// rest, ~10 µs per wake-up. The calling thread acts as worker 0 (no wake-up for it).
///
/// `SDOT_NB_THREADS` sets their number (default: `hardware_concurrency`), `SDOT_PIN_THREADS=1`
/// pins worker `w` to CPU `w`.
///
/// Distribution: `run_threads( T, job )` calls `job( t )` for `t` in `[ 0, T )`, the virtual
/// threads being distributed in CONTIGUOUS slices over the workers. This is the split that won
/// in the benchmark (`solvers_des_familles/src/util/parallel.h`, "blocks vs strided"): a thread
/// stays in ITS region of space, which matters when the items are in tree order.
///
/// The implementation is in `loom/cpp/runtime/cpu_thread_pool.cpp`: none of this has to
/// be recompiled with each kernel.
class LOOM_EXPORT CpuThreadPool {
public:
    CpuThreadPool();
    ~CpuThreadPool();

    CpuThreadPool( const CpuThreadPool & ) = delete;
    CpuThreadPool &operator=( const CpuThreadPool & ) = delete;

    /// `job( t )` for each virtual thread `t` of `[ 0, nb_threads )`. Returns when everything is
    /// done. Not reentrant (a `job` does not relaunch the pool).
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

/// The process-wide pool, created at the first call, never destroyed (it may still own
/// threads when the process unloads its libraries).
LOOM_EXPORT CpuThreadPool &cpu_thread_pool();

} // namespace sdot
