// La file de threads du processus -- voir `loom/support/kernels/CpuThreadPool.h`.
#include <loom/support/kernels/CpuThreadPool.h>
#include <cstdlib>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace sdot {

CpuThreadPool::CpuThreadPool() {
    _nb_workers = 0;
    if ( const char *env = std::getenv( "SDOT_NB_THREADS" ) )
        _nb_workers = std::atoi( env );
    if ( _nb_workers <= 0 )
        _nb_workers = int( std::thread::hardware_concurrency() );
    if ( _nb_workers <= 0 )
        _nb_workers = 1;
    if ( const char *env = std::getenv( "SDOT_PIN_THREADS" ) )
        _pin = std::atoi( env ) != 0;
}

CpuThreadPool::~CpuThreadPool() {
    {
        std::lock_guard<std::mutex> lock( _mutex );
        _stop = true;
        ++_generation;
    }
    _cv_job.notify_all();
    for ( auto &th : _workers )
        th.join();
}

void CpuThreadPool::run_threads( int nb_threads, const std::function<void( int )> &job ) {
    if ( nb_threads <= 0 )
        return;
    const int W = std::min( nb_threads, _nb_workers );
    if ( W == 1 ) {
        for ( int t = 0; t < nb_threads; ++t )
            job( t );
        return;
    }
    _ensure_workers();

    {
        std::lock_guard<std::mutex> lock( _mutex );
        _current_job        = &job;
        _current_nb_threads = nb_threads;
        _current_nb_workers = W;
        _nb_remaining       = W - 1;
        ++_generation;
    }
    _cv_job.notify_all();

    _run_slice( 0, W, nb_threads, job );

    std::unique_lock<std::mutex> lock( _mutex );
    _cv_done.wait( lock, [&] { return _nb_remaining == 0; } );
    _current_job = nullptr;
}

void CpuThreadPool::_run_slice( int w, int W, int nb_threads, const std::function<void( int )> &job ) {
    const int b = int( ( long long ) w * nb_threads / W );
    const int e = int( ( long long ) ( w + 1 ) * nb_threads / W );
    for ( int t = b; t < e; ++t )
        job( t );
}

void CpuThreadPool::_ensure_workers() {
    if ( ! _workers.empty() )
        return;
    _workers.reserve( _nb_workers - 1 );
    for ( int w = 1; w < _nb_workers; ++w )
        _workers.emplace_back( [this,w] { _worker_loop( w ); } );
}

void CpuThreadPool::_worker_loop( int w ) {
    if ( _pin ) {
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
            std::unique_lock<std::mutex> lock( _mutex );
            _cv_job.wait( lock, [&] { return _generation != seen; } );
            seen = _generation;
            if ( _stop )
                return;
            job = _current_job; nb_threads = _current_nb_threads; W = _current_nb_workers;
        }
        if ( w < W )
            _run_slice( w, W, nb_threads, *job );
        {
            std::lock_guard<std::mutex> lock( _mutex );
            if ( w < W && --_nb_remaining == 0 )
                _cv_done.notify_one();
        }
    }
}

CpuThreadPool &cpu_thread_pool() {
    static CpuThreadPool &pool = *new CpuThreadPool();
    return pool;
}

} // namespace sdot
