// The process-wide thread pool -- see `loom/support/kernels/CpuThreadPool.h`.
#include <loom/support/kernels/CpuThreadPool.h>
#include <chrono>
#include <cstdlib>
#if defined( __x86_64__ ) || defined( _M_X64 )
#include <immintrin.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace sdot {

/// HOW LONG A THREAD SPINS before it sleeps: the workers waiting for the next launch, the caller waiting for the end of a
/// launch. Long enough to bridge the gap between two loops of a solver ( a few µs ), short enough that a pool between two
/// calls of the user's program goes back to sleep at once. `SDOT_SPIN_US` overrides it ( 0: always sleep ).
static double spin_seconds() {
    static const double s = [] { const char *e = std::getenv( "SDOT_SPIN_US" ); return ( e ? std::atof( e ) : 100.0 ) * 1e-6; }();
    return s;
}

static inline void cpu_relax() {
#if defined( __x86_64__ ) || defined( _M_X64 )
    _mm_pause();
#elif defined( __aarch64__ )
    asm volatile( "yield" );
#endif
}

/// spins until `ready()` or `spin_seconds()` have passed; true if `ready()`
template<class F> static bool spin_until( F &&ready ) {
    if ( ready() ) return true;
    const double limit = spin_seconds();
    if ( limit <= 0 ) return false;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for ( int k = 1; ; ++k ) {
        cpu_relax();
        if ( ready() ) return true;
        if ( k % 64 == 0 && std::chrono::duration<double>( clock::now() - t0 ).count() > limit ) return false;
    }
}

CpuThreadPool::CpuThreadPool() {
    // The CPUs the process may run on: a Slurm allocation gives a few cores of a larger node, and
    // `hardware_concurrency` counts them all ( oversubscription ), while pinning to CPU `w` lands
    // outside the allocation ( `pthread_setaffinity_np` then fails, and the worker floats ).
#ifdef __linux__
    cpu_set_t allowed;
    if ( sched_getaffinity( 0, sizeof( allowed ), &allowed ) == 0 )
        for ( int c = 0; c < CPU_SETSIZE; ++c )
            if ( CPU_ISSET( c, &allowed ) )
                _cpus.push_back( c );
#endif
    _nb_workers = 0;
    if ( const char *env = std::getenv( "SDOT_NB_THREADS" ) )
        _nb_workers = std::atoi( env );
    if ( _nb_workers <= 0 )
        _nb_workers = _cpus.empty() ? int( std::thread::hardware_concurrency() ) : int( _cpus.size() );
    if ( _nb_workers <= 0 )
        _nb_workers = 1;
    if ( const char *env = std::getenv( "SDOT_PIN_THREADS" ) )
        _pin = std::atoi( env ) != 0;
}

CpuThreadPool::~CpuThreadPool() {
    {
        std::lock_guard<std::mutex> lock( _mutex );
        _stop = true;
        _launch.store( ( ( _launch.load( std::memory_order_relaxed ) >> 16 ) + 1 ) << 16 | _nb_workers, std::memory_order_release );
    }
    _cv_job.notify_all();
    for ( auto &th : _workers )
        th.join();
}

/// One launch at a time: the pool holds ONE job ( `_current_job`, `_nb_remaining`, `_launch` ), and several threads may launch
/// it at once -- XLA runs independent custom calls of one program on its own threads ( measured: two adjoints of a `jax.jit`
/// waited forever, each worker having taken the other's generation ). The callers queue here; a job still never relaunches
/// the pool ( not reentrant ).
static std::mutex &launch_mutex() {
    static std::mutex m;
    return m;
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
    std::lock_guard<std::mutex> launch( launch_mutex() );
    _ensure_workers();

    // the job is written BEFORE the launch moves ( release ): a spinning worker that sees the new launch sees the job. The
    // launch is moved under the mutex so that a worker going to sleep cannot miss it. Only the `W` workers with a slice
    // acknowledge it: the others read the launch word alone ( it carries `W` ), never the job, so the next launch may
    // rewrite the job as soon as the `W` are out -- a small loop on a node of 128 cores waits for its few workers, not for
    // all of them.
    _current_job        = &job;
    _current_nb_threads = nb_threads;
    _nb_remaining.store( W - 1, std::memory_order_relaxed );
    {
        std::lock_guard<std::mutex> lock( _mutex );
        _launch.store( ( ( _launch.load( std::memory_order_relaxed ) >> 16 ) + 1 ) << 16 | W, std::memory_order_release );
    }
    _cv_job.notify_all();

    // The calling thread runs slice 0. With pinning on it must sit on the first allowed CPU like the workers sit on theirs:
    // left floating it can share a core with a worker's hyperthread sibling, and since the slices are
    // equal the whole call waits for it ( measured: a bimodal +45 % on a 3D diagram ). Its mask is
    // restored afterwards: it is the user's thread, not ours.
#ifdef __linux__
    cpu_set_t previous;
    const bool pinned = _pin && sched_getaffinity( 0, sizeof( previous ), &previous ) == 0;
    if ( pinned ) {
        cpu_set_t set;
        CPU_ZERO( &set );
        CPU_SET( _cpu( 0 ), &set );
        sched_setaffinity( 0, sizeof( set ), &set );
    }
#endif
    _run_slice( 0, W, nb_threads, job );
#ifdef __linux__
    if ( pinned )
        sched_setaffinity( 0, sizeof( previous ), &previous );
#endif

    auto done = [&] { return _nb_remaining.load( std::memory_order_acquire ) == 0; };
    if ( ! spin_until( done ) ) {
        std::unique_lock<std::mutex> lock( _mutex );
        _cv_done.wait( lock, done );
    }
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
        CPU_SET( _cpu( w ), &set );
        pthread_setaffinity_np( pthread_self(), sizeof( set ), &set );
#endif
    }
    long seen = 0;
    for ( ;; ) {
        auto launched = [&] { return _launch.load( std::memory_order_acquire ) != seen; };
        if ( ! spin_until( launched ) ) {
            std::unique_lock<std::mutex> lock( _mutex );
            _cv_job.wait( lock, launched );
        }
        seen = _launch.load( std::memory_order_acquire );
        if ( _stop.load( std::memory_order_acquire ) )
            return;
        const int W = int( seen & 0xffff );
        if ( w >= W )
            continue;           // not in this launch: the job may already be the next launch's
        _run_slice( w, W, _current_nb_threads, *_current_job );
        // the last one out wakes the caller if it sleeps ( under the mutex: it cannot miss the notification )
        if ( _nb_remaining.fetch_sub( 1, std::memory_order_acq_rel ) == 1 ) {
            std::lock_guard<std::mutex> lock( _mutex );
            _cv_done.notify_one();
        }
    }
}

CpuThreadPool &cpu_thread_pool() {
    static CpuThreadPool &pool = *new CpuThreadPool();
    return pool;
}

} // namespace sdot
