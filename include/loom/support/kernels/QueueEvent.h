#pragma once

#include <functional>
#include <vector>
#include <array>

namespace sdot {

/// What a launch returns: "synchronous by default, asynchronous if you manage it".
///
/// As long as it has not been *consumed* (via `wait()`, `detach()` or `take()` — e.g. taken over as a
/// dependency of a following run), its destruction does a `wait()`. This avoids reading
/// results that are not ready yet by mistake, without imposing a `wait()` on every run: if the caller
/// keeps the handle and chains it, no `wait()` is forced.
///
/// The device says what waiting means: `waiter` is empty on CPU (the launch returned
/// once the work was done), and it will be a `cudaEvent` to wait on on GPU. The rest --
/// consumption, finalizers -- is common.
///
/// `finalizers` is run *after* the wait (before marking the event consumed): it serves e.g.
/// to copy the result of a reduction from the device back to the host variable.
///
/// Move-only: one event = one responsibility (a single owner at a time).
struct QueueEvent {
    std::function<void()>              waiter;            ///< empty = already complete
    bool                               consumed = false;
    std::vector<std::function<void()>> finalizers;        ///< run after the wait

    /* */       QueueEvent () = default;
    /* */       QueueEvent ( std::function<void()> waiter ) : waiter( std::move( waiter ) ) {}

    /* */       QueueEvent ( QueueEvent &&o ) noexcept : waiter( std::move( o.waiter ) ), consumed( o.consumed ), finalizers( std::move( o.finalizers ) ) { o.consumed = true; }
    QueueEvent& operator=  ( QueueEvent &&o ) noexcept {
        if ( this != &o ) {
            _finish();
            waiter = std::move( o.waiter ); consumed = o.consumed; finalizers = std::move( o.finalizers ); o.consumed = true;
        }
        return *this;
    }
    /* */       QueueEvent ( const QueueEvent & ) = delete;
    QueueEvent& operator=  ( const QueueEvent & ) = delete;

    /* */       ~QueueEvent() { _finish(); }

    void        _finish    () { if ( consumed ) return; if ( waiter ) waiter(); for ( auto &f : finalizers ) f(); finalizers.clear(); consumed = true; }

    void        wait       () { _finish(); }               ///< waits explicitly for completion (and runs the finalizers)
    void        detach     () { consumed = true; }         ///< "I take care of it": no wait on destruction

    /// takes what is needed to wait (e.g. as a dependency of a following run). The finalizers are
    /// carried along: they will run at the first wait.
    std::function<void()> take() {
        consumed = true;
        return [waiter=std::move( waiter ),finalizers=std::move( finalizers )]() mutable {
            if ( waiter ) waiter();
            for ( auto &f : finalizers ) f();
            finalizers.clear();
        };
    }
};

/// Set of dependencies passed right after `queue_list` to chain submissions.
/// Produced by `after(...)`. Detectable by type via `is_dependencies`.
template<std::size_t N>
struct Dependencies {
    std::array<std::function<void()>,N> waiters;
    void wait_all() const { for ( auto &w : waiters ) if ( w ) w(); }
};

template<class>             constexpr bool is_dependencies                  = false;
template<std::size_t N>     constexpr bool is_dependencies<Dependencies<N>> = true;

/// Builds the dependencies from `QueueEvent`s (which are *consumed* via take()).
auto after( auto &&...handles ) {
    return Dependencies<sizeof...( handles )>{ { handles.take()... } };
}

} // namespace sdot
