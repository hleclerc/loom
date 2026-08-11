#pragma once

#include <SYCL/sycl.hpp>
#include <functional>
#include <vector>
#include <array>

namespace sdot {

/// RAII autour d'un `sycl::event` : « synchrone par défaut, asynchrone si on le gère ».
///
/// Tant qu'il n'est pas *consommé* (via `wait()`, `detach()` ou `take()` — p.ex. repris comme
/// dépendance d'un run suivant), sa destruction fait un `wait()`. On évite ainsi de lire des
/// résultats pas encore prêts par mégarde, sans imposer un `wait()` à chaque run : si l'appelant
/// garde le handle et l'enchaîne, aucun `wait()` n'est forcé.
///
/// `finalizers` est exécuté *après* le `wait()` (avant de marquer l'event consommé) : il sert p.ex.
/// à recopier le résultat d'une réduction depuis l'USM vers la variable hôte puis à libérer l'USM.
///
/// Move-only : un event = une responsabilité (un seul propriétaire à la fois).
struct QueueEvent {
    sycl::event                        event;
    bool                               consumed = false;
    std::vector<std::function<void()>> finalizers; ///< exécutés après le wait (recopie réductions, free USM)

    /* */       QueueEvent ( sycl::event e ) : event( e ) {}
    /* */       QueueEvent () = default;

    /* */       QueueEvent ( QueueEvent &&o ) noexcept : event( o.event ), consumed( o.consumed ), finalizers( std::move( o.finalizers ) ) { o.consumed = true; }
    QueueEvent& operator=  ( QueueEvent &&o ) noexcept {
        if ( this != &o ) {
            _finish();
            event = o.event; consumed = o.consumed; finalizers = std::move( o.finalizers ); o.consumed = true;
        }
        return *this;
    }
    /* */       QueueEvent ( const QueueEvent & ) = delete;
    QueueEvent& operator=  ( const QueueEvent & ) = delete;

    /* */       ~QueueEvent() { _finish(); }

    void        _finish    () { if ( consumed ) return; event.wait(); for ( auto &f : finalizers ) f(); finalizers.clear(); consumed = true; }

    void        wait       () { _finish(); }                     ///< attend explicitement la fin (et exécute les finalizers)
    void        detach     () { consumed = true; }               ///< « je m'en occupe » : pas de wait à la destruction
    sycl::event take       () { consumed = true; return event; } ///< récupère l'event (p.ex. comme dépendance d'un run suivant)
};

/// Jeu de dépendances (events) passé juste après `queue_list` pour chaîner les soumissions.
/// Produit par `after(...)`. Détectable par type via `is_dependencies`.
template<std::size_t N>
struct Dependencies {
    std::array<sycl::event,N> events;
};

template<class>             constexpr bool is_dependencies                  = false;
template<std::size_t N>     constexpr bool is_dependencies<Dependencies<N>> = true;

/// Construit les dépendances à partir de `QueueEvent` (qu'on *consomme* via take()) ou d'`sycl::event`.
auto after( auto &&...handles ) {
    return Dependencies<sizeof...( handles )>{ { handles.take()... } };
}

} // namespace sdot
