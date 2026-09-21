#pragma once

#include <functional>
#include <vector>
#include <array>

namespace sdot {

/// Ce qu'un lancement rend : « synchrone par défaut, asynchrone si on le gère ».
///
/// Tant qu'il n'est pas *consommé* (via `wait()`, `detach()` ou `take()` — p.ex. repris comme
/// dépendance d'un run suivant), sa destruction fait un `wait()`. On évite ainsi de lire des
/// résultats pas encore prêts par mégarde, sans imposer un `wait()` à chaque run : si l'appelant
/// garde le handle et l'enchaîne, aucun `wait()` n'est forcé.
///
/// Le device dit ce qu'attendre veut dire : `waiter` est vide sur CPU (le lancement a rendu la
/// main une fois le travail fait), et ce sera un `cudaEvent` à attendre sur GPU. Le reste --
/// consommation, finalizers -- est commun.
///
/// `finalizers` est exécuté *après* l'attente (avant de marquer l'event consommé) : il sert p.ex.
/// à recopier le résultat d'une réduction depuis le device vers la variable hôte.
///
/// Move-only : un event = une responsabilité (un seul propriétaire à la fois).
struct QueueEvent {
    std::function<void()>              waiter;            ///< vide = déjà complet
    bool                               consumed = false;
    std::vector<std::function<void()>> finalizers;        ///< exécutés après l'attente

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

    void        wait       () { _finish(); }               ///< attend explicitement la fin (et exécute les finalizers)
    void        detach     () { consumed = true; }         ///< « je m'en occupe » : pas de wait à la destruction

    /// récupère de quoi attendre (p.ex. comme dépendance d'un run suivant). Les finalizers sont
    /// emportés avec : ils s'exécuteront à la première attente.
    std::function<void()> take() {
        consumed = true;
        return [waiter=std::move( waiter ),finalizers=std::move( finalizers )]() mutable {
            if ( waiter ) waiter();
            for ( auto &f : finalizers ) f();
            finalizers.clear();
        };
    }
};

/// Jeu de dépendances passé juste après `queue_list` pour chaîner les soumissions.
/// Produit par `after(...)`. Détectable par type via `is_dependencies`.
template<std::size_t N>
struct Dependencies {
    std::array<std::function<void()>,N> waiters;
    void wait_all() const { for ( auto &w : waiters ) if ( w ) w(); }
};

template<class>             constexpr bool is_dependencies                  = false;
template<std::size_t N>     constexpr bool is_dependencies<Dependencies<N>> = true;

/// Construit les dépendances à partir de `QueueEvent` (qu'on *consomme* via take()).
auto after( auto &&...handles ) {
    return Dependencies<sizeof...( handles )>{ { handles.take()... } };
}

} // namespace sdot
