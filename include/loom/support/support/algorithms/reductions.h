#pragma once

#include "../kernels/run_parallel.h" // run_parallel, RedList, InpList
#include "indices_of.h"
#include <SYCL/sycl.hpp>
#include <limits>

namespace sdot {

/// Réductions construites *au-dessus* de `run_parallel` : `RedList(op)` demande à `run_parallel`
/// d'allouer l'USM, de créer la `sycl::reduction` et de recopier le résultat dans la cible hôte.
/// Le `QueueEvent` temporaire renvoyé par `run_parallel` est détruit en fin d'expression : il attend
/// la fin du kernel puis exécute son finalizer (recopie dans `res`) -> `res` est prêt au `return`.

auto sum( auto &&queue_list, auto &&a ) {
    using TF = typename DECAYED_TYPE_OF( a )::TF;
    TF res = 0;
    run_parallel( FORWARD( queue_list ), indices_of( a ),
        []( auto idx, auto &r, auto a ) { r.combine( a[ idx ].value() ); },
        RedList( sycl::plus<TF>() ), res, InpList(), a );
    return res;
}

auto max( auto &&queue_list, auto &&a ) {
    using TF = typename DECAYED_TYPE_OF( a )::TF;
    TF res = std::numeric_limits<TF>::lowest();
    run_parallel( FORWARD( queue_list ), indices_of( a ),
        []( auto idx, auto &r, auto a ) { r.combine( a[ idx ].value() ); },
        RedList( sycl::maximum<TF>() ), res, InpList(), a );
    return res;
}

} // namespace sdot
