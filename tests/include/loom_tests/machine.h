#pragma once

#include <loom/support/kernels/run_parallel.h>
#include <loom/support/algorithms/CartesianIndices.h>
#include <loom/support/kernels/IoCategory.h>

/// De quoi EPROUVER `Machine` : un noyau qui recopie ses quatre champs dans une sortie, pour que
/// Python puisse les regarder. Passe par un kernel et non par l'hote, parce que sur carte la sortie
/// est en memoire device.
namespace loom_tests {

using sdot::operator""_c;

struct PoserMachine {
    HD void operator()( auto item, auto out, auto m ) const {
        const sdot::SI k = item[ 0_c ];
        out( k ) = k == 0 ? m.nb_workers
                 : k == 1 ? m.sub_group_width
                 : k == 2 ? m.local_mem_bytes
                 :          m.suggested_group;
    }
};

void poser_machine( auto &a ) {
    sdot::run_parallel( a.queue, sdot::indices_over( sdot::SI( 4 ) ), PoserMachine{},
                        a.champs_io, a.champs,
                        sdot::InpList(), a.machine );
}

} // namespace loom_tests
