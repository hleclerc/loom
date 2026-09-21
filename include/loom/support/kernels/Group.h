#pragma once

#include <barrier>

namespace sdot {

/// Le groupe de voies d'un kernel coopératif, vu du CPU : `group_size` fils système autour d'un
/// `std::barrier` (nul quand le groupe est réduit à une voie -- le cas de production sur CPU, où
/// `group_barrier` ne fait alors rien).
///
/// Ce que le corps peut en faire est le SOUS-ENSEMBLE commun aux devices : `group_barrier`,
/// `get_local_linear_range`. La version CUDA (`CudaGroup`) exposera le même contrat sur
/// `__syncthreads`.
struct CpuGroup {
    int get_local_linear_range() const { return size; }

    std::barrier<> *barrier; ///< nul si `size == 1`
    int             size;
};

inline void group_barrier( const CpuGroup &group ) {
    if ( group.barrier )
        group.barrier->arrive_and_wait();
}

/// Le sous-groupe (le warp) d'une voie. Sur CPU chaque voie est son propre sous-groupe : rien
/// n'y exécute en lockstep, donc `get_local_linear_range() == 1` et la barrière est vide. Un
/// corps qui coopère par sous-groupe doit dégénérer correctement à cette largeur (voir
/// `OtPlan1d.cxx::sort_diracs`).
struct CpuSubGroup {
    int get_local_linear_id   () const { return 0; }
    int get_local_linear_range() const { return size; }
    int get_group_linear_id   () const { return lane; }

    int lane;
    int size;
};

inline void group_barrier( const CpuSubGroup & ) {}

} // namespace sdot
