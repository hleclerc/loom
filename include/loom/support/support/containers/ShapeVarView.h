#pragma once

#include "../kernels/make_avaiable.h"
#include "../kernels/transfer_cost.h"
#include "../kernels/IoCategory.h"
#include "../common_types.h"
#include "ErrorBuffer.h"

namespace sdot {

/// View on a `ShapeVar`: a tensor of counts (rank 0 for a scalar count, rank > 0 for ragged
/// ones), the capacity bound `max`, and the error buffer of the call (plus `id`, which says who
/// we are in it).
///
/// Assigning writes the count -- a scalar broadcasts over all the cells of a ragged one -- and is
/// where the capacity check lives: `max` is what the tensors depending on this ShapeVar were
/// SIZED on, so a bigger count would send the body writing past them. Hence the clamp: the count
/// one can read back never exceeds the capacity (a body looping over its own count therefore
/// stays inside the buffers), while the count that was ASKED for goes to the error buffer, which
/// is what lets the host reserve more and run again.
///
/// `max == -1` means "unbounded": nothing sizes itself on this var, so this call had no reason to
/// be given a capacity for it, and there is nothing to check.
template<class View,class ErrBuf>
struct ShapeVarView {
    View   view;
    SI     max;
    ErrBuf errors;   ///< the call's error buffer -- shared with everything else that can fail
    SI     id;       ///< who we are in it (see `ErrorKind::capacity_overflow`)

    /// Write the count, capacity-checked, and say whether it FIT. On overflow it records the
    /// overflow once (atomic, into the call's error buffer) and clamps `view = max` -- so a body
    /// looping over the count read back stays inside the buffers -- then returns `false`, which
    /// lets the caller STOP GROWING / skip work instead of silently truncating. This is the one
    /// place a capacity is tested: the tensors stay dumb, unchecked views (see the class doc).
    bool set( auto v ) {
        const SI wanted = SI( v );
        if ( max >= 0 && wanted > max ) {
            errors.record( ErrorKind::capacity_overflow, id, wanted );
            view = max;
            return false;
        }
        view = wanted;
        return true;
    }

    /// Same write, bool discarded: for the sites that do not branch on whether it fit (the loops
    /// that then read the clamped count back as their bound are safe either way).
    // ShapeVarView &operator=( auto v ) {
    //     set( v );
    //     return *this;
    // }

    operator SI() const { return view.value(); }

    /// Select axes, by name or by position, exactly as on a tensor: a ShapeVar has axes too as
    /// soon as something gives it some -- a `vmap` makes it one count PER BATCH ITEM. The `max`
    /// follows the sub-view: a capacity is a bound on a count, whichever item it belongs to. So
    /// does the error buffer: there is ONE per call, not one per item (recording is atomic).
    ///
    /// Indexing by an empty multi-index is a no-op, so `nb_vertices( batch_index ) = 1` is the
    /// single spelling: unbatched, `batch_index` is the empty tuple and this gives back the very
    /// same view.
    auto operator()( auto &&...index ) const {
        return make_shape_var_view( view( FORWARD( index )... ), max, errors, id );
    }

    /// remise à zéro d'un compteur de sortie. Passe par la queue (donc par un kernel) : sur un
    /// device, le buffer n'est pas accessible depuis l'hôte -- ce n'est pas une boucle hôte.
    void fill_with( auto &&queue, auto v ) { view.fill_with( FORWARD( queue ), v ); }

    // comme argument de `run_parallel` : on rend la vue disponible, et on reconstruit la même
    // `ShapeVarView` autour (`max` et `id` sont des constantes, elles suivent la vue dans le
    // kernel). Le buffer d'erreurs est lu ET écrit (un compteur atomique) -> MutList.
    auto transfer_cost ( const auto &queue, auto io_category ) const {
        return sdot::transfer_cost( queue, io_category, view )
             + sdot::transfer_cost( queue, MutList(), errors );
    }

    auto make_available( auto &&queue, auto io_category, auto &&cont ) const {
        return sdot::make_available( queue, io_category, view, [&]( auto &&kernel_view ) {
            return sdot::make_available( queue, MutList(), errors, [&]( auto &&kernel_errors ) {
                return cont( ShapeVarView<DECAYED_TYPE_OF( kernel_view ),DECAYED_TYPE_OF( kernel_errors )>{
                    FORWARD( kernel_view ), max, FORWARD( kernel_errors ), id
                } );
            } );
        } );
    }
};

template<class View,class ErrBuf>
auto make_shape_var_view( View view, SI max, ErrBuf errors, SI id ) {
    return ShapeVarView<View,ErrBuf>{ view, max, errors, id };
}

} // namespace sdot
