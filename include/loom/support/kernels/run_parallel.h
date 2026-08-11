#pragma once

#include "../containers/Tuple.h" // IWYU pragma: export
#include "../common_macros.h" // IWYU pragma: export  (FORWARD, DECAYED_TYPE_OF)
#include "CpuQueue.h" // IWYU pragma: export

namespace sdot {

/// Wrap a functor with an explicit `max_nb_threads` cap that `run_parallel` reads (`_run_kernel`) to
/// bound the launched work-items to `min( nb_items, cap )` -- so a body's PER-THREAD scratch is sized
/// on threads, not items. A generated body's LAMBDA cannot carry the hook, and a LOCAL struct cannot
/// have the templated `operator()`/`max_nb_threads` a kernel needs (C++ forbids member templates in a
/// local class); hence this wrapper lives at namespace scope. Stateless but for the int cap -> cheap
/// to capture by value into the SYCL kernel. `operator()` just forwards to the wrapped functor (plain,
/// non-`HD`, exactly like the generated lambda it wraps), so the optional `thread_index`/`nb_threads`
/// args (see `_do_submit`) pass straight through.
template<class Func>
struct MaxThreads {
    int  cap;
    Func func;
    int  max_nb_threads( auto &&... ) const { return cap; }
    void operator()( auto &&...args ) const { func( FORWARD( args )... ); }
};
template<class Func>
MaxThreads<std::decay_t<Func>> with_max_threads( int cap, Func &&func ) {
    return { cap, FORWARD( func ) };
}

/// Same idea as `MaxThreads`, one level up: each launched work-ITEM becomes a work-GROUP of
/// `group_size` cooperating work-items (a `sycl::nd_range` launch instead of a flat range) --
/// `cap` still bounds the number of GROUPS (`max_nb_threads`, unchanged meaning: a body's
/// per-group scratch is sized on concurrent groups, not items). `local_elems` sizes a raw
/// `int32` `sycl::local_accessor` (`_do_submit` builds it from the handler, see run_parallel.cxx)
/// that the body gets as `local_scratch` -- deliberately a raw SYCL local view, not wrapped in a
/// `Tensor`: the cooperative algorithm (histogram/scan/scatter) lives entirely in the C++ body,
/// this facility only has to launch the nd_range and hand it the group + local memory. `func`'s
/// optional `local_index`/`local_size`/`group`/`local_scratch` params (see `_do_submit`) pass
/// straight through, same opt-in `if constexpr(requires{...})` mechanism as `thread_index`.
template<class Func>
struct GroupKernel {
    int  cap;
    int  size;
    int  local_elems;
    Func func;
    int  max_nb_threads( auto &&... ) const { return cap; }
    int  group_size( auto &&... )     const { return size; }
    int  local_mem_elems( auto &&... ) const { return local_elems; }
    void operator()( auto &&...args ) const { func( FORWARD( args )... ); }
};
template<class Func>
GroupKernel<std::decay_t<Func>> with_group_kernel( int cap, int group_size, int local_elems, Func &&func ) {
    return { cap, group_size, local_elems, FORWARD( func ) };
}

/// call func for each list item, parallel way.
///   func may define directly (in method) or indirectly (via surdefinitions) the limits in terms of nb threads, ...
///
/// On sélectionne la sycl::queue en fonction des arguments
///
/// On transforme tous les objets en LocalMemory pour le kernel
///
/// run_parallel( range(), []( auto idx, auto &&a, auto &&b, auto &&v ) { a = b; ... },
///   OutList(), a
///   InpList(), b, 34
/// )
/// `second` = item_list, ou un `Dependencies` (via `after(...)`) suivi de l'item_list.
///
/// `queue_list` peut aussi être une queue seule (`run_parallel( queue, ... )`) : c'est le cas
/// courant d'un kernel généré, qui n'a qu'un contexte d'exécution -- la liste ne sert que quand
/// il y a un choix à faire (on prend alors le moins coûteux, transferts compris).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest );

} // namespace sdot

#include "run_parallel.cxx" // IWYU pragma: export
