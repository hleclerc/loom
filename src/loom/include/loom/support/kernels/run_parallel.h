#pragma once

#include "../containers/Tuple.h" // IWYU pragma: export
#include "../common_macros.h" // IWYU pragma: export  (FORWARD, DECAYED_TYPE_OF)
#include "CpuQueue.h" // IWYU pragma: export

namespace sdot {

/// NOTE: nobody calls these two adapters anymore. They were used to inject into a functor
/// a launch geometry DECIDED IN PYTHON (`FfiCode( thread_cap = "..." )`, a C++
/// expression string rendered at the call site). The geometry is now written where it makes
/// sense -- methods of the functor, which `_run_kernel` detects directly (`requires`) -- so
/// the generated code passes the functor as is. They remain for a HAND-WRITTEN functor that
/// wants to set a cap without declaring a method.
///
/// Wrap a functor with an explicit `max_nb_threads` cap that `run_parallel` reads (`_run_kernel`) to
/// bound the launched work-items to `min( nb_items, cap )` -- so a body's PER-THREAD scratch is sized
/// on threads, not items. A generated body's LAMBDA cannot carry the hook, and a LOCAL struct cannot
/// have the templated `operator()`/`max_nb_threads` a kernel needs (C++ forbids member templates in a
/// local class); hence this wrapper lives at namespace scope. Stateless but for the int cap -> cheap
/// to copy into a kernel. `operator()` just forwards to the wrapped functor, so the optional
/// `thread_index`/`nb_threads` args (see the queue's `submit_kernel`) pass straight through.
template<class Func>
struct MaxThreads {
    int  cap;
    Func func;
    int  max_nb_threads( auto &&... ) const { return cap; }
    HD void operator()( auto &&...args ) const { func( FORWARD( args )... ); }
};
template<class Func>
MaxThreads<std::decay_t<Func>> with_max_threads( int cap, Func &&func ) {
    return { cap, FORWARD( func ) };
}

/// Same idea as `MaxThreads`, one level up: each launched work-ITEM becomes a work-GROUP of
/// `group_size` cooperating lanes (a cooperative launch instead of a flat one, see the queue's
/// `submit_kernel_grouped`) -- `cap` still bounds the number of GROUPS (`max_nb_threads`,
/// unchanged meaning: a body's per-group scratch is sized on concurrent groups, not items).
/// `local_elems` sizes a raw `int32` scratch SHARED by the group's lanes, that the body gets as
/// `local_scratch` -- deliberately a raw pointer-like view, not wrapped in a `Tensor`: the
/// cooperative algorithm (histogram/scan/scatter) lives entirely in the C++ body, this facility
/// only has to launch the groups and hand them the group handle + local memory. The body's
/// `group_index`/`local_index`/`local_size`/`group`/`local_scratch`/`sub_group` params pass
/// straight through.
template<class Func>
struct GroupKernel {
    int  cap;
    int  size;
    int  local_elems;
    Func func;
    int  max_nb_threads( auto &&... ) const { return cap; }
    int  group_size( auto &&... )     const { return size; }
    int  local_mem_elems( auto &&... ) const { return local_elems; }
    HD void operator()( auto &&...args ) const { func( FORWARD( args )... ); }
};
template<class Func>
GroupKernel<std::decay_t<Func>> with_group_kernel( int cap, int group_size, int local_elems, Func &&func ) {
    return { cap, group_size, local_elems, FORWARD( func ) };
}

/// call func for each list item, parallel way.
///   func may define directly (in method) or indirectly (via overloads) the limits in terms of nb threads, ...
///
/// The queue is selected according to the arguments
///
/// All objects are transformed into LocalMemory for the kernel
///
/// run_parallel( range(), []( auto idx, auto &&a, auto &&b, auto &&v ) { a = b; ... },
///   OutList(), a
///   InpList(), b, 34
/// )
/// `second` = item_list, or a `Dependencies` (via `after(...)`) followed by the item_list.
///
/// `queue_list` may also be a single queue (`run_parallel( queue, ... )`): this is the usual
/// case of a generated kernel, which has only one execution context -- the list is only useful when
/// there is a choice to make (the cheapest is then taken, transfers included).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest );

} // namespace sdot

#include "run_parallel.cxx" // IWYU pragma: export
