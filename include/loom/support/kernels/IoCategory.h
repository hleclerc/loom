#pragma once

#include <loom/support/common_macros.h> // HD

#include <type_traits>

namespace sdot {

struct UndefList { void display( auto &ds ) const { ds << "UndefList"; } };
struct OutList   { void display( auto &ds ) const { ds << "OutList"; } };
struct InpList   { void display( auto &ds ) const { ds << "InpList"; } };
struct MutList   { void display( auto &ds ) const { ds << "MutList"; } };

/// "Reduction" category: carries the operator (e.g. `plus<double>()`, see `Reducer.h`). The argument
/// that follows in `run_parallel` is the host target: the queue gives the body a private accumulator,
/// initialized to the identity, and combines it into the target once the kernel is done.
template<class Op>
struct RedList {
    Op   op;
    HD void display( auto &ds ) const { ds << "RedList"; }
};
template<class Op> RedList( Op ) -> RedList<Op>;

template<class T>  constexpr bool is_red_list               = false;
template<class Op> constexpr bool is_red_list<RedList<Op>>  = true;

/// An io POLICY: one category per member, for an argument that has several.
///
/// A plain tag (`InpList`, ...) describes a whole argument; it says nothing about an aggregate
/// one member of which is read while another is written. A policy is an aggregate of the SAME
/// SHAPE as the argument (the same member names), holding tags -- e.g. the `Cell_io` generated
/// beside a `Cell`:
///
///   run_parallel( queue, items, kernel, cell_io, cell );                 // member by member
///   run_parallel( queue, items, kernel, InpList(), cell );               // all read-only
///   run_parallel( queue, items, kernel, Cell_io{ InpList(), OutList() }, cell );
///
/// Io is thus a USE (this `run_parallel`), never a property of the data: two kernels chained over
/// one object may read and write different parts of it. Reading the policy is up to the argument
/// (it is the one that knows its members); here we only recognize it as a category.
///
/// A policy DECLARES itself rather than deriving from a base: it has to stay a pure aggregate, or
/// CTAD would stop deducing it (a base class counts as its first element, and one would have to
/// write `Cell_io{ {}, InpList(), ... }`).
template<class T> constexpr bool is_io_policy = requires { T::is_io_policy; };

/// true for any io category tag (`RedList<...>` and policies included).
template<class T> constexpr bool is_io_category =
    std::is_same_v<T,UndefList> || std::is_same_v<T,OutList> ||
    std::is_same_v<T,MutList>   || std::is_same_v<T,InpList>  || is_red_list<T> ||
    is_io_policy<T>;

/// Reduction target "mapped" by `run_parallel`: op + pointer to the host result variable.
/// Produced in place of `kernel_form` when the current category is a `RedList`.
template<class Op, class T>
struct ReductionTarget {
    Op  op;
    T  *host;
};

template<class T>          constexpr bool is_reduction_target                       = false;
template<class Op,class T> constexpr bool is_reduction_target<ReductionTarget<Op,T>> = true;

} // namespace sdot
