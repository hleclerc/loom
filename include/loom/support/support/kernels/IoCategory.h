#pragma once

#include <type_traits>

namespace sdot {

struct UndefList { void display( auto &ds ) const { ds << "UndefList"; } };
struct OutList   { void display( auto &ds ) const { ds << "OutList"; } };
struct InpList   { void display( auto &ds ) const { ds << "InpList"; } };
struct MutList   { void display( auto &ds ) const { ds << "MutList"; } };

/// Catégorie « réduction » : porte l'opérateur SYCL (p.ex. `sycl::plus<double>()`). L'argument qui
/// suit dans `run_parallel` est la cible hôte : `run_parallel` alloue l'USM, l'initialise à l'identité,
/// construit la `sycl::reduction`, et recopie le résultat dans la cible une fois le kernel terminé.
template<class Op>
struct RedList {
    Op   op;
    void display( auto &ds ) const { ds << "RedList"; }
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

/// Cible de réduction « mappée » par `run_parallel` : op SYCL + pointeur vers la variable hôte
/// résultat. Produite à la place de `make_available` quand la catégorie courante est un `RedList`.
template<class Op, class T>
struct ReductionTarget {
    Op  op;
    T  *host;
};

template<class T>          constexpr bool is_reduction_target                       = false;
template<class Op,class T> constexpr bool is_reduction_target<ReductionTarget<Op,T>> = true;

} // namespace sdot
