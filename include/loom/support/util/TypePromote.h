#pragma once

#include "../containers/Void.h"
#include "../common_types.h"

namespace sdot {

template<class ...Args>
struct TypePromote;

template<class A,class B,class Head,class ...Tail>
struct TypePromote<A,B,Head,Tail...> {
    using type = typename TypePromote<typename TypePromote<A,B>::type,Head,Tail...>::type;
};

template<class A> struct TypePromote<A> { using type = A; };

// A type promoted with itself is itself. This is not redundant with the explicit pairs below:
// `SI` is `long long` while `SI64` is `std::int64_t`, and on LP64 Linux the latter is `long` --
// a DISTINCT type -- so `TypePromote<SI,SI>` matches nothing there and the kernel fails to
// compile (on macOS both are `long long`, which is why it only ever showed up on Linux).
template<class A> struct TypePromote<A,A> { using type = A; };

template<class A> struct TypePromote<A,Void> { using type = A; };
template<class A> struct TypePromote<Void,A> { using type = A; };
// <Void,Void> matches the three partial specializations above equally: spell it out, a full
// specialization wins over all of them.
template<> struct TypePromote<Void,Void> { using type = Void; };

template<> struct TypePromote<SI32,SI32> { using type = SI32; };
template<> struct TypePromote<SI32,SI64> { using type = SI64; };
template<> struct TypePromote<SI32,PI32> { using type = SI32; };
template<> struct TypePromote<SI32,PI64> { using type = SI64; };

template<> struct TypePromote<SI64,SI32> { using type = SI64; };
template<> struct TypePromote<SI64,SI64> { using type = SI64; };
template<> struct TypePromote<SI64,PI32> { using type = SI64; };
template<> struct TypePromote<SI64,PI64> { using type = SI64; };

template<> struct TypePromote<PI32,SI32> { using type = SI32; };
template<> struct TypePromote<PI32,SI64> { using type = SI64; };
template<> struct TypePromote<PI32,PI32> { using type = PI32; };
template<> struct TypePromote<PI32,PI64> { using type = PI64; };

template<> struct TypePromote<PI64,SI32> { using type = SI64; };
template<> struct TypePromote<PI64,SI64> { using type = SI64; };
template<> struct TypePromote<PI64,PI32> { using type = PI64; };
template<> struct TypePromote<PI64,PI64> { using type = PI64; };


} // namespace sdot
