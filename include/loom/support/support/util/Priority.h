#pragma once

namespace sdot {

// priority tag for ordered SFINAE dispatch
template<int N> struct Priority : Priority<N-1> {
};

template<> struct Priority<0> {
};

} // namespace sdot
