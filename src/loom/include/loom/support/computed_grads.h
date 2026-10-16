#pragma once

#include <type_traits>

namespace sdot {

/// Whether a backward is ASKED for the pair `d output / d input`: the cotangent of the output may
/// be non-zero ( it is not a `ZeroTensor` / `NoneTensor` ), and the gradient of the input is wanted
/// ( it is not a `NoneTensor` ). Both are facts of the TYPES, so a variant knows at compile time
/// which of its `computed_grads` flags it has to find set ( see `Kernels.computed_grads` ).
template<class Cotangent, class Gradient>
constexpr bool grad_pair_is_asked() {
    using C = std::decay_t<Cotangent>;
    using G = std::decay_t<Gradient>;
    bool live = true, wanted = true;
    if constexpr ( requires { C::surely_null; } )
        live = ! C::surely_null;
    if constexpr ( requires { G::is_valid; } )
        wanted = G::is_valid;
    return live && wanted;
}

} // namespace sdot
