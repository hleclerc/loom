#pragma once

// A KERNEL'S MATH goes through `sdot::` (`sdot::sqrt`, `sdot::exp`, ...), never through
// `std::` directly: it is the single point where the device chooses its implementation. On the host
// it is `std::`; in CUDA device code it is the toolkit's intrinsics (`::sqrtf`, ...),
// which nvcc's `<cmath>` exposes under the same unqualified names. A file of `sdot/include`
// written in `namespace sdot` can therefore simply call `sqrt( x )`.
//
// Together with `atomic_add.h`, one of the only two places where an `#if` on the target is legitimate.

#include <cmath>

namespace sdot {

#ifdef __CUDA_ARCH__
using ::sqrt; using ::fabs; using ::exp; using ::log; using ::atan; using ::atan2; using ::erf;
using ::sin; using ::cos; using ::pow; using ::floor; using ::ceil; using ::fmin; using ::fmax;
#else
using std::sqrt; using std::fabs; using std::exp; using std::log; using std::atan; using std::atan2; using std::erf;
using std::sin; using std::cos; using std::pow; using std::floor; using std::ceil; using std::fmin; using std::fmax;
#endif

} // namespace sdot
