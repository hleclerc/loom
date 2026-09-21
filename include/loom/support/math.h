#pragma once

// LES MATHÉMATIQUES D'UN KERNEL passent par `sdot::` (`sdot::sqrt`, `sdot::exp`, ...), jamais par
// `std::` directement : c'est le point unique où le device choisit son implémentation. Sur l'hôte
// c'est `std::` ; dans du code device CUDA ce sont les intrinsèques du toolkit (`::sqrtf`, ...),
// que `<cmath>` de nvcc expose sous les mêmes noms non qualifiés. Un fichier de `sdot/include`
// écrit dans `namespace sdot` peut donc appeler `sqrt( x )` tout court.
//
// Avec `atomic_add.h`, l'un des deux seuls endroits où un `#if` sur la cible est légitime.

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
