#include "scaled.h"
#ifndef SCALE
#define SCALE 1
#endif
namespace sdot { int scaled( int v ) { return SCALE * v; } }
