#pragma once
// A function compiled SEPARATELY (`scaled.cpp`, one unit per value of `SCALE`) and linked into the
// kernel -- which is what `FfiCode( sources = ... )` routes through the compilation graph.
#include <loom/support/export.h>
namespace sdot { LOOM_EXPORT int scaled( int v ); }
