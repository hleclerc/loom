#pragma once
// A function compiled SEPARATELY (`scaled.cpp`, one unit per value of `SCALE`) and linked into the
// kernel -- which is what `FfiCode( sources = ... )` routes through the compilation graph.
namespace sdot { int scaled( int v ); }
