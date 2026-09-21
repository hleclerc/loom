#pragma once
// Une fonction compilée À PART (`scaled.cpp`, une unité par valeur de `SCALE`) et liée dans le
// noyau -- ce que `FfiCode( sources = ... )` fait passer par le graphe de compilation.
namespace sdot { int scaled( int v ); }
