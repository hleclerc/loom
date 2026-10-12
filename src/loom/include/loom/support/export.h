#pragma once

// Marks an entry point of a unit (`loom.FfiModule`): kernels are compiled with hidden visibility, so what a
// kernel calls across a shared library must say so.
#ifndef LOOM_EXPORT
#if defined( _WIN32 )
    #define LOOM_EXPORT __declspec( dllexport )
#else
    #define LOOM_EXPORT __attribute__(( visibility( "default" ) ))
#endif
#endif
