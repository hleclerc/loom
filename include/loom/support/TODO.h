#pragma once

#include <iostream> // IWYU pragma: export
#include <assert.h>

#define ERROR( MSG ) ( [&]() { std::cerr << __FILE__ << ":" << __LINE__ << ": ERROR: " << MSG << "\n"; exit( 1 ); } )()

#ifdef __CUDA_ARCH__
    #define TODO
    // __trap()
#else
    #define TODO throw std::runtime_error( std::string( __FILE__ ) + ":" + std::to_string( __LINE__ ) + ": not yet implemented" )
#endif
