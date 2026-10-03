#pragma once

#include <loom/support/atomic_add.h>
#include <loom/support/common_macros.h>
#include <type_traits>

/// The functors of the GPU-side `scratch` spike. They are here and not in the Python body because
/// a launched kernel needs a functor at NAMESPACE level ( C++ forbids template methods
/// in a local class ), and the body of a `FfiCode.handler` is a sequence
/// of statements inside the handler.
namespace loom_tests {

/// copies into the scratch what is positive ( zero otherwise )
struct KeepPositives {
    template<class T_tmp,class T_src>
    HD void operator()( auto id, T_tmp tmp, T_src src ) const {
        // NB the type is EXPLICIT: `auto v = src( id ); v > 0 ? v : 0` silently truncates --
        // the ternary looks for a common type between the accessor and the literal `0`, and finds
        // an integer. An easy and silent trap: the result stays plausible.
        double v = src( id );
        tmp( id ) = v > 0 ? v : 0.0;
    }
};

/// sums the scratch into the output. NB loom's `atomic_add` does not cover `SI` on CUDA
/// ( `atomicAdd` has no signed 64-bit overload ), hence a sum in `double`.
struct Summer {
    template<class T_out,class T_tmp>
    HD void operator()( auto id, T_out out, T_tmp tmp ) const {
        auto &o = out( 0 ).ref();
        sdot::atomic_add( o, std::remove_reference_t<decltype( o )>( tmp( id ) ) );
    }
};

/// counts, on the card, how many entries are positive
struct Counter {
    template<class T_counter,class T_src>
    HD void operator()( auto id, T_counter counter, T_src src ) const {
        double v = src( id );
        if ( v > 0 ) {
            auto &c = counter( 0 ).ref();
            sdot::atomic_add( c, std::remove_reference_t<decltype( c )>( 1 ) );
        }
    }
};

/// compacts the positives into `dst`, reserving its slot with an atomic ticket
struct Compactor {
    template<class T_dst,class T_counter,class T_src>
    HD void operator()( auto id, T_dst dst, T_counter counter, T_src src ) const {
        double v = src( id );
        if ( v > 0 ) {
            auto &c = counter( 0 ).ref();
            auto k = sdot::atomic_fetch_add( c, std::remove_reference_t<decltype( c )>( 1 ) );
            if ( k < dst.shape( 0 ) )
                dst( k ) = v;
        }
    }
};

/// puts a SCALAR coming from the host into an output. Passing a bare scalar to `run_parallel` is
/// exactly what made the nvcc frontend crash before `KernelFormProbe`.
struct Setter {
    template<class T_out,class T_v>
    HD void operator()( auto, T_out out, T_v v ) const { out( 1 ) = v; }
};

} // namespace loom_tests
