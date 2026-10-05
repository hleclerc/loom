#pragma once

#include "../containers/TensorView.h"
#include "../common_types.h"
#include <cstdlib>
#include <string>
#include <vector>

namespace sdot {

/// Memory allocated DURING the call, at a size that only the kernel knows.
///
/// We had assumed this was impossible: Jax preallocates most of the GPU's RAM, so there would be
/// nothing left. The premise is false -- and above all it concerns the wrong constraint.
/// What XLA requires at tracing time is the shape of an OUTPUT. An INTERNAL size, it never
/// asked for. So everything that is built and consumed within the same call ( an index, a sort
/// buffer, a CSR ) can be sized EXACTLY, with no guessed bound, under `jit` as in
/// eager. That is the difference with `scratch_attributes`, whose capacity comes down from Python.
///
/// WHERE THE MEMORY COMES FROM, and why the answer is not the same everywhere:
///
///   * on GPU, we need XLA's pool -- it is the one that owns the card. `XLA_FFI_DeviceMemory_
///     Allocate` is a field of the `XLA_FFI_Api` struct, hence of the stable C ABI.
///   * on CPU, no: the handler runs on the host, its buffers ARE host memory, and a
///     `malloc` does the job. This is not a stopgap -- XLA's CPU backend answers
///     "No device memory allocator available on this platform", and it is right: there is no
///     device.
///
/// The choice is the DEVICE's ( `Device.cpp_scratch_decl` ), not the body's: a body written once
/// works on both sides.
///
/// FAILURE IS A VALUE, not an exception: a pool can say no. `view()` then returns an EMPTY view
/// and raises `failed`, which the generated handler reports to XLA on exit. A body that bounds
/// its loops on THE VIEW -- and not on what it wanted to put in it -- then does nothing, which
/// is the intended behavior. A body that bounds on its intention writes through a null
/// pointer: that is exactly the segfault that revealed the CPU's refusal, so the mistake is
/// easy to make and worth stating here.
///
/// A REFUSAL IS FINAL for the call: once the pool said no, the next requests are refused HERE,
/// without asking it again. On a GPU the pool is XLA's BFC allocator, which does not say no at
/// once: it waits ( `AllocatorRetry`, ~10 s ) for memory that other streams may free -- and a body
/// that takes a dozen buffers before it checks them waited a dozen times ( a 1e7-seed solve spent
/// twenty minutes there before it raised ). The call has failed anyway; asking again only adds a wait.
///
/// The refusal's MESSAGE ( `refusal_message`, what the generated handler reports ): the size refused,
/// what the call had already taken, and `why` -- what the body says of itself ( what it computes,
/// for how many items, what to do ), which it may set at any time before it returns.
template<class _MemorySpace>
struct Scratch {
    using        MemorySpace = _MemorySpace;

    /// how we allocate, made opaque ON PURPOSE: this header does not know XLA ( it is included by
    /// hand-written headers, which have no business dragging in `xla/ffi/api/ffi.h` ). It is the
    /// generated source that plugs in these two pointers.
    using        AllocFn     = void *( * )( void *ctx, SI nb_bytes, SI alignment );
    /// `nullptr` when the pool frees by itself at the end of the call ( XLA's case ).
    using        FreeFn      = void ( * )( void *ctx, void *ptr );

    /* */        Scratch     ( AllocFn alloc_fn, void *ctx = nullptr, FreeFn free_fn = nullptr )
                                : alloc_fn( alloc_fn ), free_fn( free_fn ), ctx( ctx ) {}
    /* */        Scratch     ( const Scratch & ) = delete;
    Scratch&     operator=   ( const Scratch & ) = delete;

    /* */        ~Scratch    () {
        if ( free_fn )
            for ( void *p : blocks )
                free_fn( ctx, p );
    }

    /// `n` elements of type `T`, contiguous, NOT initialized -- like a `malloc`, and for the same
    /// reason: seeding costs, and the caller knows whether it writes before reading.
    template<class T>
    auto         view        ( SI n ) {
        T *ptr = nullptr;
        if ( n > 0 ) { // 0 is not allocated: nothing to ask for, hence nothing to refuse
            const SI nb_bytes = n * SI( sizeof( T ) );
            if ( ! failed ) // a refusal is final: the pool is not asked again ( see above )
                ptr = reinterpret_cast<T *>( alloc_fn( ctx, nb_bytes, SI( alignof( T ) ) ) );
            if ( ptr == nullptr ) {
                if ( ! failed )
                    refused = nb_bytes;
                failed = true;
                n = 0;
            } else {
                taken += nb_bytes;
                if ( free_fn )
                    blocks.push_back( ptr );
            }
        }
        return tensor_view<MemorySpace>( ptr, tuple( n ) );
    }

    /// what the handler reports when `failed`
    std::string  refusal_message() const {
        auto mb = []( SI b ) { return std::to_string( ( b + ( 1 << 19 ) ) >> 20 ) + " MB"; };
        std::string res = "loom: the scratch pool refused " + mb( refused ) + " ( " + std::to_string( refused ) +
                          " bytes ), after " + mb( taken ) + " taken by this call";
        if ( ! why.empty() )
            return res + ". " + why;
        return res + ". On a GPU the pool is XLA's ( at most `XLA_PYTHON_CLIENT_MEM_FRACTION` of the card, 0.75 by "
                     "default ); on the CPU backend, XLA's pool answers \"No device memory allocator available on this platform\"";
    }

    AllocFn             alloc_fn;
    FreeFn              free_fn;
    void               *ctx;
    std::vector<void *> blocks;          ///< empty when the pool frees by itself
    bool                failed = false; ///< at least one refusal ( read by the generated handler )
    SI                  taken = 0;      ///< the bytes given to this call
    SI                  refused = 0;    ///< the size of the request refused ( the first one: the others were not asked )
    std::string         why;            ///< the body's word on a refusal ( see `refusal_message` )
};

/// the scratch of a HOST handler: `aligned_alloc`, freed on exit. See the docstring above
/// for why this is not a stopgap on CPU.
template<class MemorySpace>
Scratch<MemorySpace> host_scratch() {
    return Scratch<MemorySpace>(
        []( void *, SI nb_bytes, SI alignment ) -> void * {
            // `aligned_alloc` wants a size that is a multiple of the alignment
            SI a = alignment < SI( sizeof( void * ) ) ? SI( sizeof( void * ) ) : alignment;
            SI s = ( nb_bytes + a - 1 ) / a * a;
            return std::aligned_alloc( size_t( a ), size_t( s ) );
        },
        nullptr,
        []( void *, void *ptr ) { std::free( ptr ); }
    );
}

} // namespace sdot
