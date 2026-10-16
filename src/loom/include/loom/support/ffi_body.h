#pragma once

// =====================================================================================
// WHAT A KERNEL BODY IS WRITTEN AGAINST: buffers as the frame described them, with no framework in sight.
//
// A generated kernel is two pieces. The BODY (`sdot_ffi_impl( ffi::BufferR2<ffi::F64>, ffi::Result<...>, ... )`, with everything it
// instantiates) is compiled ONCE, as a unit of its own, and exported as a plain function of `Raw` buffers and `int64` attributes.
// The ENTRY of each framework -- Jax's XLA frame ( `ffi_raw.h` ), the pointer frame of Torch, Cupy and Numpy -- is a few lines that
// decode what its framework hands over and call that function. So the same body serves them all, and a catalogue holds it once.
// The names are those of XLA's typed API (`ffi::BufferR2<ffi::F64>`, `ffi::Result`, `ffi::Error`): the body is written the same way.
// =====================================================================================

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace loom_raw {

enum class ErrorCode { kResourceExhausted, kFailedPrecondition };

/// the result of a handler: success, or a message ( copied: the caller's string may be a temporary )
struct Error {
    const char *msg = nullptr;
    Error() = default;
    Error( ErrorCode, const std::string &m ) { static thread_local std::string keep; keep = m; msg = keep.c_str(); }
    static Error Success() { return {}; }
};

// element types: the XLA tag and the C++ type
struct F32 { using T = float;         static constexpr int dtype = 11; };
struct F64 { using T = double;        static constexpr int dtype = 12; };
struct S32 { using T = std::int32_t;  static constexpr int dtype = 4; };
struct S64 { using T = std::int64_t;  static constexpr int dtype = 5; };
struct U32 { using T = std::uint32_t; static constexpr int dtype = 8; };
struct U64 { using T = std::uint64_t; static constexpr int dtype = 9; };

/// what the frame says about one buffer
/// What a card's call brings besides its buffers: the stream it is ordered on, and the pool the call may take memory from ( XLA's, or the
/// framework's ). Null on the CPU.
struct Context {
    void *stream = nullptr;
    void *( *allocate )( void *context, std::size_t nb_bytes, std::size_t alignment ) = nullptr;
    void *allocate_context = nullptr;
    /// gives a block back before the call returns ( a scratch replaced by a larger one, `ScratchBuffer.h` ); null when the pool cannot
    void  ( *free )( void *context, void *ptr ) = nullptr;
};

/// `xla::ffi::ScratchAllocator`, over a `Context`
class ScratchAllocator {
public:
    explicit ScratchAllocator( const Context *c ) : c( c ) {}
    std::optional<void *> Allocate( std::size_t nb_bytes, std::size_t alignment ) {
        void *p = c && c->allocate ? c->allocate( c->allocate_context, nb_bytes, alignment ) : nullptr;
        if ( ! p )
            return std::nullopt;
        return p;
    }
    /// `ptr` given back now, when the pool can ( a no-op otherwise: the block is then freed when the call returns )
    void Free( void *ptr ) {
        if ( c && c->free && ptr )
            c->free( c->allocate_context, ptr );
    }
private:
    const Context *c;
};

/// `strides_bytes`: the byte strides of the buffer, when the framework gives them (null: dense, as XLA's are)
struct Raw { void *data; const std::int64_t *dims; const std::int64_t *strides_bytes = nullptr; };

/// the dimensions, as `buffer.dimensions()[ d ]` reads them
struct Dims {
    const std::int64_t *p;
    std::int64_t operator[]( int d ) const { return p[ d ]; }
};

template<class E,int R>
struct BufferTyped {
    Raw raw;
    using T = typename E::T;
    T   *typed_data() const { return static_cast<T *>( raw.data ); }
    Dims dimensions() const { return { raw.dims }; }
    std::int64_t stride_bytes( std::size_t d ) const { return raw.strides_bytes[ d ]; }
};

template<class B> struct Result {
    B b;
    B       *operator->()       { return &b; }
    const B *operator->() const { return &b; }
};

// `ffi::BufferR2<ffi::F64>` : one alias per rank
template<class E> using BufferR0 = BufferTyped<E,0>;
template<class E> using BufferR1 = BufferTyped<E,1>;
template<class E> using BufferR2 = BufferTyped<E,2>;
template<class E> using BufferR3 = BufferTyped<E,3>;
template<class E> using BufferR4 = BufferTyped<E,4>;
template<class E> using BufferR5 = BufferTyped<E,5>;
template<class E> using BufferR6 = BufferTyped<E,6>;
template<class E> using BufferR7 = BufferTyped<E,7>;
template<class E> using BufferR8 = BufferTyped<E,8>;

} // namespace loom_raw
