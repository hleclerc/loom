// The subset of the XLA FFI C++ API that a generated kernel spells, over plain host buffers.
//
// A kernel is generated against `xla/ffi/api/ffi.h` (see `drivers/FfiSource.py`): `ffi::BufferR2<ffi::F64>`
// arguments, `ffi::Result<...>` outputs, `ffi::Ffi::Bind().Arg<>().Ret<>().Attr<>()`, and a handler
// symbol defined by `XLA_FFI_DEFINE_HANDLER_SYMBOL`. With jax, that header is jaxlib's. WITHOUT jax
// (the numpy driver, `drivers/NumpyFfi.py`), this one stands in: same spelling, so the very same
// generated source compiles -- and the call frame is a few plain structs numpy can fill through ctypes.
//
// Only what the generated code uses is here: buffers (with their strides), int64 attributes, an error return,
// and the two things XLA's context gives a GPU handler -- the platform stream and a scratch allocator
// (`Ctx<>`) -- which the caller supplies in the frame: a framework that runs on a card hands over ITS
// stream and ITS memory pool (see `drivers/PointerFfi.py`). The buffers are wherever the framework put them.
#pragma once

#include <cmath>     // the real xla header drags it in; kernel bodies use `exp`, `M_PI`... unqualified
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <optional>
#include <utility>
#include <exception>
#include <type_traits>

// ---------------------------------------------------------------------------- the C ABI
// What Python hands over: for each buffer its address, rank, extents and BYTE strides (a buffer is
// read where the framework holds it, whatever its strides); attributes as int64.
// `strides_bytes` is not part of XLA's API: a kernel only reads it when it was generated for a
// strided input (see `CallArg_Tensor.runtime_strides`), which never happens against XLA's own header.
struct XLA_FFI_Buffer {
    void *  data;
    int64_t rank;
    int64_t dims[ 8 ];
    int64_t strides_bytes[ 8 ];
};

struct XLA_FFI_CallFrame {
    const XLA_FFI_Buffer *args;     // the inputs, in `Bind()` order
    const XLA_FFI_Buffer *rets;     // the outputs, ditto
    const int64_t        *attrs;    // the attributes, ditto

    void *stream;                   // the platform stream the call is ordered on (a `cudaStream_t`), or null
    // the call's scratch pool: `allocate( context, nb_bytes, alignment )`, null when it refuses.
    // What it hands out lives until the call returns.
    void *( *allocate )( void *context, std::size_t nb_bytes, std::size_t alignment );
    void *allocate_context;
};

// what a failed call returns: a message, in storage of its own (one per thread, never freed)
struct XLA_FFI_Error {
    char message[ 1024 ];
};

namespace xla::ffi {

// ---------------------------------------------------------------------------- errors
enum class ErrorCode { kOk, kCancelled, kUnknown, kInvalidArgument, kNotFound, kInternal, kResourceExhausted };

class Error {
public:
    Error( ErrorCode code, std::string message ) : code_( code ), message_( std::move( message ) ) {}
    static Error Success() { return Error( ErrorCode::kOk, "" ); }
    bool success() const { return code_ == ErrorCode::kOk; }
    const std::string &message() const { return message_; }
private:
    ErrorCode code_;
    std::string message_;
};

// ---------------------------------------------------------------------------- element types
enum class DataType { F32, F64, S32, S64, U32, U64 };
inline constexpr DataType F32 = DataType::F32;
inline constexpr DataType F64 = DataType::F64;
inline constexpr DataType S32 = DataType::S32;
inline constexpr DataType S64 = DataType::S64;
inline constexpr DataType U32 = DataType::U32;
inline constexpr DataType U64 = DataType::U64;

template<DataType D> struct NativeType;
template<> struct NativeType<DataType::F32> { using type = float; };
template<> struct NativeType<DataType::F64> { using type = double; };
template<> struct NativeType<DataType::S32> { using type = std::int32_t; };
template<> struct NativeType<DataType::S64> { using type = std::int64_t; };
template<> struct NativeType<DataType::U32> { using type = std::uint32_t; };
template<> struct NativeType<DataType::U64> { using type = std::uint64_t; };

// ---------------------------------------------------------------------------- buffers
struct Dimensions {
    const int64_t *p;
    std::size_t    n;
    const int64_t &operator[]( std::size_t i ) const { return p[ i ]; }
    std::size_t    size() const { return n; }
    const int64_t *begin() const { return p; }
    const int64_t *end() const { return p + n; }
};

template<DataType D, std::size_t R>
struct Buffer {
    using T = typename NativeType<D>::type;

    explicit Buffer( const XLA_FFI_Buffer &b ) : b_( &b ) {}

    T *typed_data() const { return static_cast<T *>( b_->data ); }
    Dimensions dimensions() const { return { b_->dims, R }; }
    int64_t stride_bytes( std::size_t d ) const { return b_->strides_bytes[ d ]; }
    std::size_t element_count() const { std::size_t n = 1; for ( std::size_t i = 0; i < R; ++i ) n *= b_->dims[ i ]; return n; }

private:
    const XLA_FFI_Buffer *b_;
};

template<DataType D> using BufferR0 = Buffer<D,0>;
template<DataType D> using BufferR1 = Buffer<D,1>;
template<DataType D> using BufferR2 = Buffer<D,2>;
template<DataType D> using BufferR3 = Buffer<D,3>;
template<DataType D> using BufferR4 = Buffer<D,4>;
template<DataType D> using BufferR5 = Buffer<D,5>;
template<DataType D> using BufferR6 = Buffer<D,6>;
template<DataType D> using BufferR7 = Buffer<D,7>;
template<DataType D> using BufferR8 = Buffer<D,8>;

// an output: pointer-like, as in XLA (`out->typed_data()`)
template<class B>
struct Result {
    explicit Result( B b ) : b_( b ) {}
    B *operator->() { return &b_; }
    const B *operator->() const { return &b_; }
    B &operator*() { return b_; }
private:
    B b_;
};

// ---------------------------------------------------------------------------- the call's context
// What `Bind().Ctx<...>()` gives a handler, as in XLA: the stream the call is ordered on...
template<class Stream> struct PlatformStream {};

// ... and an allocator whose memory lives until the call returns.
struct ScratchAllocator {
    void *( *allocate )( void *context, std::size_t nb_bytes, std::size_t alignment );
    void *context;

    std::optional<void *> Allocate( std::size_t nb_bytes, std::size_t alignment ) const {
        void *p = allocate ? allocate( context, nb_bytes, alignment ) : nullptr;
        if ( p == nullptr )
            return std::nullopt;
        return p;
    }
};

// ---------------------------------------------------------------------------- the binding
namespace detail {
    enum class Kind { arg, ret, attr, ctx, none };

    template<class C> struct CtxOf;
    template<class S> struct CtxOf<PlatformStream<S>> {
        static S get( const XLA_FFI_CallFrame &f ) { return reinterpret_cast<S>( f.stream ); } };
    template<> struct CtxOf<ScratchAllocator> {
        static ScratchAllocator get( const XLA_FFI_CallFrame &f ) { return { f.allocate, f.allocate_context }; } };

    template<class B> struct ArgTag  { static constexpr Kind kind = Kind::arg;
        static B get( const XLA_FFI_CallFrame &f, std::size_t i ) { return B( f.args[ i ] ); } };
    template<class B> struct RetTag  { static constexpr Kind kind = Kind::ret;
        static Result<B> get( const XLA_FFI_CallFrame &f, std::size_t i ) { return Result<B>( B( f.rets[ i ] ) ); } };
    template<class T> struct AttrTag { static constexpr Kind kind = Kind::attr;
        static T get( const XLA_FFI_CallFrame &f, std::size_t i ) { return static_cast<T>( f.attrs[ i ] ); } };
    template<class C> struct CtxTag  { static constexpr Kind kind = Kind::ctx;
        static auto get( const XLA_FFI_CallFrame &f, std::size_t ) { return CtxOf<C>::get( f ); } };

    // how many tags before position `i` are of the same kind: the index into that kind's list
    template<class... Tags>
    constexpr std::size_t slot( std::size_t i ) {
        constexpr Kind kinds[] = { Tags::kind..., Kind::none };
        std::size_t n = 0;
        for ( std::size_t j = 0; j < i; ++j )
            n += kinds[ j ] == kinds[ i ];
        return n;
    }

    inline XLA_FFI_Error *fail( const std::string &message ) {
        thread_local XLA_FFI_Error error;
        std::snprintf( error.message, sizeof( error.message ), "%s", message.c_str() );
        return &error;
    }

    template<class... Tags>
    struct Binding {
        // The parameters, in the order they are bound. Attributes carry a name in XLA, which matches by
        // name; here they match by position -- the generator lists both in the same order.
        template<class B> Binding<Tags..., ArgTag<B>>  Arg() const { return {}; }
        template<class B> Binding<Tags..., RetTag<B>>  Ret() const { return {}; }
        template<class T> Binding<Tags..., AttrTag<T>> Attr( const char * ) const { return {}; }
        template<class C> Binding<Tags..., CtxTag<C>>  Ctx() const { return {}; }

        template<class Fn>
        XLA_FFI_Error *Call( Fn fn, XLA_FFI_CallFrame *frame ) const {
            try {
                return call( fn, *frame, std::index_sequence_for<Tags...>{} );
            } catch ( const std::exception &e ) {
                return fail( std::string( "loom: the kernel threw: " ) + e.what() );
            } catch ( ... ) {
                return fail( "loom: the kernel threw" );
            }
        }

    private:
        template<class Fn, std::size_t... I>
        XLA_FFI_Error *call( Fn fn, const XLA_FFI_CallFrame &frame, std::index_sequence<I...> ) const {
            Error error = fn( Tags::get( frame, slot<Tags...>( I ) )... );
            return error.success() ? nullptr : fail( error.message() );
        }
    };
} // namespace detail

struct Ffi {
    static detail::Binding<> Bind() { return {}; }
};

} // namespace xla::ffi

// `XLA_FFI_DEFINE_HANDLER_SYMBOL( name, fn, binding )`: the extern "C" entry `name( frame )`, which
// decodes the frame as `binding` says and calls `fn`. Returns null on success, else the error.
#define XLA_FFI_DEFINE_HANDLER_SYMBOL( name, fn, bindings )                                       \
    extern "C" XLA_FFI_Error *name( XLA_FFI_CallFrame *call_frame ) {                              \
        return ( bindings ).Call( fn, call_frame );                                                \
    }
