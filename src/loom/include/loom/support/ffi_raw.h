#pragma once

// =====================================================================================
// THE XLA FFI CALL FRAME, DECODED BY HAND.
//
// `xla::ffi::Ffi::Bind().Arg<...>()...To( impl )` builds, for EVERY kernel, a handler template specialised on its whole
// parameter list: a decoder per buffer ( dtype, rank and dimension checks ), the attribute lookup ( names sorted with a
// `std::sort` of `std::string` ), the error plumbing. Measured on a CPU kernel of sdot, that machinery is more than half
// of the object code. Here the frame is read by one small non-template function, and the typed parameters the kernel body
// is written against ( `ffi::BufferR2<ffi::F64>`, `ffi::Result<...>`, `ffi::Error` ) are thin wrappers around what it found:
// the same spelling, so the generated body is the same text.
//
// CPU only: no stream, no allocator -- those are `Ctx` of the typed API. The default for CPU kernels; `LOOM_RAW_FFI=0` turns it off ( `FfiSource` ).
// =====================================================================================

#include "xla/ffi/api/c_api.h"
#include <loom/support/ffi_body.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace loom_raw {

static_assert( F32::dtype == XLA_FFI_DataType_F32 && F64::dtype == XLA_FFI_DataType_F64 && S32::dtype == XLA_FFI_DataType_S32
            && S64::dtype == XLA_FFI_DataType_S64 && U32::dtype == XLA_FFI_DataType_U32 && U64::dtype == XLA_FFI_DataType_U64 );

/// ( the body returns its message only, `FfiSource._BODY_EXPORT`: a refusal of the scratch pool, `Scratch::refusal_message`, is
/// told by its words, and is RESOURCE_EXHAUSTED as with the typed API -- what XLA's own pool says when it is full )
inline XLA_FFI_Error *fail( const XLA_FFI_CallFrame *f, const char *message ) {
    static constexpr char refusal[] = "loom: the scratch pool refused ";
    XLA_FFI_Error_Create_Args a;
    a.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    a.extension_start = nullptr;
    a.message = message;
    a.errc = std::strncmp( message, refusal, sizeof( refusal ) - 1 ) == 0 ? XLA_FFI_Error_Code_RESOURCE_EXHAUSTED : XLA_FFI_Error_Code_INVALID_ARGUMENT;
    return f->api->XLA_FFI_Error_Create( &a );
}

/// The frame of an EXECUTE call: `na` buffers in, `nr` buffers out, then the int64 attributes named `names`.
/// `dtype[ i ]` / `rank[ i ]` are what each of the `na + nr` buffers must be; they are checked, and `bufs` receives the
/// address and the dimensions. Returns null when all is right.
/// JAX first asks a handler for its METADATA ( the version of the API it was written against, its traits ), through an
/// extension of the frame; nothing is executed then. Returns true when it was such a call.
inline bool answer_metadata( XLA_FFI_CallFrame *f ) {
    if ( f->extension_start == nullptr || f->extension_start->type != XLA_FFI_Extension_Metadata )
        return false;
    XLA_FFI_Metadata *m = reinterpret_cast<XLA_FFI_Metadata_Extension *>( f->extension_start )->metadata;
    m->api_version = XLA_FFI_Api_Version{ XLA_FFI_Api_Version_STRUCT_SIZE, nullptr, XLA_FFI_API_MAJOR, XLA_FFI_API_MINOR };
    m->traits = 0;
    return true;
}

inline XLA_FFI_Error *decode( const XLA_FFI_CallFrame *f, int na, int nr, int nt, const int *dtype, const int *rank,
                              Raw *bufs, const char *const *names, std::int64_t *attrs ) {
    if ( f->args.size != na || f->rets.size != nr )
        return fail( f, "loom: wrong number of buffers in the call" );
    for ( int i = 0; i < na + nr; ++i ) {
        const bool ret = i >= na;
        const int  j = ret ? i - na : i;
        if ( ret ? f->rets.types[ j ] != XLA_FFI_RetType_BUFFER : f->args.types[ j ] != XLA_FFI_ArgType_BUFFER )
            return fail( f, "loom: a call argument is not a buffer" );
        const XLA_FFI_Buffer *b = static_cast<const XLA_FFI_Buffer *>( ret ? f->rets.rets[ j ] : f->args.args[ j ] );
        if ( int( b->dtype ) != dtype[ i ] )
            return fail( f, "loom: a buffer has not the element type the kernel was compiled for" );
        if ( int( b->rank ) != rank[ i ] )
            return fail( f, "loom: a buffer has not the rank the kernel was compiled for" );
        bufs[ i ] = { b->data, b->dims, nullptr };
    }
    for ( int t = 0; t < nt; ++t ) {
        const std::size_t len = std::strlen( names[ t ] );
        int found = -1;
        for ( std::int64_t a = 0; a < f->attrs.size; ++a )
            if ( f->attrs.names[ a ]->len == len && std::memcmp( f->attrs.names[ a ]->ptr, names[ t ], len ) == 0 ) { found = int( a ); break; }
        if ( found < 0 || f->attrs.types[ found ] != XLA_FFI_AttrType_SCALAR )
            return fail( f, "loom: a scalar attribute of the call is missing" );
        const XLA_FFI_Scalar *s = static_cast<const XLA_FFI_Scalar *>( f->attrs.attrs[ found ] );
        if ( s->dtype != XLA_FFI_DataType_S64 )
            return fail( f, "loom: a scalar attribute of the call is not an int64" );
        attrs[ t ] = *static_cast<const std::int64_t *>( s->value );
    }
    return nullptr;
}

/// The context of a call on a card: XLA's stream, and its memory pool. What the pool gave is handed back when the call is over -- as
/// `xla::ffi::ScratchAllocator` does. Build it, call `begin()`, and give the body `&context`.
struct GpuScope {
    const XLA_FFI_CallFrame *frame;
    Context                  context;
    std::vector<std::pair<void *,std::size_t>> blocks;

    explicit GpuScope( const XLA_FFI_CallFrame *f ) : frame( f ) {
        context.allocate = &GpuScope::allocate;
        context.free = &GpuScope::free;
        context.allocate_context = this;
    }
    GpuScope( const GpuScope & ) = delete;

    XLA_FFI_Error *begin() {
        XLA_FFI_Stream_Get_Args a;
        a.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
        a.extension_start = nullptr;
        a.ctx = frame->ctx;
        a.stream = nullptr;
        if ( XLA_FFI_Error *e = frame->api->XLA_FFI_Stream_Get( &a ) )
            return e;
        context.stream = a.stream;
        return nullptr;
    }

    static void *allocate( void *self, std::size_t nb_bytes, std::size_t alignment ) {
        GpuScope *g = static_cast<GpuScope *>( self );
        XLA_FFI_DeviceMemory_Allocate_Args a;
        a.struct_size = XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
        a.extension_start = nullptr;
        a.ctx = g->frame->ctx;
        a.size = nb_bytes;
        a.alignment = alignment;
        a.data = nullptr;
        if ( XLA_FFI_Error *e = g->frame->api->XLA_FFI_DeviceMemory_Allocate( &a ) ) {
            XLA_FFI_Error_Destroy_Args d;
            d.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
            d.extension_start = nullptr;
            d.error = e;
            g->frame->api->XLA_FFI_Error_Destroy( &d );
            return nullptr;
        }
        g->blocks.emplace_back( a.data, nb_bytes );
        return a.data;
    }

    /// a block given back before the call returns ( a scratch replaced by a larger one ): XLA frees it now, and the scope forgets it
    static void free( void *self, void *ptr ) {
        GpuScope *g = static_cast<GpuScope *>( self );
        for ( std::size_t i = 0; i < g->blocks.size(); ++i )
            if ( g->blocks[ i ].first == ptr ) {
                g->free_block( g->blocks[ i ] );
                g->blocks.erase( g->blocks.begin() + i );
                return;
            }
    }

    void free_block( const std::pair<void *,std::size_t> &b ) {
        XLA_FFI_DeviceMemory_Free_Args a;
        a.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
        a.extension_start = nullptr;
        a.ctx = frame->ctx;
        a.size = b.second;
        a.data = b.first;
        if ( XLA_FFI_Error *e = frame->api->XLA_FFI_DeviceMemory_Free( &a ) ) {
            XLA_FFI_Error_Destroy_Args d;
            d.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
            d.extension_start = nullptr;
            d.error = e;
            frame->api->XLA_FFI_Error_Destroy( &d );
        }
    }

    ~GpuScope() {
        for ( auto &b : blocks )
            free_block( b );
    }
};

/// an error of the body, as XLA wants it
inline XLA_FFI_Error *to_xla( const XLA_FFI_CallFrame *f, const Error &e ) {
    return e.msg ? fail( f, e.msg ) : nullptr;
}

} // namespace loom_raw
