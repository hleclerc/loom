// THE CUDA SUPPORT UNIT: kernels that every CUDA call would otherwise instantiate for itself, compiled ONCE as a shared library and
// linked by the calls that want it ( `LOOM_CUDA_SUPPORT_UNIT`, see `CudaQueue.h` ).
//
// A fill is the same machine code whatever the shapes and axis names of the view it fills: it writes the bytes of a value at the
// addresses a shape and byte strides describe. Typed as `TensorView<...>` it was one 1 to 10 kB kernel per call and per view
// type -- identical in all of them. Here it is typed by the size of the element only.

#include <loom/support/export.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace {

constexpr int max_rank = 8;

struct Layout {
    long long dims   [ max_rank ];
    long long strides[ max_rank ];     // in bytes
};

template<class T>
__global__ void fill_contiguous_kernel( T *p, long long n, T v ) {
    for ( long long i = blockIdx.x * static_cast<long long>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<long long>( gridDim.x ) * blockDim.x )
        p[ i ] = v;
}

template<class T>
__global__ void fill_strided_kernel( char *p, int rank, Layout layout, long long n, T v ) {
    for ( long long i = blockIdx.x * static_cast<long long>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<long long>( gridDim.x ) * blockDim.x ) {
        long long r = i, offset = 0;
        for ( int k = rank - 1; k >= 0; --k ) {
            offset += ( r % layout.dims[ k ] ) * layout.strides[ k ];
            r /= layout.dims[ k ];
        }
        *reinterpret_cast<T *>( p + offset ) = v;
    }
}

int grid_for( long long n ) {
    const long long blocks = ( n + 255 ) / 256;
    return int( blocks < 65535 ? ( blocks > 0 ? blocks : 1 ) : 65535 );
}

template<class T>
int contiguous( cudaStream_t stream, void *ptr, long long n, const void *value ) {
    T v;
    std::memcpy( &v, value, sizeof v );
    fill_contiguous_kernel<T><<<grid_for( n ), 256, 0, stream>>>( static_cast<T *>( ptr ), n, v );
    return int( cudaGetLastError() );
}

template<class T>
int strided( cudaStream_t stream, void *ptr, int rank, const long long *dims, const long long *strides_bytes, const void *value ) {
    T v;
    std::memcpy( &v, value, sizeof v );
    Layout layout{};
    long long n = 1;
    for ( int k = 0; k < rank; ++k ) {
        layout.dims[ k ] = dims[ k ];
        layout.strides[ k ] = strides_bytes[ k ];
        n *= dims[ k ];
    }
    if ( n > 0 )
        fill_strided_kernel<T><<<grid_for( n ), 256, 0, stream>>>( static_cast<char *>( ptr ), rank, layout, n, v );
    return int( cudaGetLastError() );
}

} // namespace

/// Fill `n` contiguous elements of `elem_size` bytes ( 1, 2, 4 or 8 ) with the bytes at `value`. Returns a `cudaError_t` ( 0: success ),
/// or -1 for an element size it has no kernel for.
extern "C" LOOM_EXPORT int loom_cuda_fill_contiguous( void *stream, void *ptr, long long n, const void *value, int elem_size ) {
    cudaStream_t s = static_cast<cudaStream_t>( stream );
    switch ( elem_size ) {
    case 1: return contiguous<std::uint8_t >( s, ptr, n, value );
    case 2: return contiguous<std::uint16_t>( s, ptr, n, value );
    case 4: return contiguous<std::uint32_t>( s, ptr, n, value );
    case 8: return contiguous<std::uint64_t>( s, ptr, n, value );
    }
    return -1;
}

/// The same for a view: `rank` ( up to 8 ) extents and byte strides, in any axis order.
extern "C" LOOM_EXPORT int loom_cuda_fill_strided( void *stream, void *ptr, int rank, const long long *dims, const long long *strides_bytes, const void *value, int elem_size ) {
    if ( rank < 1 || rank > max_rank )
        return -1;
    cudaStream_t s = static_cast<cudaStream_t>( stream );
    switch ( elem_size ) {
    case 1: return strided<std::uint8_t >( s, ptr, rank, dims, strides_bytes, value );
    case 2: return strided<std::uint16_t>( s, ptr, rank, dims, strides_bytes, value );
    case 4: return strided<std::uint32_t>( s, ptr, rank, dims, strides_bytes, value );
    case 8: return strided<std::uint64_t>( s, ptr, rank, dims, strides_bytes, value );
    }
    return -1;
}
