"""Call a compiled kernel on the buffers a framework already holds, by ADDRESS -- the engine shared by
every driver that has no graph to put the call in (numpy, torch, and whatever comes next).

A kernel is generated once (`FfiSource._render_call`) and compiled against the plain-buffer FFI shim of
`loom/support/numpy_ffi`: its handler takes a call frame -- a few C structs -- that is filled here with
`ctypes`. What differs from one framework to the next is only how a value becomes an address and how
an output is born, and that is the whole of a `PointerAdapter`:

    PointerAdapter.operand( x, device )   a framework value -> `Operand`  ( address, shape, what keeps it alive )
    PointerAdapter.zeros( shape, dtype )  a new output, as the framework spells it ( `dtype`: a numpy dtype )
    PointerAdapter.operand_of_output( x ) the `Operand` of such an output

Nothing is copied unless the framework cannot give the address of what it holds. A strided input
(a transposed view, a slice, a broadcast) crosses as it is, with its own strides: the kernel is
generated to read them (`CallArg_Tensor.runtime_strides`, decided by `PointerAdapter.is_strided` on
the very value the call receives). A kernel that was generated for a dense buffer gets one -- an
adapter's `operand` is the one place where a value is conformed to that.

Jax is another matter and stays out of here: its kernels are XLA targets inside a traced program
(see `JaxFfi`), where there is no address to take.

Two things keep this ABI apart from the Jax one, on purpose:
* the compiled library is named after the shim (`np_` prefix, in the hash too), because a kernel
  built against XLA's header has another ABI -- the same source text must never find the other's binary;
* the precompiled catalogue is not consulted, for the same reason (it is built for XLA).
"""
from __future__ import annotations

import ctypes
import time
import sys

import numpy

from ..compilation import include_roots, journal, make_library
from ..compilation.build import KernelDir
from ..util.encode_base_62 import encode_base_62
from .FfiSource import call_signature, _render_call, _resolve_source
from ..compilation.generated_headers import headers_key, write_overlay

_HANDLER_SYMBOL = "sdot_ffi_entry"
_MAX_RANK = 8       # `XLA_FFI_Buffer::dims`

# a loaded handler per kernel name, for the life of the process: never `dlclose`d (a thread pool may
# still own threads in it when the process tears down)
_loaded: dict = {}


class _Buffer( ctypes.Structure ):
    _fields_ = [ ( "data", ctypes.c_void_p ), ( "rank", ctypes.c_int64 ), ( "dims", ctypes.c_int64 * _MAX_RANK ),
                 ( "strides_bytes", ctypes.c_int64 * _MAX_RANK ) ]


# `void *allocate( void *context, size_t nb_bytes, size_t alignment )`: the call's scratch pool
_ALLOCATE = ctypes.CFUNCTYPE( ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t )


class _CallFrame( ctypes.Structure ):
    _fields_ = [ ( "args", ctypes.POINTER( _Buffer ) ), ( "rets", ctypes.POINTER( _Buffer ) ), ( "attrs", ctypes.POINTER( ctypes.c_int64 ) ),
                 ( "stream", ctypes.c_void_p ), ( "allocate", _ALLOCATE ), ( "allocate_context", ctypes.c_void_p ) ]


def shim_include_dir() -> str:
    """The directory holding our `xla/ffi/api/ffi.h`, `-isystem` so it is the one `#include` finds."""
    from ..compilation import loom_include_root
    return str( loom_include_root() / "loom" / "support" / "numpy_ffi" )


def _lib_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _load( source: str, device, prefix: str, sources, code_name, signature, headers = None ):
    """The handler of this source, compiled (or reused from disk) and loaded."""
    # the key is what changes the binary: the source, how it is compiled -- and WHICH FFI it is
    # compiled against, so a pointer-FFI kernel never collides with a Jax one of the same text.
    # and the generated headers it was rendered with (`JaxFfi.compile_and_register`)
    name = "np_" + prefix + encode_base_62( f"{ source }|{ sources }|{ device.compiler.build_signature }|{ include_roots() }|numpy_ffi_stream"
                                            + headers_key( headers ) )
    if name in _loaded:
        journal.record_reuse()
        return _loaded[ name ]

    started = time.monotonic()
    kernel_dir = KernelDir( name )          # built privately, published by a rename
    with kernel_dir as work_dir:
        src_path = work_dir / f"kernel{ device.compiler.source_suffix() }"
        if not ( src_path.exists() and src_path.read_text() == source ):
            src_path.write_text( source )

        make_library(
            name + _lib_suffix(), [ src_path ], device,
            extra_flags = [ "-isystem", shim_include_dir() ],
            sources = [ ( _resolve_source( p ), dict( d ) ) for p, d in sources ],
            work_dir = work_dir,
            include_overlay = write_overlay( work_dir / "include", headers ) if headers else None,
        )
    lib_path = kernel_dir.final / ( name + _lib_suffix() )
    lib = ctypes.CDLL( str( lib_path ) )
    entry = getattr( lib, _HANDLER_SYMBOL )
    entry.restype = ctypes.c_void_p
    entry.argtypes = [ ctypes.POINTER( _CallFrame ) ]
    journal.record( name, code_name, signature, "compiled", time.monotonic() - started )

    _loaded[ name ] = entry
    return entry


class Operand:
    """A buffer as the kernel will read it: where it is, its extents, and what keeps it alive (the
    frame only holds addresses, so whatever owns the memory must outlive the call)."""

    __slots__ = ( "address", "shape", "strides_bytes", "owner" )

    def __init__( self, address: int, shape, owner, strides_bytes = None, itemsize = None ):
        """`strides_bytes`: the buffer's own, in bytes, when it is not dense row-major (then `itemsize`
        is not needed). Left out, the buffer is dense and its strides are those of its shape."""
        shape = tuple( int( n ) for n in shape )
        if len( shape ) > _MAX_RANK:
            raise ValueError( f"pointer ffi: a buffer of rank { len( shape ) } (the limit is { _MAX_RANK })" )
        if strides_bytes is None:
            strides_bytes, step = [], int( itemsize )
            for n in reversed( shape ):
                strides_bytes.append( step )
                step *= max( n, 1 )
            strides_bytes = tuple( reversed( strides_bytes ) )
        self.address = int( address )
        self.shape = shape
        self.strides_bytes = tuple( int( s ) for s in strides_bytes )
        self.owner = owner

    def as_buffer( self ) -> _Buffer:
        buffer = _Buffer( self.address, len( self.shape ) )
        for i, ( n, s ) in enumerate( zip( self.shape, self.strides_bytes ) ):
            buffer.dims[ i ] = n
            buffer.strides_bytes[ i ] = s
        return buffer


def foreign_operand( x, device, strided, own_framework ):
    """The `Operand` of `x` when it is held by ANOTHER framework than the adapter's, read in place -- or
    `None` when it is the adapter's own (it knows its values best). The address, shape and strides come off
    the buffer by our own means (`Buffer.memory`): no conversion, so no framework gets to decide on a copy.
    The only copy left is the kernel's own request: it was generated for a dense buffer and this one is not.
    Its device is its own, whatever the kernel's: that is what the view's memory space spells."""
    from ..tensor.storage.buffers import buffer_class_for
    cls = buffer_class_for( x )
    if cls is None or cls.framework in ( None, own_framework, "numpy" ):
        return None             # numpy is every adapter's to read (host memory), as before
    buffer = cls( x )
    memory = buffer.memory()
    itemsize = buffer.dtype.numpy_dtype.itemsize
    if not strided and not memory.is_dense( itemsize ):
        memory = cls( buffer.contiguous() ).memory()
    return Operand( memory.address, memory.shape, memory.owner, strides_bytes = memory.strides_bytes )


class PointerAdapter:
    """What a framework tells the engine: how a value of its own becomes an `Operand`, and how an
    output is allocated. One instance per framework, stateless."""

    # the framework this adapter is FOR: a value of another one is read in place (`foreign_operand`)
    framework = None

    def operand( self, x, device, strided = False ) -> Operand:
        """`x` -- an input of the call, as the framework holds it -- as an `Operand`, copied only if
        the kernel cannot read it where it is. `strided`: the kernel was generated to read the strides
        the buffer has, so `x` crosses as it is; otherwise it reads a dense row-major buffer."""
        raise NotImplementedError

    def is_strided( self, x ) -> bool:
        """Does `x` hold its data otherwise than dense row-major -- a transposed view, a slice, a
        broadcast? What the code generator asks, to spell a view that reads the strides."""
        return False

    def zeros( self, shape, dtype, device ):
        """A new output buffer of `shape` and `dtype` ( a numpy dtype: what `out_shape_dtype` says,
        the error buffer's too ), zero-filled."""
        raise NotImplementedError

    def empty( self, shape, dtype, device ):
        """The same, left uninitialized: what a scratch pool hands out."""
        return self.zeros( shape, dtype, device )

    def stream( self, device ):
        """The address of the stream a kernel of `device` is launched on, `None` on the CPU. It is the one the
        framework itself works on -- so that what it launched before is done when the kernel starts, and what
        the kernel launched is done when it reads the outputs -- unless that is the legacy default stream: a
        kernel may capture CUDA graphs on its stream (`cudaStreamBeginCapture`), which that one forbids. Then
        it is a stream of our own, and `order_before` / `order_after` do the ordering the framework's would."""
        return None

    def order_before( self, device ):
        """What the framework launched so far is done before what the kernel launches next (a no-op when the
        kernel is on the framework's own stream)."""

    def order_after( self, device ):
        """What the kernel launched is done before what the framework launches next (idem)."""

    def operand_of_output( self, x ) -> Operand:
        """The `Operand` of a buffer made by `zeros` -- always readable in place, by construction."""
        raise NotImplementedError


def run( code, ca, device, prefix, adapter: PointerAdapter, values = None ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result buffers )`, nothing written back.
    `values`: the inputs to read in place of the ones `ca` holds ( same order ) -- what a batched call
    is handed, whose buffers are not the ones the lowering was built on."""
    source, inputs, outputs, attrs, sources, headers = _render_call( code, ca, device )
    entry = _load( source, device, prefix, sources, code.name, call_signature( ca ), headers )

    # kept alive by these lists until the kernel is done: the frame only holds addresses
    if values is None:
        values = [ b.jax_input_array() for b in inputs ]
    ins  = [ foreign_operand( x, device, getattr( b, "runtime_strides", False ), adapter.framework )
             or adapter.operand( x, device, getattr( b, "runtime_strides", False ) ) for b, x in zip( inputs, values ) ]
    outs = [ adapter.zeros( *b.out_shape_dtype(), device ) for b in outputs ]
    out_operands = [ adapter.operand_of_output( o ) for o in outs ]

    # the call's scratch pool, if the kernel asks for one (`FfiCode( allocator = True )`, CUDA): memory of the
    # framework's own, kept here until the kernel is done
    scratch = []

    def allocate( _, nb_bytes, alignment ):
        try:
            block = adapter.empty( ( nb_bytes + alignment, ), numpy.uint8, device )
            scratch.append( block )
            adapter.order_before( device )          # what the kernels about to use it launch comes after the allocation
            address = adapter.operand_of_output( block ).address
            return ( address + alignment - 1 ) // alignment * alignment
        except Exception:
            return None

    args = ( _Buffer * max( 1, len( ins ) ) )( *[ o.as_buffer() for o in ins ] )
    rets = ( _Buffer * max( 1, len( outs ) ) )( *[ o.as_buffer() for o in out_operands ] )
    values = ( ctypes.c_int64 * max( 1, len( attrs ) ) )( *[ int( v ) for _, _, v in attrs ] )
    frame = _CallFrame( args, rets, values, adapter.stream( device ), _ALLOCATE( allocate ), None )

    adapter.order_before( device )
    try:
        error = entry( ctypes.byref( frame ) )
    finally:
        adapter.order_after( device )
    if error:
        raise RuntimeError( ctypes.string_at( error ).decode( errors = "replace" ) )
    return outputs, outs


def call( code, ca, device, prefix, adapter: PointerAdapter ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`, minus differentiation)."""
    outputs, results = run( code, ca, device, prefix, adapter )
    for buffer, result in zip( outputs, results ):
        buffer.jax_write_back( result )
