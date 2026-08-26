from typing_extensions import Optional, overload
from typing import TYPE_CHECKING, cast, Any
from numpy._typing import ArrayLike

# from jax._src.custom_derivatives import CustomVJPPrimal
# from jax._src import ad_util

from .CallArgsAnalysis import CallArgsAnalysis

from .JaxFramework import JaxFramework
from ..devices.Device import Device
from ..tensor.Dtype import Dtype

from ..compilation.FfiCode import FfiCode
from ..util.info import info
from .JaxFfi import call_body, call as ffi_call

# from textwrap import dedent, indent
# import importlib
# import numpy
# import re

import jax.core as jax_core
import jax.numpy as jnp
import jax


# def _has_tracer( data ) -> bool:
#     if isinstance( data, jax_core.Tracer ):
#         return True
#     if isinstance( data, ( list, tuple ) ):
#         return any( _has_tracer( x ) for x in data )
#     return False


class JaxDriver:
    """
    JAX implementation for sdot centralization.
    """

    if TYPE_CHECKING:
        device: Any

    def __init__( self, framework: JaxFramework, device: Device | None, ftype: Dtype | None, itype: Dtype | None ):
        if device is None:
            device = JaxDriver.default_device_for( ftype )
        if itype is None:
            itype = JaxDriver.default_itype_for( device )
        if ftype is None:
            ftype = JaxDriver.default_ftype_for( device )

        #
        self.framework = framework
        self.device = device
        self.ftype  = ftype
        self.itype  = itype

        # fill device_type for ftype and itype
        itype._driver_version = self.driver_dtype_version( itype.kind, itype.size )
        ftype._driver_version = self.driver_dtype_version( ftype.kind, ftype.size )
        assert itype.floating_point == False
        assert ftype.floating_point == True
        if itype.size == 64:
            jax.config.update( "jax_enable_x64", True )

        #
        device.driver_version = device.driver_version_for_jax( jax.devices )

    def driver_dtype_version( self, kind, size ):
        """The jax dtype a `( kind, size )` denotes. `size is None` means "the driver's own",
        which is exactly what `TF` / `TI` are -- resolved here, once the driver exists."""
        from ..tensor.Dtype import REAL, SINT, UINT, BOOL

        if kind == BOOL:
            return jnp.bool_

        if kind == REAL:
            if size is None:
                return self.ftype._driver_version
            try:
                return { 16: jnp.float16, 32: jnp.float32, 64: jnp.float64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported ftype size: { size }" )

        if kind == SINT:
            if size is None:
                return self.itype._driver_version
            try:
                return { 8: jnp.int8, 16: jnp.int16, 32: jnp.int32, 64: jnp.int64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported itype size: { size }" )

        assert kind == UINT
        try:
            return { 8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64 }[ size ]
        except KeyError:
            raise ValueError( f"unsupported unsigned itype size: { size }" )

    def dtype_of( self, x ):
        """The `Dtype` a jax buffer ACTUALLY has -- what a declaration is checked against."""
        from ..tensor.Dtype import Dtype
        aval = getattr( x, "aval", None )          # a SymbolicZero carries its type in its aval
        return Dtype.from_numpy( ( aval if aval is not None else x ).dtype )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a `Dtype`). Tracer-safe, and a no-op when it already is."""
        from ..tensor.Dtype import Dtype
        return jnp.asarray( x, dtype = Dtype.factory( dtype ).driver_version )

    @staticmethod
    def default_device_for( ftype ):
        platforms = { d.platform for d in jax.devices() }
        if "gpu" in platforms:
            from ..devices.CudaGpu import CudaGpu
            return CudaGpu( 0 )

        # Metal (jax-metal) — auto-select when available; always uses FP32
        if "METAL" in platforms and ftype in ( None, "FP32" ):
            from ..devices.AppleGpu import AppleGpu
            return AppleGpu()

        from ..devices.Cpu import Cpu
        return Cpu()

    @property
    def available_gpus( self ):
        res = 0
        for device in jax.devices():
            res += "gpu" in device.platform
        return res

    @staticmethod
    def default_ftype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.fp( 32 )
        return Dtype.fp( 64 )

    @staticmethod
    def default_itype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.si( 32 )
        return Dtype.si( 64 )

    if TYPE_CHECKING:
        @overload
        def array( self, data: ArrayLike, dtype = None, device = None ) -> jax.Array: ...
        @overload
        def array( self, data: None, dtype = None, device = None ) -> None: ...

    def array( self, data, dtype = None, device = None ):
        if data is None:
            return None
        dtype_ver = Dtype.factory( dtype or self.ftype ).driver_version
        if _has_tracer( data ):
            return jnp.asarray( data, dtype = dtype_ver )
        # device = self.device.driver_version
        return jnp.asarray( data, dtype = dtype_ver )

    # functional building blocks (tracer-safe, differentiable): used to assemble
    # padded buffers without in-place mutation, which does not fit Jax.
    def zeros( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def full( self, shape, value, dtype = None ):
        return jnp.full( shape, value, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def stack( self, tensors, axis = 0 ):
        return jnp.stack( tensors, axis = axis )

    def pad( self, tensor, pad_width ):
        return jnp.pad( tensor, pad_width )

    def transpose( self, a, axes = None ):
        return jnp.transpose( a, axes )

    def vmap( self, func ):
        """Map `func` over a new leading axis. A `driver.call` inside it is not replayed item by
        item: it recompiles into a kernel that runs the whole batch (see `JaxFfi._make_op`)."""
        return jax.vmap( func )

    def grad( self, func, argnums = 0 ):
        """Gradient of a scalar-valued `func`. A `driver.call` inside it reaches its `bwd_code`
        through the VJP rule the call registers (see `JaxFfi._call_with_vjp`)."""
        return jax.grad( func, argnums = argnums )

    def jit( self, func ):
        """Trace + compile `func` ONCE and reuse the executable across calls. A `driver.call` inside
        it is a proper FFI primitive, so it survives the trace unchanged; only the surrounding
        tensor algebra is fused by XLA. Reusing the jitted function across optimizer steps is what
        removes the per-step retrace/lower cost (bounded, flat memory) instead of re-lowering the
        whole graph each iteration. Under this trace, a `driver.call` cannot grow a capacity from
        Python, so an overflow is reported at run time via `jax.debug.callback` (see `call`)."""
        return jax.jit( func )

    def vjp( self, func, *primals ):
        """`( func( *primals ), pullback )` -- the framework's reverse-mode primitive. Lets a test
        seed an output cotangent directly (a symbolic zero included), without going through a
        scalar loss."""
        return jax.vjp( func, *primals )

    def checkpoint( self, func ):
        """`func` with its intermediates RECOMPUTED during the backward instead of retained.

        Trades one extra forward evaluation of `func` for dropping its residuals from the tape.
        Decisive when a loss is a SUM of many such pieces: without it the tape holds every piece's
        intermediates at once, so peak memory grows with the number of pieces even though the
        forward only ever holds one at a time (`OtPlan1d._update_outputs_via_angle_loop` and
        `Image.try_update_otplan1d` do this per angle; `DiskProjector` per chunk of disks).

        Pairs with `fold` (or a `lax.map`): a checkpoint alone bounds NOTHING if the pieces are
        UNROLLED side by side, since nothing then forces the compiler to run them one at a time
        (measured on `DiskProjector`: unrolled, checkpointed or not, the peak stayed one chunk's
        buffers TIMES the number of chunks). The loop is what serializes the execution; the
        checkpoint is what keeps the loop's per-iteration residuals down to its inputs."""
        return jax.checkpoint( func )

    def fold( self, body, init, xs ):
        """`body( carry, x )` applied SEQUENTIALLY over the leading axis of `xs` (an array, or a
        pytree of arrays sharing that axis), returning the final carry -- differentiable, and a
        single LOOP in the compiled graph instead of one unrolled copy of `body` per element.

        That is the whole point: an unrolled Python loop lets the compiler schedule every iteration
        concurrently (and grows the trace, hence the compile time, linearly with the iteration
        count), whereas this lowers to one `lax.scan`, where only ONE iteration's buffers are live.
        Checkpoint the expensive part of `body` (see `checkpoint`) to bound what the loop stores per
        iteration -- but leave the CARRY UPDATE outside that checkpoint, or the carry itself becomes
        a residual and gets stored once per iteration."""
        return jax.lax.scan( lambda carry, x: ( body( carry, x ), None ), init, xs )[ 0 ]

    def random( self, shape, dtype = None ):
        seed = getattr( self, "_rng_seed", 0 )
        self._rng_seed = seed + 1
        dtype_ver = Dtype.factory( dtype or self.ftype ).driver_version
        return jax.random.uniform( jax.random.PRNGKey( seed ), tuple( shape ), dtype = dtype_ver )

    # A symbolic zero: a SHAPED, TYPED, BUFFERLESS value that reads as 0 -- what a `Tensor` holds
    # in `_raw` for a symbolic-zero cotangent (lowers to a `ZeroTensor`, dropped at compile time).
    # Jax hands us its own `SymbolicZero` in the backward; we can also mint one from a shape/dtype.
    def symbolic_zero( self, shape, dtype = None ):
        from jax.custom_derivatives import SymbolicZero
        dv = Dtype.factory( dtype or self.ftype ).driver_version
        return SymbolicZero( jax_core.ShapedArray( tuple( shape ), dv ) )

    def is_symbolic_zero( self, x ):
        from jax.custom_derivatives import SymbolicZero
        return isinstance( x, SymbolicZero )

    # whether `x` is bound to a trace rather than being a concrete value. What tells a HOST value
    # from a device one (see `ShapeArray`, which refuses to be built from a tracer): a count Python
    # can size a buffer with, versus one that only exists inside the trace.
    def is_traced( self, x ):
        return isinstance( x, jax_core.Tracer )

    # detaches `x` from the gradient tape: a value computed FROM a perturbed input but that itself
    # carries no meaningful gradient (e.g. `Image.cell_cum_mass`, a routing helper -- see
    # `distributions/Image.py::_update_cell_cum_mass`). Applied where such a value is COMPUTED, not
    # at each read site: once detached, nothing downstream ever sees a gradient trace through it.
    def stop_gradient( self, x ):
        return jax.lax.stop_gradient( x )

    # reductions -- the backend-agnostic verbs `Tensor` reduces through (`axis` is
    # a dimension index or a tuple of them; `None` reduces everything to a scalar).
    def sum( self, a, axis = None ):
        return jnp.sum( a, axis = axis )

    def prod( self, a, axis = None ):
        return jnp.prod( a, axis = axis )

    def max( self, a, axis = None ):
        return jnp.max( a, axis = axis )

    def min( self, a, axis = None ):
        return jnp.min( a, axis = axis )

    def mean( self, a, axis = None ):
        return jnp.mean( a, axis = axis )

    def all( self, a, axis = None ):
        return jnp.all( a, axis = axis )

    def any( self, a, axis = None ):
        return jnp.any( a, axis = axis )

    def where( self, cond, a, b ):
        return jnp.where( cond, a, b )

    # elementwise math -- the backend-agnostic verbs `Tensor` maps through (shape preserved).
    def sqrt( self, a ):
        return jnp.sqrt( a )

    def arcsin( self, a ):
        return jnp.arcsin( a )

    def clip( self, a, lo = None, hi = None ):
        return jnp.clip( a, lo, hi )


    def call( self, code : FfiCode | str, output_attributes = (), output_exceptions = (), input_exceptions = (), output_capacities = {}, batch_alignment = None, has_dynamic_capacity = True, scratch_attributes = (), **kwargs ):
        """Run the C++ `code` on the objects passed as kwargs.

        The objects are built by the caller; nothing is returned. Every list below names
        attributes by dotted path (`"cell.vertex_positions"`, or `"cell"` for a whole subtree).

        `output_attributes` are what the kernel PRODUCES: a fresh buffer, rebound onto the
        attribute once the call returns. An attribute that already holds data is an input; an
        empty, undeclared one is not bound at all (the kernel sees a null view -- it may simply
        be an optional field this kernel does not use).

        `output_attribute_exceptions` carve holes in that subtree: a path under a named output
        the kernel does NOT produce this time (`output_attributes = [ "cell" ]` with
        `output_attribute_exceptions = [ "cell.vertex_indices" ]` -- a run that leaves the vertex
        indices alone). The carved-out attribute falls back to being observed, exactly as if it
        had never been under an output.

        `input_exceptions` is the symmetric carve-out on the OTHER default: a path forced to stay
        UNBOUND even though the attribute holds data (`input_exceptions = [ "cell.cut_offsets" ]`
        on a `cell` this kernel never reads) -- it is never transferred, and (being a `NoneTensor`
        rather than a real buffer) never becomes a differentiable primal, so its backward never
        needs a cotangent for it either. Use it for members an aggregate happens to carry but this
        particular kernel has no business touching.

        `capacities` says how big to allocate: `{ "cell.nb_vertices": 8 }`. It belongs to the
        call and not to the object, because it is a decision about THIS allocation -- an object
        only ever states what it IS. A capacity already materialized in a buffer need not be
        restated (it is read back from it).

        Inputs and outputs are disjoint, as in XLA: an update in place is a Python-side
        rebinding, for the caller to make between two calls.

        A capacity may of course turn out to be too small -- only the kernel knows how many items
        it produces. It says so (it records the count that did not fit, see
        `support/containers/ErrorBuffer.h`), and we simply RUN AGAIN with room for it: what a
        failed run wrote is discarded, outputs being fresh buffers anyway. The new capacity is
        `max( what was asked for, twice what we had )` -- a capacity exceeded once tends to be
        exceeded again, so we make room rather than track a count.
        """
        if isinstance( code, str ):
            code = FfiCode( code )

        prefix = code.name
        if prefix:
            prefix += "_"

        output_capacities = dict( output_capacities )   # ours to grow: the caller's dict is not ours to touch
        while True:
            ca = CallArgsAnalysis( kwargs, self.device, output_attributes, output_capacities, output_exceptions, input_exceptions, batch_alignment, scratch_attributes )
            ffi_call( code, ca, self.device, prefix )

            overflows = ca.capacity_overflows()
            if overflows is None:
                # under a `jit` / `vmap` trace, the buffer holds a traced value: Python cannot
                # look at it here, hence cannot grow anything and run again. What it can still do
                # is not return silently truncated results -- so the check moves to run time.
                #
                # `has_dynamic_capacity` lets the caller drop that run-time check when NONE of the
                # outputs has a kernel-decided count (all sizes prescribed upstream, e.g. OtPlan1d
                # outputs sized from `nb_diracs`): overflow is then impossible and the check is pure
                # overhead -- a `jax.debug.callback` that XLA cannot elide and that forces a
                # device->host SYNC per call (a few % on CPU, a pipeline stall on GPU). It stays True
                # by default because it cannot be decided a priori from the call args alone; only the
                # caller knows whether a count is prescribed or produced.
                if has_dynamic_capacity:
                    jax.debug.callback( _raise_on_error, ca.errors.raw )
                return

            if not overflows:
                return

            for path, wanted, capacity in overflows:
                output_capacities[ path ] = max( wanted, 2 * capacity )


    # class CapacityOverflow( RuntimeError ):
    #     """Raised when a dynamic-capacity tensor overflows inside jax.jit.

    #     Increase max_of_<axis> or call outside jax.jit to enable the automatic retry loop.
    #     """
    #     pass


    # @staticmethod
    # def is_capacity_overflow( e: BaseException ) -> str:
    #     """True if e is, or wraps (via __context__), a CapacityOverflow.

    #     Needed because jax.debug.callback exceptions are wrapped in jax.errors.JaxRuntimeError.
    #     """
    #     if isinstance( e, jax.errors.JaxRuntimeError ) and len( e.args ):
    #         for arg in e.args:
    #             txt = cast( str, arg )
    #             pos = txt.find( "CapacityOverflow" )
    #             if pos >= 0:
    #                 return txt[ pos: ]
    #     return ""

    # @property
    # def name( self ) -> str:
    #     return "jax"

    # @staticmethod
    # def find_device( normalized_device: str ):
    #     """ find the jax device from a normalized name like cpu, cuda:1, metal """
    #     if normalized_device.startswith( "cpu" ):
    #         return jax.devices( "cpu" )[ 0 ]
    #     if normalized_device.startswith( "cuda" ):
    #         idx = int( normalized_device.split( ":" )[ 1 ] ) if ":" in normalized_device else 0
    #         return jax.devices( "gpu" )[ idx ]
    #     if normalized_device.startswith( "metal" ):
    #         return jax.devices( "METAL" )[ 0 ]
    #     raise RuntimeError( f"Unknown device { normalized_device }" )

    # def find_dtype( self, dtype ):
    #     if self is not None and dtype is None:
    #         return self.dtype

    #     if dtype is float:
    #         return self.dtype

    #     if dtype is int:
    #         return self.itype

    #     if dtype == "FP32":
    #         return jnp.float32
    #     if dtype == "FP64":
    #         return jnp.float64
    #     if dtype == "PI8":
    #         return jnp.uint8
    #     if dtype == "PI32":
    #         return jnp.uint32
    #     if dtype == "PI64":
    #         return jnp.uint64

    #     return jnp.dtype( dtype )

    # @property
    # def array_type( self ):
    #     return ( jax.Array, jax_core.Tracer )

    # def normalized_type_for( self, dtype ) -> str:
    #     """TL type name ( e.g. 'FP32', 'SI64' ) for a python / numpy / framework scalar
    #     type or a Dtype. Driver-independent: it goes through the Dtype abstraction and
    #     resolves an unsized type to the driver's float / index width."""
    #     if dtype is float:
    #         dtype = self.ftype
    #     elif dtype is int:
    #         dtype = self.itype
    #     d = Dtype.factory( dtype )
    #     if d.size is None:
    #         d = Dtype( d.floating_point, ( self.ftype if d.floating_point else self.itype ).size, d.signed )
    #     return d.cpp_name

    # @property
    # def normalized_dtype( self ):
    #     return self.normalized_type_for( self.dtype )

    # @property
    # def normalized_itype( self ) -> str:
    #     return self.normalized_type_for( self.itype )

    # @property
    # def uint64( self ):
    #     return jnp.uint64

    # def is_int_dtype( self, dtype ):
    #     return jnp.issubdtype( dtype, jnp.integer )

    # def grad( self, f, *args ):
    #     """
    #     Jacobian of f w.r.t. its float scalar/tensor args (sequential backward passes, vmap-free).

    #     jax.jacobian is intentionally avoided: it uses vmap internally, which is not
    #     supported by our ffi_call primitives. Instead, jax.vjp is called once for the
    #     forward pass and then once per output element for the backward pass.

    #     - scalar output → gradient (same shape as each arg)
    #     - tensor output, scalar arg → derivative array (same shape as output)
    #     - tensor output, tensor arg → full Jacobian (output_shape + arg_shape)
    #     - multiple differentiable args → tuple of the above
    #     - no differentiable args → empty tuple
    #     """
    #     def _is_float_arg( a ):
    #         if isinstance( a, float ):
    #             return True
    #         if isinstance( a, ( jax.Array, numpy.ndarray, jax_core.Tracer ) ):
    #             return jnp.issubdtype( jnp.result_type( a ), jnp.floating )
    #         return False

    #     diff_indices = tuple( i for i, a in enumerate( args ) if _is_float_arg( a ) )
    #     if not diff_indices:
    #         return ()

    #     diff_args = tuple( jnp.asarray( args[ i ] ) for i in diff_indices )

    #     def _f_of_diff( *d_args ):
    #         full_args = list( args )
    #         for pos, val in zip( diff_indices, d_args ):
    #             full_args[ pos ] = val
    #         return f( *full_args )

    #     # single forward pass + capture vjp closure (no vmap)
    #     primals, f_vjp = jax.vjp( _f_of_diff, *diff_args )

    #     out_shape = primals.shape if hasattr( primals, 'shape' ) else ()
    #     out_size  = primals.size  if hasattr( primals, 'size'  ) else 1
    #     out_dtype = primals.dtype if hasattr( primals, 'dtype' ) else self.ftype.driver_version

    #     if out_size == 1:
    #         # single backward pass
    #         grads = f_vjp( jnp.ones( out_shape, dtype = out_dtype ) )
    #     else:
    #         # one backward pass per output element — O(out_size) passes, no vmap
    #         rows = [
    #             f_vjp( jnp.zeros( out_size, dtype = out_dtype ).at[ i ].set( 1.0 ).reshape( out_shape ) )
    #             for i in range( out_size )
    #         ]
    #         # rows[i] is a tuple (one grad per diff arg); rebuild Jacobians
    #         grads = tuple(
    #             jnp.stack( [ rows[ i ][ j ] for i in range( out_size ) ] ).reshape( out_shape + diff_args[ j ].shape )
    #             for j in range( len( diff_indices ) )
    #         )

    #     if len( diff_indices ) == 1:
    #         return grads[ 0 ]
    #     return grads

    # def nb_threads( self, **kwargs ):
    #     return self.device.nb_threads( **kwargs )



    # def t3( self, tensor, dtype = None ):
    #     """ make a rank 3 tensor """
    #     return self.tn( tensor, 3, dtype = dtype )

    # def t2( self, tensor, dtype = None ):
    #     """ make a rank 2 tensor """
    #     return self.tn( tensor, 2, dtype = dtype )

    # def t1( self, tensor, dtype = None ):
    #     """ make a rank 1 tensor """
    #     return self.tn( tensor, 1, dtype = dtype )

    # def t0( self, tensor, dtype = None ):
    #     """ make a rank 0 tensor """
    #     return self.tn( tensor, 0, dtype = dtype )

    # def tn( self, tensor, ndim, name = None, dtype = None ):
    #     """ make a rank ndim tensor """
    #     if tensor is None:
    #         return tensor

    #     dtype = Dtype.factory( dtype or self.ftype )

    #     if isinstance( tensor, jax.ShapeDtypeStruct ):
    #         return jnp.empty( [ s or 0 for s in tensor.shape ], dtype = dtype.driver_version )

    #     res = jnp.asarray( tensor, dtype = dtype.driver_version, device = self.device.driver_version )

    #     if ndim is not None and res.ndim != ndim:
    #         if name is not None:
    #             raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
    #         raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

    #     return res

    # def i0( self, tensor ):
    #     """ make a rank 0 int tensor """
    #     return self.tn( tensor, 0, dtype = self.itype )

    # def i1( self, tensor ):
    #     """ make a rank 1 int tensor """
    #     return self.tn( tensor, 1, dtype = self.itype )

    # def i2( self, tensor ):
    #     """ make a rank 2 int tensor """
    #     return self.tn( tensor, 2, dtype = self.itype )

    # def ones( self, shape, dtype = None ):
    #     return jnp.ones( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    # def zeros( self, shape, dtype = None ):
    #     return jnp.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    # def linspace( self, a, b, n, dtype = None ):
    #     return jnp.linspace( a, b, n, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    # def empty( self, shape, dtype = None ):
    #     return jnp.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    # def expand_dims( self, tensor, index ):
    #     return jnp.expand_dims( tensor, index )

    # def repeat( self, tensor, shape ):
    #     return jnp.tile( tensor, shape )

    # def stack( self, tensors, axis ):
    #     return jnp.stack( tensors, axis=axis )

    # def concatenate( self, tensors, axis ):
    #     return jnp.concatenate( tensors, axis=axis )

    # def linalg_solve( self, A, b ):
    #     return jnp.linalg.solve( A, b )

    # def moveaxis( self, tensor, source, destination ):
    #     return jnp.moveaxis( tensor, source, destination )

    # def hstack( self, lst ):
    #     return jnp.hstack( lst )

    # def to_numpy( self, t ):
    #     return numpy.array( t )

    # def is_a_tensor( self, value ):
    #     return isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray, jax.ShapeDtypeStruct, ad_util.SymbolicZero ) )

    # def vmap( self, func ):
    #     return jax.vmap( func )

    # def differentiable_type( self, dtype ):
    #     if dtype is None or dtype is float:
    #         return True
    #     if dtype is int:
    #         return False
    #     if isinstance( dtype, str ):
    #         if dtype.startswith( "FP" ):
    #             return True
    #         if dtype.startswith( "PI" ) or dtype.startswith( "SI" ):
    #             return False
    #     return not jax.numpy.issubdtype( dtype, jax.numpy.integer )

    # def call( self, code: str | FfiCode, mlir = True, **args ):
    #     """Call a C++ function via JAX XLA FFI.

    #     Args may be:
    #       - Mutable(obj)               — read+write; obj arrays reassigned after call
    #       - Return(Type, **kwargs)     — produces a new object or tensor
    #       - plain JAX array            — read-only input
    #       - int / float / str / ...    — scalar XLA attribute
    #     """

    #     if isinstance( code, str ):
    #         code = FfiCode( code )

    #     # argument analysis (get a jax compatible set of arg lists, ...)
    #     fai = CallArgsAnalysis( args, "Parameters" )

    #     # fail early if the provided tensor shapes disagree on a shared axis variable
    #     fai.check_axis_consistency()

    #     # check ffi function is registered
    #     module_name = self._module_name_for( code, fai )
    #     self._register_ffi_target( module_name, code, fai )

    #     # register vmap batching rule (once per module)
    #     from .FfiCode import FfiCodeParallel
    #     from .JaxMlirPrimitive import _vmap_rules
    #     if isinstance( code, FfiCodeParallel ) and module_name not in _vmap_rules:
    #         self._register_vmap_rule( module_name, code, args, fai )

    #     # forward helper
    #     def _call_ffi( differentiable_input_values ):
    #         # update fai content with the actual values
    #         fai.update_differentiable_input_values_with( differentiable_input_values )

    #         # loop until capacities are large enough
    #         while True:
    #             # make the call
    #             func = jax.ffi.ffi_call( module_name, fai.ffi_outputs )

    #             # normalize the output
    #             ret = func( *fai.ffi_inputs, **fai.ffi_attributes )

    #             # break if ok
    #             u64_output = ret[ fai.index_u64_output ]
    #             if u64_output[ fai.index_dynamic_size_exception ] == 0:
    #                 break

    #             # else delete outputs
    #             del ret

    #             # get faulty data
    #             da = fai.dynamix_axes[ u64_output[ fai.index_dynamic_size_exception + 0 ] - 1 ]
    #             needed_size = int( u64_output[ fai.index_dynamic_size_exception + 1 ] )
    #             faulty_axis_name = "max_of_" + da.name_in_parent


    #             # update output shape
    #             made_a_change = False
    #             for tensor_output in fai.tensor_outputs:
    #                 assert tensor_output.parent is not None
    #                 if tensor_output.parent() != da.parent():
    #                     continue
    #                 if tensor_output.ctor_kwargs is not None and faulty_axis_name in tensor_output.ctor_kwargs:
    #                     old_value = int( tensor_output.ctor_kwargs[ faulty_axis_name ] )
    #                     new_value = max( needed_size, 2 * old_value )

    #                     print( "resize", da.name_in_parent, "to", new_value )

    #                     tensor_output.ctor_kwargs[ faulty_axis_name ] = new_value
    #                     made_a_change = True

    #             assert( made_a_change )

    #         # always return a tuple
    #         if isinstance( ret, jax.Array ):
    #             return ( ret, )
    #         if isinstance( ret, tuple ):
    #             return ret
    #         return tuple( ret )

    #     if mlir and code.has_grad_code:
    #         from .JaxMlirPrimitive import get_or_create

    #         def _call_prim( differentiable_inputs ):
    #             fai.update_differentiable_input_values_with( differentiable_inputs )
    #             prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
    #             ret = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
    #             if isinstance( ret, jax.Array ):
    #                 return ( ret, )
    #             return tuple( ret )

    #         @jax.custom_vjp
    #         def my_mlir_op( differentiable_inputs ):
    #             return _call_prim( differentiable_inputs )

    #         def my_mlir_op_fwd( _differentiable_inputs ):
    #             perturbed_flags = tuple( v.perturbed if isinstance( v, CustomVJPPrimal ) else True for v in _differentiable_inputs )
    #             differentiable_inputs = tuple( v.value if isinstance( v, CustomVJPPrimal ) else v for v in _differentiable_inputs )
    #             outputs = _call_prim( differentiable_inputs )
    #             return outputs, ( differentiable_inputs, outputs, perturbed_flags )

    #         def my_mlir_op_bwd( residuals, grads_of_the_outputs ):
    #             differentiable_inputs, outputs, perturbed_flags = residuals
    #             if not isinstance( grads_of_the_outputs, ( tuple, list ) ):
    #                 grads_of_the_outputs = ( grads_of_the_outputs, )
    #             bfai = fai.backward_version( self, outputs, grads_of_the_outputs, "GradParameters", differentiable_inputs, perturbed_flags )
    #             bwd_module_name = self._ensure_backward_target( code, bfai, module_name )
    #             func = jax.ffi.ffi_call( bwd_module_name, bfai.ffi_outputs )
    #             ret = func( *bfai.ffi_inputs, **bfai.ffi_attributes )
    #             if isinstance( ret, jax.Array ):
    #                 ret = ( ret, )
    #             return ( tuple( ret[ ct.num_in_outputs ] for ct in bfai.tensor_outputs ), )

    #         my_mlir_op.defvjp( my_mlir_op_fwd, my_mlir_op_bwd, symbolic_zeros = True )
    #         outputs = my_mlir_op( tuple( fai.differentiable_ffi_inputs ) )

    #     elif mlir:
    #         outputs = self._call_via_primitive( fai, module_name )

    #     elif not grad:
    #         outputs = _call_ffi( tuple( fai.differentiable_ffi_inputs ) )
    #     else:
    #         @jax.custom_vjp
    #         def my_ffi_op( differentiable_inputs ):
    #             return _call_ffi( differentiable_inputs )

    #         def my_ffi_op_fwd( _differentiable_inputs ):
    #             # With symbolic_zeros = True, JAX wraps each input in CustomVJPPrimal( value, perturbed )
    #             perturbed_flags = tuple( v.perturbed if isinstance( v, CustomVJPPrimal ) else True for v in _differentiable_inputs )
    #             differentiable_inputs = tuple( v.value if isinstance( v, CustomVJPPrimal ) else v for v in _differentiable_inputs )
    #             outputs = _call_ffi( differentiable_inputs )

    #             return outputs, ( differentiable_inputs, outputs, perturbed_flags )

    #         def my_ffi_op_bwd( residuals, grads_of_the_outputs ):
    #             differentiable_inputs, outputs, perturbed_flags = residuals
    #             if not isinstance( grads_of_the_outputs, ( tuple, list ) ):
    #                 grads_of_the_outputs = ( grads_of_the_outputs, )

    #             bfai = fai.backward_version( self, outputs, grads_of_the_outputs, "GradParameters", differentiable_inputs, perturbed_flags )
    #             bwd_module_name = self._ensure_backward_target( code, bfai, module_name )
    #             func = jax.ffi.ffi_call( bwd_module_name, bfai.ffi_outputs )
    #             ret = func( *bfai.ffi_inputs, **bfai.ffi_attributes )
    #             if isinstance( ret, jax.Array ):
    #                 ret = ( ret, )

    #             return ( tuple( ret[ ct.num_in_outputs ] for ct in bfai.tensor_outputs ), )

    #         my_ffi_op.defvjp( my_ffi_op_fwd, my_ffi_op_bwd, symbolic_zeros = True )

    #         # --- appel ---
    #         outputs = my_ffi_op( tuple( fai.differentiable_ffi_inputs ) )

    #     # ret assembly
    #     fai.update_objects( outputs )
    #     res = fai.assemble_returns()

    #     # item or list
    #     if len( res ) == 0:
    #         return None
    #     if len( res ) == 1:
    #         return res[ 0 ]
    #     return res

    # def ffi_tensor_input_bind_code( self, ndim, dtype: Dtype ) -> str:
    #     return f"Arg<xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>>()"

    # def ffi_tensor_input_arg_code( self, ndim, dtype: Dtype ) -> str:
    #     return f"xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>"

    # def ffi_tensor_output_bind_code( self, ndim, dtype: Dtype ) -> str:
    #     return f"Ret<xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>>()"

    # def ffi_tensor_output_arg_code( self, ndim, dtype: Dtype ) -> str:
    #     return f"xla::ffi::ResultBuffer<{ dtype.jax_ffi_tensor_type() }>"

    # def ffi_tensor_output_spec( self, shape, dtype: Dtype ):
    #     return jax.ShapeDtypeStruct( shape, dtype.driver_version )

    # def ffi_parameter_bind_code( self, dtype, name: str ) -> str:
    #     return f"Attr<{ self.normalized_type_for( dtype ) }>( \"{ name }\" )"

    # def is_zero_tensor( self, value ):
    #     return isinstance( value, ad_util.SymbolicZero )

    # def _call_via_primitive( self, fai: CallArgsAnalysis, module_name: str ):
    #     from .JaxMlirPrimitive import get_or_create

    #     fai.update_differentiable_input_values_with(
    #         tuple( fai.differentiable_ffi_inputs )
    #     )

    #     # Detect if we're inside an outer jax.jit trace.
    #     # Inside a trace, any JAX op produces a Tracer instead of a concrete array.
    #     inside_jit = isinstance( jax.numpy.zeros( () ), jax_core.Tracer )

    #     if inside_jit and fai.dynamix_axes:
    #         # Try to run the retry loop concretely at trace time using
    #         # ensure_compile_time_eval. This works when all FFI inputs are concrete
    #         # (e.g. captured constants). If an input is abstract (JAX traced arg),
    #         # we fall back to a runtime callback that raises CapacityOverflow.
    #         # Note: JaxRuntimeError wraps the callback exception — use
    #         # sdot.is_capacity_overflow(e) or catch jax.errors.JaxRuntimeError.
    #         concrete_retry_done = False
    #         with jax.ensure_compile_time_eval():
    #             try:
    #                 while True:
    #                     func = jax.ffi.ffi_call( module_name, fai.ffi_outputs )
    #                     eager_ret = func( *fai.ffi_inputs, **fai.ffi_attributes )
    #                     if isinstance( eager_ret, jax.Array ):
    #                         eager_ret = ( eager_ret, )
    #                     else:
    #                         eager_ret = tuple( eager_ret )
    #                     u64 = eager_ret[ fai.index_u64_output ]
    #                     if int( u64[ fai.index_dynamic_size_exception ] ) == 0:
    #                         break
    #                     da = fai.dynamix_axes[ int( u64[ fai.index_dynamic_size_exception ] ) - 1 ]
    #                     need = int( u64[ fai.index_dynamic_size_exception + 1 ] )
    #                     name = "max_of_" + da.name_in_parent
    #                     for t in fai.tensor_outputs:
    #                         assert t.parent is not None
    #                         if t.parent() == da.parent() and t.ctor_kwargs is not None and name in t.ctor_kwargs:
    #                             t.ctor_kwargs[ name ] = max( need, 2 * int( t.ctor_kwargs[ name ] ) )
    #                 concrete_retry_done = True
    #             except Exception:
    #                 pass  # abstract inputs — will use runtime callback below

    #         prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
    #         ret = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
    #         if isinstance( ret, jax.Array ):
    #             ret = ( ret, )
    #         else:
    #             ret = tuple( ret )

    #         if not concrete_retry_done:
    #             def _raise_if_overflow( u64 ):
    #                 if u64[ fai.index_dynamic_size_exception ] != 0:
    #                     idx = int( u64[ fai.index_dynamic_size_exception ] ) - 1
    #                     da = fai.dynamix_axes[ idx ]
    #                     raise JaxDriver.CapacityOverflow(
    #                         f"Overflow on dynamic axis '{ da.name_in_parent }' inside jax.jit. "
    #                         f"Increase max_of_{ da.name_in_parent } or call outside jax.jit. "
    #                         f"Note: JAX wraps this in JaxRuntimeError — use sdot.is_capacity_overflow(e)."
    #                     )
    #             jax.debug.callback( _raise_if_overflow, ret[ fai.index_u64_output ], ordered = True )

    #     elif inside_jit:
    #         # No dynamic axes: plain JIT call.
    #         prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
    #         ret  = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
    #         if isinstance( ret, jax.Array ):
    #             ret = ( ret, )
    #         else:
    #             ret = tuple( ret )

    #     else:
    #         # Eager mode: retry loop until capacity is sufficient.
    #         while True:
    #             prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
    #             ret = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )

    #             if isinstance( ret, jax.Array ):
    #                 ret = ( ret, )
    #             else:
    #                 ret = tuple( ret )

    #             u64_output = ret[ fai.index_u64_output ]
    #             if u64_output[ fai.index_dynamic_size_exception ] == 0:
    #                 break

    #             da = fai.dynamix_axes[ int( u64_output[ fai.index_dynamic_size_exception + 0 ] ) - 1 ]
    #             needed_size = int( u64_output[ fai.index_dynamic_size_exception + 1 ] )
    #             faulty_axis_name = "max_of_" + da.name_in_parent

    #             made_a_change = False
    #             for tensor_output in fai.tensor_outputs:
    #                 assert tensor_output.parent is not None
    #                 if tensor_output.parent() != da.parent():
    #                     continue
    #                 if tensor_output.ctor_kwargs is not None and faulty_axis_name in tensor_output.ctor_kwargs:
    #                     old_value = int( tensor_output.ctor_kwargs[ faulty_axis_name ] )
    #                     new_value = max( needed_size, 2 * old_value )
    #                     tensor_output.ctor_kwargs[ faulty_axis_name ] = new_value
    #                     made_a_change = True

    #             assert made_a_change

    #     return ret

    # def _register_vmap_rule( self, module_name: str, code, orig_args: dict, fai: CallArgsAnalysis ):
    #     """Register a JAX vmap batching rule for an FfiCodeParallel primitive.

    #     Type-stable convention (mirrors the C++ "Cell stays Cell" choice): vmap prepends one
    #     batch axis as a leading tensor dimension on every batched value, without changing any
    #     type. An aggregate gains an entry in its instance-level `batch_axes`; a Tensor Return
    #     gains a leading axis; the FfiCodeParallel gains a prepended batch axis whose size is read
    #     at runtime from the first batched input's leading shape. The axis is never a struct member.
    #     """
    #     from jax.interpreters import batching as jax_batching
    #     from .JaxMlirPrimitive import _vmap_rules, _cache
    #     from ..compilation.CallArgsAnalysis import CallArgsAnalysis

    #     # Flat list of all input tensors in FFI order
    #     all_inputs = fai.differentiable_tensor_inputs + fai.non_differentiable_tensor_inputs

    #     # Attribute path ([ kwarg, field, ... ]) of each flat input in the Parameters struct
    #     def _path( tensor ):
    #         parts = []
    #         ca = tensor
    #         while ca.parent is not None:
    #             parts.append( ca.name_in_parent )
    #             ca = ca.parent()
    #         parts.reverse()
    #         return parts

    #     input_paths = [ _path( t ) for t in all_inputs ]

    #     # Group flat input indices by top-level kwarg, keeping the remaining field path.
    #     # rest == []  → the kwarg is itself a top-level tensor input
    #     # rest != []  → the kwarg is an aggregate, rest is the field path inside it
    #     flat_by_kwarg : dict[ str, list ] = {}
    #     for i, parts in enumerate( input_paths ):
    #         flat_by_kwarg.setdefault( parts[ 0 ], [] ).append( ( i, parts[ 1: ] ) )

    #     jax_driver = self

    #     def batch_rule( flat_args, in_dims ):
    #         n_inp = len( all_inputs )

    #         # Find batch size N and which input tensor provides it
    #         N               = None
    #         first_batch_idx = None
    #         for i in range( n_inp ):
    #             dim = in_dims[ i ] if i < len( in_dims ) else None
    #             if dim is not None:
    #                 N               = flat_args[ i ].shape[ dim ]
    #                 first_batch_idx = i
    #                 break

    #         if N is None:
    #             return flat_args, [ None ] * len( flat_args )

    #         # Move batch dims to position 0; expand non-batched inputs to (N, ...)
    #         # so the batched C++ kernel can index every input with the batch index.
    #         moved = list( flat_args )
    #         for i in range( n_inp ):
    #             dim = in_dims[ i ] if i < len( in_dims ) else None
    #             if dim is not None and dim != 0:
    #                 moved[ i ] = jnp.moveaxis( flat_args[ i ], dim, 0 )
    #             elif dim is None:
    #                 moved[ i ] = jnp.broadcast_to(
    #                     jnp.expand_dims( flat_args[ i ], 0 ),
    #                     ( N, ) + flat_args[ i ].shape
    #                 )

    #         # One vmap = one named batch axis, shared by every input of this call (so same name ⇒
    #         # joined iteration in the BatchPlan). The name is derived from how many batch axes are
    #         # already present, so a nested vmap gets a fresh name — deterministically (no per-object
    #         # counter ⇒ no false joins, and a stable module signature).
    #         from ..aggregate.Batched import Batched
    #         vmap_name      = f"vmap_{ jax_driver._existing_axis_count( orig_args ) }"
    #         batched_code   = code.with_prepended_batch_axis( vmap_name )

    #         # Prepend the vmap axis on every batched value. Our argument types (Return, Mutable,
    #         # aggregate instances) each know how to do this via `with_prepended_batch_axis`, so the
    #         # dispatch is polymorphic — no isinstance on our own types. A foreign top-level tensor is
    #         # wrapped in `Batched` to name its ( moved-to-0 ) batch axis; a scalar stays unchanged.
    #         new_args = {}
    #         for kw, arg in orig_args.items():
    #             prepend = getattr( arg, "with_prepended_batch_axis", None )
    #             if prepend is not None:
    #                 moved_leaves = { rest[ 0 ]: moved[ idx ] for idx, rest in flat_by_kwarg.get( kw, [] ) if rest }
    #                 new_args[ kw ] = prepend( N, moved_leaves, vmap_name )
    #             elif kw in flat_by_kwarg:                       # foreign top-level tensor input
    #                 new_args[ kw ] = Batched( moved[ flat_by_kwarg[ kw ][ 0 ][ 0 ] ], [ ( vmap_name, 0 ) ] )
    #             else:                                           # scalar / unbatched value
    #                 new_args[ kw ] = arg

    #         # Build batched CallArgsAnalysis and compile the batched module
    #         batched_fai    = CallArgsAnalysis( new_args, "Parameters" )
    #         batched_module = jax_driver._module_name_for( batched_code, batched_fai )
    #         jax_driver._register_ffi_target( batched_module, batched_code, batched_fai )

    #         # Call the batched primitive (jax.jit inside _call_via_primitive escapes the BatchingTrace)
    #         outputs = jax_driver._call_via_primitive( batched_fai, batched_module )
    #         if isinstance( outputs, jax.Array ):
    #             outputs = ( outputs, )
    #         else:
    #             outputs = tuple( outputs )

    #         # Tensor outputs have batch_dim=0; the internal u64 buffer is not mapped
    #         n_tensor_outs = len( batched_fai.tensor_outputs )
    #         out_dims      = [ 0 ] * n_tensor_outs + [ jax_batching.not_mapped ]
    #         return outputs, out_dims

    #     # Store and attach to any already-cached primitive for this module
    #     _vmap_rules[ module_name ] = batch_rule
    #     for cache_key, prim in _cache.items():
    #         if cache_key[ 0 ] == module_name:
    #             jax_batching.primitive_batchers[ prim ] = batch_rule

    # @staticmethod
    # def _existing_axis_count( orig_args: dict ) -> int:
    #     """How many batch axes are already present among the call's arguments (0 for a fresh
    #     top-level vmap). Used to name a nested vmap's axis without colliding with outer ones."""
    #     from ..aggregate.Return import Return
    #     best = 0
    #     for arg in orig_args.values():
    #         ba = getattr( arg, "batch_axes", None )
    #         if ba is not None:
    #             best = max( best, len( ba ) )
    #         elif isinstance( arg, Return ):
    #             axes = arg.type_kwargs.get( "batch_axes", getattr( arg.return_type, "batch_axes", None ) )
    #             best = max( best, len( axes or [] ) )
    #     return best

    # def _module_name_for( self, code: FfiCode, main_list: CallArgsAnalysis ):
    #     # get signature — include device type to avoid CPU/GPU cache collision
    #     base_signature = [ code.signature(), main_list.arguments.signature(), self.device.signature ]

    #     # module name
    #     from loom.util import encode_base_62
    #     res = re.sub( r'[^\w]', '_', str.join( "_", base_signature ) )
    #     while "__" in res:
    #         res = res.replace( "__", "_" )
    #     if len( res ) > 50:
    #         res = res[ : 50 - 11 ] + encode_base_62( res[ 50 - 11: ] )

    #     return res

    # _registered_ffi_targets = set()

    # def _register_ffi_target( self, module_name: str, code: FfiCode, args: CallArgsAnalysis ):
    #     if module_name in JaxDriver._registered_ffi_targets:
    #         return
    #     JaxDriver._registered_ffi_targets.add( module_name )

    #     from ..compilation.force_build import force_build

    #     # Metal binding: a plain Obj-C++ dylib loaded via ctypes (no nanobind module to import).
    #     if self.device.is_apple_gpu:
    #         if not force_build():
    #             try:
    #                 self._register_metal_target( module_name )
    #                 return
    #             except OSError:
    #                 pass
    #         self._make_metal_dylib( code, args, module_name )
    #         self._register_metal_target( module_name )
    #         return

    #     if not force_build():
    #         try:
    #             self._try_to_import_and_register_ffi_target( module_name )
    #             return
    #         except ( ImportError, SystemError ):
    #             pass

    #     self._make_dylib( code, args, module_name )
    #     self._try_to_import_and_register_ffi_target( module_name )

    # def _try_to_import_and_register_ffi_target( self, module_name: str ):
    #     if self.device.is_cuda_gpu:
    #         platform = "gpu"
    #     elif self.device.is_apple_gpu:
    #         platform = "METAL"
    #     else:
    #         platform = "cpu"
    #     module  = importlib.import_module( "sdot.generated_files." + module_name )
    #     capsule = getattr( module, module_name )()
    #     try:
    #         jax.ffi.register_ffi_target( module_name, capsule, platform = platform )
    #     except AttributeError:
    #         # jax.ffi not available in older JAX (< 0.4.37) — use xla_extension directly
    #         from jaxlib import xla_extension as xe
    #         xe.register_custom_call_target( module_name, capsule, platform = platform )

    # def _register_metal_target( self, module_name: str ):
    #     """Load the Metal binding dylib via ctypes and register its FFI handler with JAX.

    #     The handler is registered on the *CPU* backend: the JAX graph runs on CPU and the
    #     handler launches the Metal kernel internally. No nanobind / importable module is
    #     involved — we read the handler address through an extern "C" trampoline and wrap it in
    #     a PyCapsule by hand. Raises OSError if the dylib is missing (triggers a rebuild).
    #     """
    #     import ctypes
    #     from sdot.generated_files import compilation_directories

    #     dylib_dir = compilation_directories.dylib_dir()
    #     candidates = sorted( dylib_dir.glob( f"{ module_name }*.so" ) ) + sorted( dylib_dir.glob( f"{ module_name }*.dylib" ) )
    #     if not candidates:
    #         raise OSError( f"Metal binding dylib for '{ module_name }' not found in { dylib_dir }" )

    #     lib = ctypes.CDLL( str( candidates[ 0 ] ) )
    #     trampoline = getattr( lib, f"sdot_ffi_capsule_{ module_name }" )
    #     trampoline.restype = ctypes.c_void_p
    #     handler_addr = trampoline()

    #     PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    #     PyCapsule_New.restype  = ctypes.py_object
    #     PyCapsule_New.argtypes = [ ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p ]
    #     capsule = PyCapsule_New( handler_addr, b"xla._CUSTOM_CALL_TARGET", None )

    #     # keep a reference so the dylib is not unloaded while the target stays registered
    #     JaxDriver._metal_libs.setdefault( module_name, [] ).append( lib )

    #     try:
    #         jax.ffi.register_ffi_target( module_name, capsule, platform = "cpu" )
    #     except AttributeError:
    #         from jaxlib import xla_extension as xe
    #         xe.register_custom_call_target( module_name, capsule, platform = "cpu" )

    # _metal_libs: dict = {}

    # def _ensure_backward_target( self, code: FfiCode, bfai: CallArgsAnalysis, fwd_module_name: str ) -> str:
    #     """Lazily compile and register the backward FFI target from the actual bfai.

    #     Called at JAX trace time when the backward is first needed, so bfai correctly
    #     reflects which tensors are None/Zero vs real based on the actual cotangents.
    #     """
    #     from loom.util import encode_base_62
    #     raw = f"bwd_{ fwd_module_name }_{ bfai.arguments.signature() }"
    #     bwd_module_name = re.sub( r'[^\w]', '_', raw )
    #     while "__" in bwd_module_name:
    #         bwd_module_name = bwd_module_name.replace( "__", "_" )
    #     if len( bwd_module_name ) > 60:
    #         bwd_module_name = bwd_module_name[ :50 ] + encode_base_62( bwd_module_name[ 50: ] )

    #     if bwd_module_name in JaxDriver._registered_ffi_targets:
    #         return bwd_module_name
    #     JaxDriver._registered_ffi_targets.add( bwd_module_name )

    #     from ..compilation.force_build import force_build
    #     if not force_build():
    #         try:
    #             self._try_to_import_and_register_ffi_target( bwd_module_name )
    #             return bwd_module_name
    #         except ( ImportError, SystemError ):
    #             pass

    #     self._make_bwd_dylib( code, bfai, bwd_module_name )
    #     self._try_to_import_and_register_ffi_target( bwd_module_name )
    #     return bwd_module_name

    # def _make_dylib( self, code: FfiCode, fai: CallArgsAnalysis, module_name: str ):
    #     already_visited = set()
    #     fai.arguments.generate_structures( already_visited )

    #     includes = set( code.includes_for( "fwd" ) )
    #     includes.add( "sdot/jax_ffi_wrappers.h" )
    #     includes.add( "nanobind/nanobind.h" )

    #     lines = []
    #     lines.append( "" )
    #     lines.append( "namespace nb = nanobind;" )
    #     lines.append( "using namespace sdot;" )
    #     lines.append( "" )

    #     header = code.header_for( "fwd" )
    #     if header:
    #         lines.append( dedent( header ) )

    #     if self.device.is_cuda_gpu:
    #         lines.append( "cudaStream_t ExecutionContext_Cuda::default_stream;" )

    #     self._handler_source( includes, lines, code.code_for( "fwd", fai ), fai, module_name, "Parameters" )
    #     lines.append( "" )

    #     self._append_nb_module( lines, module_name )

    #     include_lines = [ f"#include <{ include }>" for include in sorted( includes, key = lambda s: ( -len( s ), s ) ) ]

    #     from ..compilation.make_dylib_from_source import make_dylib_from_source
    #     return make_dylib_from_source( str.join( "\n", include_lines + lines ), module_name, [], self.device )

    # def _make_metal_dylib( self, code: FfiCode, fai: CallArgsAnalysis, module_name: str ):
    #     """Compile the Metal binding: same XLA FFI handler as the CPU path, but emitted as
    #     Objective-C++ (it launches a Metal kernel), with no nanobind module — instead an
    #     extern "C" trampoline exposes the handler address to the ctypes loader."""
    #     already_visited = set()
    #     fai.arguments.generate_structures( already_visited )

    #     includes = set( code.includes_for( "fwd" ) )
    #     includes.add( "sdot/jax_ffi_wrappers.h" )
    #     includes.add( "sdot/metal/metal_launch.h" )

    #     lines = []
    #     lines.append( "" )
    #     lines.append( "using namespace sdot;" )
    #     lines.append( "" )

    #     # Each FfiCode decides what to emit for Metal (FfiCodeCustom keeps its hand-written body;
    #     # FfiCodeParallel generates an MSL kernel). No isinstance: uniform polymorphic interface.
    #     header_lines, body = code.metal_source( "fwd", fai, module_name )
    #     lines.extend( header_lines )

    #     self._handler_source( includes, lines, body, fai, module_name, "Parameters" )
    #     lines.append( "" )

    #     # extern "C" trampoline — returns the FFI handler address for the ctypes-built capsule
    #     lines.append( f'extern "C" void *sdot_ffi_capsule_{ module_name }() {{' )
    #     lines.append( f'    return reinterpret_cast<void *>( binding_{ module_name } );' )
    #     lines.append( "}" )

    #     include_lines = [ f"#include <{ include }>" for include in sorted( includes, key = lambda s: ( -len( s ), s ) ) ]

    #     from ..compilation.make_dylib_from_source import make_dylib_from_source
    #     return make_dylib_from_source( str.join( "\n", include_lines + lines ), module_name, [], self.device )

    # def _make_bwd_dylib( self, code: FfiCode, bfai: CallArgsAnalysis, bwd_module_name: str ):
    #     """Compile a backward-only binding using the actual bfai (with correct None/Zero/real types)."""
    #     already_visited = set()
    #     bfai.arguments.generate_structures( already_visited )

    #     includes = set( code.includes_for( "bwd" ) )
    #     includes.add( "sdot/jax_ffi_wrappers.h" )
    #     includes.add( "nanobind/nanobind.h" )

    #     lines = []
    #     lines.append( "" )
    #     lines.append( "namespace nb = nanobind;" )
    #     lines.append( "using namespace sdot;" )
    #     lines.append( "" )

    #     header = code.header_for( "bwd" )
    #     if header:
    #         lines.append( dedent( header ) )

    #     if self.device.is_cuda_gpu:
    #         lines.append( "cudaStream_t ExecutionContext_Cuda::default_stream;" )

    #     self._handler_source( includes, lines, code.code_for( "bwd", bfai ), bfai, bwd_module_name, "GradParameters" )
    #     lines.append( "" )

    #     self._append_nb_module( lines, bwd_module_name )

    #     include_lines = [ f"#include <{ include }>" for include in sorted( includes, key = lambda s: ( -len( s ), s ) ) ]

    #     from ..compilation.make_dylib_from_source import make_dylib_from_source
    #     return make_dylib_from_source( str.join( "\n", include_lines + lines ), bwd_module_name, [], self.device )

    # @staticmethod
    # def _append_nb_module( lines: list, module_name: str ):
    #     lines.append( "" )
    #     lines.append( "template<typename T>" )
    #     lines.append( "nb::capsule EncapsulateFfiCall( T *fn ) {" )
    #     lines.append( "  static_assert( std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>, \"Encapsulated function must be and XLA FFI handler\");" )
    #     lines.append( "  return nb::capsule( reinterpret_cast<void *>( fn ), \"xla._CUSTOM_CALL_TARGET\" );" )
    #     lines.append( "}" )
    #     lines.append( "" )
    #     lines.append( f"NB_MODULE( { module_name }, m ) {{" )
    #     lines.append( f"  m.def( \"{ module_name }\", []() {{ return EncapsulateFfiCall( binding_{ module_name } ); }} );" )
    #     lines.append( "}" )

    # def _handler_source( self, includes, lines, code: str, fai: CallArgsAnalysis, module_name: str, struct_name: str, suffix = "" ):
    #     is_gpu = self.device.is_cuda_gpu

    #     fai.get_code_for_parameters_struct( includes, lines, struct_name )
    #     lines.append( "" )

    #     # handler signature — append stream for GPU (PlatformStream<T> decodes to T directly)
    #     arg_decl = fai.arg_decl()
    #     if is_gpu:
    #         stream_arg = "cudaStream_t stream"
    #         arg_decl = f"{ arg_decl }, { stream_arg }" if arg_decl else stream_arg
    #     lines.append( f"xla::ffi::Error impl_{ module_name }{ suffix }( { arg_decl } ) {{" )

    #     if is_gpu:
    #         lines.append( "ExecutionContext_Cuda::default_stream = stream;" )

    #     lines.append( f"    using TF = { self.ftype.cpp_name };" ) #
    #     lines.append( f"    using TI = { self.itype.cpp_name };" )

    #     # arch instance
    #     if is_gpu:
    #         lines.append( f"    { self.device.cpp_type }::default_stream = stream;" )
    #     lines.append( f"    { self.device.cpp_type } execution_context;" )
    #     lines.append( f"    { self.device.mem_type } memory_space;" )

    #     # u8_...
    #     if len( fai.u8_input_values ):
    #         if is_gpu:
    #             lines.append( f"    std::vector<PI8> _u8_host( { len( fai.u8_input_values ) } );" )
    #             # u8_input may be uploaded to device asynchronously on `stream`; enqueue D2H on the
    #             # same stream so it is ordered after the upload, then sync to read on host
    #             lines.append( "    cudaMemcpyAsync( _u8_host.data(), u8_input_buffer.typed_data(), _u8_host.size() * sizeof( PI8 ), cudaMemcpyDefault, stream );" )
    #             lines.append( "    cudaStreamSynchronize( stream );" )
    #             lines.append( "    const PI8 *u8_input = _u8_host.data();" )
    #         else:
    #             lines.append( "    const PI8 *u8_input = u8_input_buffer.typed_data();" )

    #     if fai.u64_output_size:
    #         lines.append( "    PI64 *u64_output = u64_output_buffer->typed_data();" )
    #         if is_gpu:
    #             lines.append( "    cudaMemsetAsync( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ), stream );" )
    #         else:
    #             lines.append( "    std::memset( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ) );" )

    #     # conversions
    #     fai.get_code_for_tensor_conversions( lines )

    #     # beg try block
    #     lines.append( "    try {" )

    #     # call the function
    #     lines.append( f"        auto p = { fai.arguments.assembled_code( '        ' ) };" )
    #     lines.append( "" )
    #     lines.append( indent( dedent( code ), '        ' ) )

    #     # end try block
    #     lines.append( '    } catch ( DynamicSizeException de ) {' )
    #     if is_gpu:
    #         # DynamicSizeException from device code is not supported yet; signal via cudaMemcpy
    #         lines.append( '        PI64 _de_vals[ 2 ] = { 1 + de.num_dynamic_axis, de.needed_size };' )
    #         lines.append( f'        cudaMemcpyAsync( u64_output + { fai.index_dynamic_size_exception }, _de_vals, 2 * sizeof( PI64 ), cudaMemcpyHostToDevice, stream );' )
    #     else:
    #         lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 0 } ] = 1 + de.num_dynamic_axis;' )
    #         lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 1 } ] = de.needed_size;' )
    #     lines.append( '    }' )

    #     # end impl
    #     if is_gpu:
    #         lines.append( '    if ( cudaError_t e = cudaGetLastError(); e != cudaSuccess )' )
    #         lines.append( '        fprintf( stderr, "[sdot] CUDA kernel error: %s\\n", cudaGetErrorString( e ) );' )
    #     lines.append( "    return xla::ffi::Error::Success();" )
    #     lines.append( "}" )

    #     # XLA_FFI_DEFINE_HANDLER_SYMBOL — append PlatformStream bind for GPU
    #     bind_chain = [ "xla::ffi::Ffi::Bind()" ] + fai.bind_chain()
    #     if is_gpu:
    #         bind_chain.append( "Ctx<xla::ffi::PlatformStream<cudaStream_t>>()" )
    #     lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( binding_{ module_name }{ suffix }, impl_{ module_name }{ suffix }, { str.join( '.', bind_chain ) } );" )


    # def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
    #     """ small helper to optimize `loss` wrt `params` using L-BFGS (via scipy.optimize).
    #         - `params`  : JAX array or list of JAX arrays
    #         - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
    #         Returns the optimized params (same type as input).
    #     """
    #     import scipy.optimize

    #     # support single array or list of arrays
    #     is_list  = isinstance( params, ( list, tuple ) )
    #     p_list   = list( params ) if is_list else [ params ]
    #     shapes   = [ p.shape for p in p_list ]
    #     sizes    = [ int( np.prod( s ) ) for s in shapes ]

    #     def pack( arrays ):
    #         return np.concatenate( [ np.array( a ).flatten() for a in arrays ] ).astype( np.float64 )

    #     def unpack( x_flat ):
    #         parts, offset = [], 0
    #         for shape, size in zip( shapes, sizes ):
    #             parts.append( jnp.asarray( x_flat[ offset : offset + size ].reshape( shape ), dtype = self.dtype, device = self.device ) )
    #             offset += size
    #         return parts

    #     val_and_grad = jax.value_and_grad( lambda *ps: loss( list( ps ) if is_list else ps[ 0 ] ) )

    #     iter_ref = [ 0 ]

    #     def f_and_g( x_flat ):
    #         ps        = unpack( x_flat )
    #         val, grad = val_and_grad( *ps )
    #         g_list    = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
    #         return float( np.array( val ) ), pack( g_list )

    #     def callback( x_flat ):
    #         ps = unpack( x_flat )
    #         if on_iter:
    #             _, grad = val_and_grad( *ps )
    #             g_list   = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
    #             grad_norm = float( np.linalg.norm( pack( g_list ) ) )
    #             on_iter( ps if is_list else ps[ 0 ], iter_ref[ 0 ], grad_norm )
    #         iter_ref[ 0 ] += 1

    #     result = scipy.optimize.minimize(
    #         f_and_g, pack( p_list ),
    #         method  = 'L-BFGS-B',
    #         jac     = True,
    #         callback = callback,
    #         options  = { 'maxiter': max_iter, 'gtol': tol_grad },
    #     )

    #     final = unpack( result.x )
    #     return final if is_list else final[ 0 ]


def _raise_on_error( errors ):
    """The error buffer, checked at RUN time -- what is left when trace time cannot see it.

    Growing a capacity means running again, and that is a Python loop: it needs the count that did
    not fit, which under a trace only exists once the kernel has run. So inside a `jit` or a
    `vmap` a capacity has to be given generously -- and when it was not, this is what says so,
    rather than letting truncated results through."""
    if int( errors[ 0 ] ) != 0:
        raise RuntimeError(
            "the kernel reported an error (a capacity too small, typically) from inside a traced "
            "call, where Python cannot grow one and run again: give the call a larger capacity."
        )


def _has_tracer( data ) -> bool:
    if isinstance( data, jax_core.Tracer ):
        return True
    if isinstance( data, ( list, tuple ) ):
        return any( _has_tracer( x ) for x in data )
    return False
