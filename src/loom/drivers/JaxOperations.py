"""The jax framework's operations (see `JaxFramework`)."""
from ..devices.Device import Device
import functools
from typing import Any
from .CallArgsAnalysis import CallArgsAnalysis
from ..compilation.FfiCode import FfiCode, Kernels
from ..util.info import info
from .. import env
from .JaxFfi import call_body, call as jax_ffi_call
from ..tensor.Dtype import Dtype
import numpy
import jax.core as jax_core
import jax.numpy as jnp
import jax


class JaxOperations:
    """What the jax framework DOES with an array that already exists: its operations, its
    transforms, its readings, its way of building one from a concrete dtype and an explicit device, and of
    launching a kernel. It has no default of its own: the defaults are loom's (`Settings`), and are handed in."""

    def dtype_of( self, x ):
        """The `Dtype` a jax buffer ACTUALLY has -- what a declaration is checked against."""
        from ..tensor.Dtype import Dtype
        aval = getattr( x, "aval", None )          # a SymbolicZero carries its type in its aval
        return Dtype.from_numpy( ( aval if aval is not None else x ).dtype )

    def reshape( self, tensor, shape ):
        return jnp.reshape( tensor, tuple( shape ) )

    def stack( self, tensors, axis = 0 ):
        return jnp.stack( tensors, axis = axis )

    def pad( self, tensor, pad_width ):
        return jnp.pad( tensor, pad_width )

    def transpose( self, a, axes = None ):
        return jnp.transpose( a, axes )

    def vmap( self, func ):
        """Map `func` over a new leading axis. A `loom.ffi_call` inside it is not replayed item by
        item: it recompiles into a kernel that runs the whole batch (see `JaxFfi._make_op`)."""
        return jax.vmap( func )

    def grad( self, func, argnums = 0 ):
        """Gradient of a scalar-valued `func`. A `loom.ffi_call` inside it reaches its backward kernel
        through the VJP rule the call registers (see `JaxFfi._call_with_vjp`)."""
        return jax.grad( func, argnums = argnums )

    def jit( self, func ):
        """Trace + compile `func` ONCE and reuse the executable across calls. A `loom.ffi_call` inside
        it is a proper FFI primitive, so it survives the trace unchanged; only the surrounding
        tensor algebra is fused by XLA. Reusing the jitted function across optimizer steps is what
        removes the per-step retrace/lower cost (bounded, flat memory) instead of re-lowering the
        whole graph each iteration. Under this trace, a `loom.ffi_call` cannot grow a capacity from
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

    def is_strided( self, x ):
        """Never: XLA hands the kernel dense buffers, whatever the array was before (see `PointerFfi`
        for the drivers that read a framework's own strides)."""
        return False

    def is_symbolic_zero( self, x ):
        from jax.custom_derivatives import SymbolicZero
        return isinstance( x, SymbolicZero )

    # whether `x` is bound to a trace rather than being a concrete value. What tells a HOST value
    # from a device one (see `ShapeArray`, which refuses to be built from a tracer): a count Python
    # can size a buffer with, versus one that only exists inside the trace.
    def is_traced( self, x ):
        return isinstance( x, jax_core.Tracer )

    # what runs inside this context with CONCRETE inputs is EVALUATED, even under a trace ( `jax.jit` ): the calls of a
    # host-driven construction ( the BSP tree, level by level, reading each level back ) whose inputs are constants of the
    # trace. A traced input still gives a tracer -- the caller checks its inputs first.
    def concrete_eval( self ):
        return jax.ensure_compile_time_eval()

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

    # a SCAN, and not a reduction: the shape is kept, `axis` designates a single one and is
    # never `None` ( see `Tensor.cumsum` ). The backend does it, so it is its
    # scan primitive -- optimized, differentiable, and which goes through `jit` / `vmap` like the rest.
    def cumsum( self, a, axis ):
        return jnp.cumsum( a, axis = axis )

    # end to end along an axis. Neither a reduction nor a scan: what comes out is BIGGER
    # than what goes in, so it is the only verb whose shape is not that of the input.
    def concatenate( self, arrays, axis = 0 ):
        return jnp.concatenate( list( arrays ), axis = axis )

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

    def exp( self, a ):
        return jnp.exp( a )

    def clip( self, a, lo = None, hi = None ):
        return jnp.clip( a, lo, hi )

    def matmul( self, a, b ):
        return a @ b

    def to_numpy( self, t ):
        """`t` as a host numpy array, wherever it lives (a card included)."""
        return numpy.asarray( t )

    @property
    def available_gpus( self ):
        return sum( "gpu" in device.platform for device in jax.devices() )

    # ---- what a framework calls a dtype -----------------------------------------------------------------
    def dtype_version( self, dtype ):
        """The jax dtype a CONCRETE `Dtype` denotes: its size is the caller's to have filled in (the
        defaults are loom's, not a framework's)."""
        from ..tensor.Dtype import Dtype, REAL, SINT, UINT, BOOL
        dtype = Dtype.factory( dtype )
        if dtype.kind == BOOL:
            return jnp.bool_
        if dtype.size is None:
            raise ValueError( f"the size of { dtype.cpp_name } is not resolved: it is loom's default to fill in, not a framework's" )
        table = { REAL: { 16: jnp.float16, 32: jnp.float32, 64: jnp.float64 },
                  SINT: { 8: jnp.int8, 16: jnp.int16, 32: jnp.int32, 64: jnp.int64 },
                  UINT: { 8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64 } }[ dtype.kind ]
        try:
            return table[ dtype.size ]
        except KeyError:
            raise ValueError( f"unsupported size for { dtype.cpp_name }" )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a concrete `Dtype`); a no-op when it already is."""
        return jnp.asarray( x, dtype = self.dtype_version( dtype ) )

    # ---- building a value, with a concrete dtype (see `DefaultSizes` for the defaults) ------------------------
    def array( self, data, dtype, device = None ):
        if data is None:
            return None
        dtype_ver = self.dtype_version( dtype )
        if _has_tracer( data ):
            return jnp.asarray( data, dtype = dtype_ver )
        # a HOST value (no tracer in it): cast through plain NUMPY, not a jax op. `jax.jit` makes
        # its dynamic trace the ambient one for the WHOLE lexical extent of the traced function, so
        # any jax primitive called there -- `jnp.asarray` included -- comes back one of its
        # tracers, even given a value that never depends on what is being traced (a domain closed
        # over as a Python constant, say). That tracer is then unreadable from Python
        # (`TracerArrayConversionError` on `np.asarray`), which a plain host constant should never
        # be. Staying in numpy here means such a value stays host-readable wherever it is built,
        # jit or not -- jax lifts a numpy constant into the trace itself, automatically, the moment
        # some actual jax primitive consumes it.
        return numpy.asarray( data, dtype = dtype_ver )

    # functional building blocks (tracer-safe, differentiable): used to assemble
    # padded buffers without in-place mutation, which does not fit Jax.
    def zeros( self, shape, dtype, device = None ):
        return jnp.zeros( shape, dtype = self.dtype_version( dtype ) )

    def full( self, shape, value, dtype, device = None ):
        return jnp.full( shape, value, dtype = self.dtype_version( dtype ) )

    def ones( self, shape, dtype, device = None ):
        return jnp.ones( shape, dtype = self.dtype_version( dtype ) )

    def arange( self, nb, dtype, device = None ):
        return jnp.arange( nb, dtype = self.dtype_version( dtype ) )

    def linspace( self, a, b, nb, dtype, device = None ):
        return jnp.linspace( a, b, nb, dtype = self.dtype_version( dtype ) )

    def random( self, shape, dtype, seed = None, device = None ):
        """A uniform draw. `seed = None` takes the next value of a process-wide COUNTER, so
        that two consecutive draws differ; an explicit `seed` bypasses it.

        Why the counter is not enough: it advances at every draw of the process, so what a
        test draws depends on HOW MANY draws the previous tests made. A test can then
        pass alone and fail in the suite (or the reverse) without anything having changed in it --
        and this has happened, see `check_grad`. A caller who wants to be reproducible passes its seed.
        """
        if seed is None:
            seed = getattr( self, "_rng_seed", 0 )
            self._rng_seed = seed + 1
        return jax.random.uniform( jax.random.PRNGKey( seed ), tuple( shape ), dtype = self.dtype_version( dtype ) )

    # Jax hands us its own `SymbolicZero` in the backward; we can also mint one from a shape/dtype.
    def symbolic_zero( self, shape, dtype ):
        from jax.custom_derivatives import SymbolicZero
        return SymbolicZero( jax_core.ShapedArray( tuple( shape ), self.dtype_version( dtype ) ) )


    def call( self, device, name, *kernels, nb_items = None, batch_alignment = None, has_dynamic_capacity = True, failures = None, **args ):
        """Runs one or two `FfiCode`s on the values passed as kwargs.

        THIS IS `loom.ffi_call`: there is no other form of call. The vocabulary of the arguments
        -- `loom.out`, `loom.mutable`, `loom.scratch`, `loom.unbound` -- and the reason for each
        choice are in `loom/calls.py`, which defines them and translates them (`lower_args`).

            temperature = loom.ffi_call(
                "diffusion_step",            # the call's name: mandatory, hence first
                forward, backward,           # the second is the ADJOINT ( optional )
                temperature = loom.mutable( temperature ),
                coef = coef,
            )

        A call takes the FORWARD, and -- if the thing must be differentiable -- the BACKWARD, both
        positional and in that order. They are two full-fledged kernels: the backward runs on
        other buffers and may want its own launch geometry (see `FfiCode`). It is
        `name` that identifies both -- it names the functors (`<name>_kernel` and
        `<name>_bwd_kernel`), prefixes the compiled target and groups the compilation journal.

        Inputs and outputs are DISJOINT, as in XLA: what looks like an in-place
        update is a rebinding on the Python side, which `loom.mutable` writes for us.

        `nb_items = n` says HOW MANY ITEMS to launch, without any object having to carry the axis: the
        body receives its rank in `flat_index` (see `FfiCode.per_item`). Without it, a call
        launches a single item -- or as many as the batch axes of its arguments make.

        Three names remain reserved here, and they are settings of the call, not data:
        `nb_items`, `batch_alignment` (the alignment of the batch dimension),
        `has_dynamic_capacity` and `failures` ( `{ code: message }`: what an `ErrorKind::failure`
        record of that code means -- `{value}` in the message is the record's value; the call then
        raises `KernelFailure` with it, eager or traced ).

        A capacity can turn out too small -- only the kernel knows how many items it produces.
        It says so (it records the count that did not fit, see
        `support/containers/ErrorBuffer.h`), and we RERUN with the room it asks for: what a
        failed pass wrote is thrown away, the outputs being brand new buffers anyway. The
        new capacity is `max( what is asked, twice what we had )` -- a capacity
        exceeded once tends to be exceeded again, so we make room rather than count.
        """
        from ..calls import lower_args, returned
        kwargs, output_attributes, scratch_attributes, input_exceptions, output_capacities, groups, returns, output_exceptions = lower_args( args )

        kernels = [ FfiCode( k ) if isinstance( k, str ) else k for k in kernels ]
        if not 1 <= len( kernels ) <= 2:
            raise ValueError( f"loom.ffi_call: expected one kernel (the forward) or two (forward, "
                              f"backward), got { len( kernels ) }" )
        code = Kernels( name, *kernels )

        prefix = name + "_"

        output_capacities = dict( output_capacities )   # ours to grow: the caller's dict is not ours to touch
        while True:
            ca = CallArgsAnalysis( kwargs, device, output_attributes, output_capacities, output_exceptions, input_exceptions, batch_alignment, scratch_attributes, groups, name, nb_items )
            ca.errors.call_name, ca.errors.failure_messages = name, dict( failures or {} )
            jax_ffi_call( code, ca, device, prefix )

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
                    jax.debug.callback( functools.partial( _raise_on_error, call_name = name, messages = dict( failures or {} ) ),
                                        ca.errors.raw )
                return returned( returns )

            if not overflows:
                return returned( returns )

            # `SDOT_DEBUG_CAPACITY=1`: what the kernel REALLY asked for, round by round. A
            # capacity that doubles endlessly is the symptom of a `wanted` that never arrives (error
            # buffer badly reset, unsynchronized read) rather than of a cell that
            # would keep growing -- and only this log tells the two apart.
            if env.flag( "DEBUG_CAPACITY" ):
                print( f"[capacity] { code.name }: " + ", ".join(
                    f"{ path } wanted={ wanted } capacity={ capacity }" for path, wanted, capacity in overflows ),
                    flush = True )

            for path, wanted, capacity in overflows:
                output_capacities[ path ] = max( wanted, 2 * capacity )

    @staticmethod
    def default_device_for( ftype ):
        platforms = { d.platform for d in jax.devices() }
        if "gpu" in platforms:
            # the card is there, but it is only a device for us if we can compile for it
            # (`device_is_present` asks the compiler): until the CUDA backend is ported, a GPU
            # machine works on its CPU.
            from ..devices.CudaGpu import CudaGpu
            gpu = CudaGpu( 0 )
            if gpu.device_is_present:
                return gpu

        # Metal (jax-metal) — auto-select when available; always uses FP32
        if "METAL" in platforms and ftype in ( None, "FP32" ):
            from ..devices.AppleGpu import AppleGpu
            return AppleGpu()

        from ..devices.Cpu import Cpu
        return Cpu()

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

    def defaults_for( self, device, ftype, itype ):
        """`( device, ftype, itype )` with what was left open (`None`) filled in for this framework."""
        if device is None:
            device = self.default_device_for( ftype )
        if itype is None:
            itype = self.default_itype_for( device )
        if ftype is None:
            ftype = self.default_ftype_for( device )
        return device, ftype, itype

    def configure( self, device, ftype, itype ):
        """Make the framework work the way these settings say: the sizes, and the device."""
        if itype.size == 64:
            jax.config.update( "jax_enable_x64", True )

        device.driver_version = device.driver_version_for_jax( jax.devices )

        # LOOM'S DEVICE IS THE DEVICE, for everything that does not say otherwise. Without this
        # line, `LOOM_DEVICE=cpu` only moved the buffers OF THE CALLS: a `jnp.array` built
        # on the side stayed on jax's default, which is the GPU as soon as there is one -- and the
        # first addition between the two fails with `Received incompatible devices for jitted
        # computation ... on platform CPU and ... on platform GPU`. The choice of device belongs to
        # the caller, not to half of its tensors.
        #
        # This is a GLOBAL jax config, set by a library: accepted, and it is already what
        # `jax_enable_x64` does just above. A tensor that names its device explicitly is not
        # affected ( `jax_default_device` only applies to the default ).
        jax.config.update( "jax_default_device", device.driver_version )

def _has_tracer( data ) -> bool:
    if isinstance( data, jax_core.Tracer ):
        return True
    if isinstance( data, ( list, tuple ) ):
        return any( _has_tracer( x ) for x in data )
    return False


def _raise_on_error( errors, call_name = "", messages = {} ):
    """The error buffer, checked at RUN time -- what is left when trace time cannot see it.

    Growing a capacity means running again, and that is a Python loop: it needs the count that did
    not fit, which under a trace only exists once the kernel has run. So inside a `jit` or a
    `vmap` a capacity has to be given generously -- and when it was not, this is what says so,
    rather than letting truncated results through. A FAILURE record ( `ErrorKind::failure` ) is
    raised with the message the call gave for its code ( `failures` )."""
    from .CallArg_Errors import KernelFailure, failure_message
    nb = min( int( errors[ 0 ] ), ( len( errors ) - 1 ) // 3 )
    records = [ tuple( int( v ) for v in errors[ 1 + 3 * i : 4 + 3 * i ] ) for i in range( nb ) ]
    message = failure_message( call_name, records, messages )
    if message is not None:
        raise KernelFailure( message )
    if int( errors[ 0 ] ) != 0:
        raise RuntimeError(
            "the kernel reported an error (a capacity too small, typically) from inside a traced "
            "call, where Python cannot grow one and run again: give the call a larger capacity."
        )
