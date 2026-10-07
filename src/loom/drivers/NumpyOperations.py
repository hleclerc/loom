"""The numpy framework's operations (see `NumpyFramework`)."""
from ..devices.Device import Device
from .CallArgsAnalysis import CallArgsAnalysis
from ..compilation.FfiCode import FfiCode, Kernels
from ..util.info import info
from .. import env
import numpy
from ..tensor.Dtype import Dtype


class SymbolicZero:
    """A SHAPED, TYPED, BUFFERLESS value that reads as 0 (see `JaxDriver.symbolic_zero`). Nothing
    in a forward-only driver produces one by itself, but a caller may mint one."""

    def __init__( self, shape, dtype ):
        self.shape = tuple( shape )
        self.dtype = numpy.dtype( dtype )
        self.ndim = len( self.shape )


class NumpyOperations:
    """What the numpy framework DOES with an array that already exists: its operations, its
    transforms, its readings, its way of building one from a concrete dtype and an explicit device, and of
    launching a kernel. It has no default of its own: the defaults are loom's (`Settings`), and are handed in."""

    # What a driver over another array module (`CupyDriver`) changes: the module, the device it runs on, and
    # the engine that launches the kernels. Everything below goes through them.
    xp = numpy

    def matmul( self, a, b ):
        return self.xp.asarray( a ) @ self.xp.asarray( b )

    def to_numpy( self, x ):
        return numpy.asarray( x )

    def dtype_of( self, x ):
        """The `Dtype` a numpy buffer ACTUALLY has -- what a declaration is checked against."""
        return Dtype.from_numpy( x.dtype )

    def reshape( self, tensor, shape ):
        return self.xp.reshape( tensor, tuple( shape ) )

    def stack( self, tensors, axis = 0 ):
        return self.xp.stack( tensors, axis = axis )

    def pad( self, tensor, pad_width ):
        return self.xp.pad( tensor, pad_width )

    def transpose( self, a, axes = None ):
        return self.xp.transpose( a, axes )

    def jit( self, func ):
        """The identity: same values, only not compiled."""
        return func

    def checkpoint( self, func ):
        """The identity: without a tape there is nothing to recompute."""
        return func

    def fold( self, body, init, xs ):
        """`body( carry, x )` applied sequentially over the leading axis of `xs` (an array, or a
        dict of arrays sharing that axis), returning the final carry."""
        leaves = xs if isinstance( xs, dict ) else { None: xs }
        n = len( next( iter( leaves.values() ) ) )
        carry = init
        for i in range( n ):
            carry = body( carry, { k: v[ i ] for k, v in leaves.items() } if isinstance( xs, dict ) else xs[ i ] )
        return carry

    def is_symbolic_zero( self, x ):
        return isinstance( x, SymbolicZero )

    def is_traced( self, x ):
        return False

    # see `JaxOperations.concrete_eval`: nothing is traced here, so there is nothing to escape
    def concrete_eval( self ):
        import contextlib
        return contextlib.nullcontext()

    def stop_gradient( self, x ):
        return x

    # -- reductions and elementwise verbs `Tensor` goes through --
    def sum( self, a, axis = None ):
        return self.xp.sum( a, axis = axis )

    def prod( self, a, axis = None ):
        return self.xp.prod( a, axis = axis )

    def cumsum( self, a, axis ):
        return self.xp.cumsum( a, axis = axis )

    def concatenate( self, arrays, axis = 0 ):
        return self.xp.concatenate( list( arrays ), axis = axis )

    def max( self, a, axis = None ):
        return self.xp.max( a, axis = axis )

    def min( self, a, axis = None ):
        return self.xp.min( a, axis = axis )

    def mean( self, a, axis = None ):
        return self.xp.mean( a, axis = axis )

    def all( self, a, axis = None ):
        return self.xp.all( a, axis = axis )

    def any( self, a, axis = None ):
        return self.xp.any( a, axis = axis )

    def where( self, cond, a, b ):
        return self.xp.where( cond, a, b )

    def sqrt( self, a ):
        return self.xp.sqrt( a )

    def arcsin( self, a ):
        return self.xp.arcsin( a )

    def exp( self, a ):
        return self.xp.exp( a )

    def clip( self, a, lo = None, hi = None ):
        return self.xp.clip( a, lo, hi )

    @property
    def available_gpus( self ):
        return 0

    # ---- what a framework calls a dtype -----------------------------------------------------------------
    def dtype_version( self, dtype ):
        """The numpy dtype a CONCRETE `Dtype` denotes: its size is the caller's to have filled in (the
        defaults are loom's, not a framework's)."""
        from ..tensor.Dtype import Dtype, REAL, SINT, UINT, BOOL
        dtype = Dtype.factory( dtype )
        if dtype.kind == BOOL:
            return numpy.bool_
        if dtype.size is None:
            raise ValueError( f"the size of { dtype.cpp_name } is not resolved: it is loom's default to fill in, not a framework's" )
        table = { REAL: { 16: numpy.float16, 32: numpy.float32, 64: numpy.float64 },
                  SINT: { 8: numpy.int8, 16: numpy.int16, 32: numpy.int32, 64: numpy.int64 },
                  UINT: { 8: numpy.uint8, 16: numpy.uint16, 32: numpy.uint32, 64: numpy.uint64 } }[ dtype.kind ]
        try:
            return table[ dtype.size ]
        except KeyError:
            raise ValueError( f"unsupported size for { dtype.cpp_name }" )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a concrete `Dtype`); a no-op when it already is."""
        return self.xp.asarray( x, dtype = self.dtype_version( dtype ) )

    # ---- building a value, with a concrete dtype (see `DefaultSizes` for the defaults) ------------------------
    def array( self, data, dtype, device = None ):
        if data is None:
            return None
        return self.xp.asarray( data, dtype = self.dtype_version( dtype ) )

    def zeros( self, shape, dtype, device = None ):
        return self.xp.zeros( shape, dtype = self.dtype_version( dtype ) )

    def full( self, shape, value, dtype, device = None ):
        return self.xp.full( shape, value, dtype = self.dtype_version( dtype ) )

    def ones( self, shape, dtype, device = None ):
        return self.xp.ones( shape, dtype = self.dtype_version( dtype ) )

    def arange( self, nb, dtype, device = None ):
        return self.xp.arange( nb, dtype = self.dtype_version( dtype ) )

    def linspace( self, a, b, nb, dtype, device = None ):
        return self.xp.linspace( a, b, nb, dtype = self.dtype_version( dtype ) )

    def random( self, shape, dtype, seed = None, device = None ):
        """A uniform draw. `seed = None` takes the next value of a process-wide COUNTER, so that
        two consecutive draws differ; an explicit `seed` makes the draw reproducible."""
        if seed is None:
            seed = getattr( self, "_rng_seed", 0 )
            self._rng_seed = seed + 1
        return self.xp.random.default_rng( seed ).random( tuple( shape ), dtype = self.dtype_version( dtype ) )

    def symbolic_zero( self, shape, dtype ):
        return SymbolicZero( shape, self.dtype_version( dtype ) )

    def _default_device( self ):
        from ..devices.Cpu import Cpu
        return Cpu()

    def _check_device( self, device ):
        if not device.is_cpu:
            raise ValueError( f"the numpy driver runs on the CPU only (asked for { device }): use jax or torch for a GPU" )

    @staticmethod
    def _ffi():
        """The module whose `call` runs a kernel, and the `PointerAdapter` it speaks."""
        from . import NumpyFfi
        return NumpyFfi

    def is_strided( self, x ):
        """Is `x` held otherwise than dense row-major: a view the kernel reads with its own strides."""
        return self._ffi()._adapter.is_strided( x )

    # -- transformations: there is no tape and no tracer here --
    def vmap( self, func ):
        raise NotImplementedError( f"the numpy driver is forward-only: no `vmap` (use jax or torch)" )

    def grad( self, func, argnums = 0 ):
        raise NotImplementedError( f"the numpy driver is forward-only: no `grad` (use jax or torch)" )

    def vjp( self, func, *primals ):
        raise NotImplementedError( f"the numpy driver is forward-only: no `vjp` (use jax or torch)" )

    # -- the call --
    def call( self, device, name, *kernels, nb_items = None, batch_alignment = None, has_dynamic_capacity = True, failures = None, **args ):
        """Runs an `FfiCode` on the values passed as kwargs -- the same thing as
        `JaxDriver.call` (see `loom/calls.py` for the argument vocabulary), minus the
        backward: the forward kernel runs, the adjoint is unused.

        A capacity that turns out too small reruns the call with the requested room, exactly
        as under Jax -- and here nothing traces, so this always applies.
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
            self._ffi().call( code, ca, device, prefix )

            overflows = ca.capacity_overflows()
            if not overflows:
                return returned( returns )

            if env.flag( "DEBUG_CAPACITY" ):
                print( f"[capacity] { code.name }: " + ", ".join(
                    f"{ path } wanted={ wanted } capacity={ capacity }" for path, wanted, capacity in overflows ),
                    flush = True )

            for path, wanted, capacity in overflows:
                output_capacities[ path ] = max( wanted, 2 * capacity )

    @staticmethod
    def default_ftype_for( device: Device ):
        return Dtype.fp( 64 )

    @staticmethod
    def default_itype_for( device: Device ):
        return Dtype.si( 64 )

    def defaults_for( self, device, ftype, itype ):
        """`( device, ftype, itype )` with what was left open (`None`) filled in for this framework."""
        if device is None:
            device = self._default_device()
        if itype is None:
            itype = self.default_itype_for( device )
        if ftype is None:
            ftype = self.default_ftype_for( device )
        return device, ftype, itype

    def configure( self, device, ftype, itype ):
        self._check_device( device )
        # numpy has no device objects: the CPU is the host
        device.driver_version = None
