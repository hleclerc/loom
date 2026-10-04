from typing import Any
import numpy

from .NumpyFramework import NumpyFramework
from .NumpyFfi import call as ffi_call
from .CallArgsAnalysis import CallArgsAnalysis
from ..compilation.FfiCode import FfiCode, Kernels
from ..devices.Device import Device
from ..tensor.Dtype import Dtype
from ..util.info import info
from .. import env


class SymbolicZero:
    """A SHAPED, TYPED, BUFFERLESS value that reads as 0 (see `JaxDriver.symbolic_zero`). Nothing
    in a forward-only driver produces one by itself, but a caller may mint one."""

    def __init__( self, shape, dtype ):
        self.shape = tuple( shape )
        self.dtype = numpy.dtype( dtype )
        self.ndim = len( self.shape )


class NumpyDriver:
    """
    numpy implementation: the FORWARD driver, for a machine that has neither jax nor torch.

    Kernels are the very same C++ as under Jax, run on the CPU queue (see `NumpyFfi`). What numpy
    cannot give is a tape, so `vmap`, `grad` and `vjp` raise and `jit` is the identity -- a
    function that is jitted still computes the same values, only without being compiled. A call
    runs its forward kernel only; its backward kernel, if it has one, is never used.
    """

    device: Any

    def __init__( self, framework: NumpyFramework, device: Device | None, ftype: Dtype | None, itype: Dtype | None ):
        if device is None:
            from ..devices.Cpu import Cpu
            device = Cpu()
        if not device.is_cpu:
            raise ValueError( f"the numpy driver runs on the CPU only (asked for { device }): use jax or torch for a GPU" )
        if itype is None:
            itype = NumpyDriver.default_itype_for( device )
        if ftype is None:
            ftype = NumpyDriver.default_ftype_for( device )

        self.framework = framework
        self.device = device
        self.ftype  = ftype
        self.itype  = itype

        itype._driver_version = self.driver_dtype_version( itype.kind, itype.size )
        ftype._driver_version = self.driver_dtype_version( ftype.kind, ftype.size )
        assert itype.floating_point == False
        assert ftype.floating_point == True

        # numpy has no device objects: the CPU is the host
        device.driver_version = None

    def driver_dtype_version( self, kind, size ):
        """The numpy dtype a `( kind, size )` denotes. `size is None` means "the driver's own",
        which is exactly what `TF` / `TI` are -- resolved here, once the driver exists."""
        from ..tensor.Dtype import REAL, SINT, UINT, BOOL

        if kind == BOOL:
            return numpy.bool_

        if kind == REAL:
            if size is None:
                return self.ftype._driver_version
            try:
                return { 16: numpy.float16, 32: numpy.float32, 64: numpy.float64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported ftype size: { size }" )

        if kind == SINT:
            if size is None:
                return self.itype._driver_version
            try:
                return { 8: numpy.int8, 16: numpy.int16, 32: numpy.int32, 64: numpy.int64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported itype size: { size }" )

        assert kind == UINT
        try:
            return { 8: numpy.uint8, 16: numpy.uint16, 32: numpy.uint32, 64: numpy.uint64 }[ size ]
        except KeyError:
            raise ValueError( f"unsupported unsigned itype size: { size }" )

    def dtype_of( self, x ):
        """The `Dtype` a numpy buffer ACTUALLY has -- what a declaration is checked against."""
        return Dtype.from_numpy( x.dtype )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a `Dtype`); a no-op when it already is."""
        return numpy.asarray( x, dtype = Dtype.factory( dtype ).driver_version )

    @staticmethod
    def default_ftype_for( device: Device ):
        return Dtype.fp( 64 )

    @staticmethod
    def default_itype_for( device: Device ):
        return Dtype.si( 64 )

    @property
    def available_gpus( self ):
        return 0

    # -- building blocks --
    def array( self, data, dtype = None, device = None ):
        if data is None:
            return None
        return numpy.asarray( data, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def zeros( self, shape, dtype = None ):
        return numpy.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def full( self, shape, value, dtype = None ):
        return numpy.full( shape, value, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def ones( self, shape, dtype = None ):
        return numpy.ones( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def arange( self, nb, dtype = None ):
        return numpy.arange( nb, dtype = Dtype.factory( dtype or self.itype ).driver_version )

    def linspace( self, a, b, nb, dtype = None ):
        return numpy.linspace( a, b, nb, dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def reshape( self, tensor, shape ):
        return numpy.reshape( tensor, tuple( shape ) )

    def stack( self, tensors, axis = 0 ):
        return numpy.stack( tensors, axis = axis )

    def pad( self, tensor, pad_width ):
        return numpy.pad( tensor, pad_width )

    def transpose( self, a, axes = None ):
        return numpy.transpose( a, axes )

    # -- transformations: there is no tape and no tracer here --
    def vmap( self, func ):
        raise NotImplementedError( "the numpy driver is forward-only: no `vmap` (use jax)" )

    def grad( self, func, argnums = 0 ):
        raise NotImplementedError( "the numpy driver is forward-only: no `grad` (use jax or torch)" )

    def vjp( self, func, *primals ):
        raise NotImplementedError( "the numpy driver is forward-only: no `vjp` (use jax or torch)" )

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

    def random( self, shape, dtype = None, seed = None ):
        """A uniform draw. `seed = None` takes the next value of a process-wide COUNTER, so that
        two consecutive draws differ; an explicit `seed` makes the draw reproducible."""
        if seed is None:
            seed = getattr( self, "_rng_seed", 0 )
            self._rng_seed = seed + 1
        return numpy.random.default_rng( seed ).random( tuple( shape ), dtype = Dtype.factory( dtype or self.ftype ).driver_version )

    def symbolic_zero( self, shape, dtype = None ):
        return SymbolicZero( shape, Dtype.factory( dtype or self.ftype ).driver_version )

    def is_symbolic_zero( self, x ):
        return isinstance( x, SymbolicZero )

    def is_traced( self, x ):
        return False

    def stop_gradient( self, x ):
        return x

    # -- reductions and elementwise verbs `Tensor` goes through --
    def sum( self, a, axis = None ):
        return numpy.sum( a, axis = axis )

    def prod( self, a, axis = None ):
        return numpy.prod( a, axis = axis )

    def cumsum( self, a, axis ):
        return numpy.cumsum( a, axis = axis )

    def concatenate( self, arrays, axis = 0 ):
        return numpy.concatenate( list( arrays ), axis = axis )

    def max( self, a, axis = None ):
        return numpy.max( a, axis = axis )

    def min( self, a, axis = None ):
        return numpy.min( a, axis = axis )

    def mean( self, a, axis = None ):
        return numpy.mean( a, axis = axis )

    def all( self, a, axis = None ):
        return numpy.all( a, axis = axis )

    def any( self, a, axis = None ):
        return numpy.any( a, axis = axis )

    def where( self, cond, a, b ):
        return numpy.where( cond, a, b )

    def sqrt( self, a ):
        return numpy.sqrt( a )

    def arcsin( self, a ):
        return numpy.arcsin( a )

    def clip( self, a, lo = None, hi = None ):
        return numpy.clip( a, lo, hi )

    # -- the call --
    def call( self, name, *kernels, nb_items = None, batch_alignment = None, has_dynamic_capacity = True, failures = None, **args ):
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
            raise ValueError( f"driver.call: expected one kernel (the forward) or two (forward, "
                              f"backward), got { len( kernels ) }" )
        code = Kernels( name, *kernels )

        prefix = name + "_"

        output_capacities = dict( output_capacities )   # ours to grow: the caller's dict is not ours to touch
        while True:
            ca = CallArgsAnalysis( kwargs, self.device, output_attributes, output_capacities, output_exceptions, input_exceptions, batch_alignment, scratch_attributes, groups, name, nb_items )
            ca.errors.call_name, ca.errors.failure_messages = name, dict( failures or {} )
            ffi_call( code, ca, self.device, prefix )

            overflows = ca.capacity_overflows()
            if not overflows:
                return returned( returns )

            if env.flag( "DEBUG_CAPACITY" ):
                print( f"[capacity] { code.name }: " + ", ".join(
                    f"{ path } wanted={ wanted } capacity={ capacity }" for path, wanted, capacity in overflows ),
                    flush = True )

            for path, wanted, capacity in overflows:
                output_capacities[ path ] = max( wanted, 2 * capacity )
