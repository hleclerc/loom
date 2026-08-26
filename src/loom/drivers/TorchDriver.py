import numpy as np
import torch

from .TorchFramework import TorchFramework
from ..devices.Device import Device
from ..tensor.Dtype import Dtype

class TorchDriver:
    """

    """
    def __init__( self, framework: TorchFramework, device: Device | None, ftype: Dtype | None, itype: Dtype | None ):
        if device is None:
            device = TorchDriver.default_device_for( ftype )
        if itype is None:
            itype = TorchDriver.default_itype_for( device )
        if ftype is None:
            ftype = TorchDriver.default_ftype_for( device )

        #
        self.framework = framework
        self.device = device
        self.ftype  = ftype
        self.itype  = itype

        # resolve the driver spelling of both policy types. `_driver_version` (the FIELD), not
        # `driver_version` (a read-only property whose fallback asks the driver -- which is us,
        # and which is not built yet).
        assert ftype.floating_point == True
        assert itype.floating_point == False
        ftype._driver_version = self.driver_dtype_version( ftype.kind, ftype.size )
        itype._driver_version = self.driver_dtype_version( itype.kind, itype.size )

    def driver_dtype_version( self, kind, size ):
        """The torch dtype a `( kind, size )` denotes -- the Torch twin of `JaxDriver
        .driver_dtype_version`. `size is None` means "the driver's own" (`TF` / `TI`)."""
        from ..tensor.Dtype import REAL, SINT, UINT, BOOL

        if kind == BOOL:
            return torch.bool

        if kind == REAL:
            if size is None:
                return self.ftype._driver_version
            try:
                return { 16: torch.float16, 32: torch.float32, 64: torch.float64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported ftype size: { size }" )

        if kind == SINT:
            if size is None:
                return self.itype._driver_version
            try:
                return { 8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64 }[ size ]
            except KeyError:
                raise ValueError( f"unsupported itype size: { size }" )

        assert kind == UINT
        try:
            return { 8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.uint64 }[ size ]
        except KeyError:
            raise ValueError( f"unsupported unsigned itype size: { size }" )

    def dtype_of( self, x ):
        """The `Dtype` a torch buffer ACTUALLY has -- what a declaration is checked against.
        Read from torch's own flags, so no host sync and no numpy round-trip."""
        from ..tensor.Dtype import Dtype
        dt = x.dtype
        if dt == torch.bool:
            return Dtype.bo()
        if dt.is_floating_point:
            return Dtype.fp( size = 8 * dt.itemsize )
        return ( Dtype.si if dt.is_signed else Dtype.pi )( size = 8 * dt.itemsize )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a `Dtype`); a no-op when it already is."""
        from ..tensor.Dtype import Dtype
        return x.to( Dtype.factory( dtype ).driver_version )


    @staticmethod
    def default_device_for( ftype ):
        # cuda
        if torch.cuda.is_available():
            from .CudaGpu import CudaGpu
            return CudaGpu( 0 )

        # Metal (MPS) only supports FP32
        if torch.backends.mps.is_available() and ftype == "FP32":
            from .AppleGpu import AppleGpu
            return AppleGpu()

        # cpu
        from .Cpu import Cpu
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









    @property
    def array_type( self ):
        return torch.Tensor

    @property
    def int_type( self ):
        return torch.int64

    _INT_DTYPES = frozenset( [ torch.int8, torch.int16, torch.int32, torch.int64,
                                torch.uint8, torch.uint16, torch.uint32, torch.uint64 ] )

    def is_int_dtype( self, dtype ):
        return dtype in TorchDriver._INT_DTYPES

    def any_requires_grad( self, tensors ) -> bool:
        return any( t.requires_grad for t in tensors )

    def t3( self, tensor ):
        """ make a rank 3 tensor """
        return self.tn( tensor, 3 )

    def t2( self, tensor ):
        """ make a rank 2 tensor """
        return self.tn( tensor, 2 )

    def t1( self, tensor ):
        """ make a rank 1 tensor """
        return self.tn( tensor, 1 )

    def t0( self, tensor ):
        """ make a rank 0 tensor """
        return self.tn( tensor, 0 )

    def tn( self, tensor, ndim = None, name = None, dtype = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor

        if dtype is None or dtype is float:
            dtype = self.dtype
        elif dtype is int:
            dtype = self.int_type
        else:
            raise NotImplementedError( f"for { dtype }" )

        res = torch.as_tensor( tensor, dtype = dtype, device = self.device )

        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

        return res

    def zeros( self, shape, dtype = None ):
        return torch.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def full( self, shape, value, dtype = None ):
        return torch.full( tuple( shape ), value, dtype = dtype or self.dtype, device = self.device )

    def ones( self, shape, dtype = None ):
        return torch.ones( shape, dtype = dtype or self.dtype, device = self.device )

    def linspace( self, a, b, n, dtype = None ):
        return torch.linspace( a, b, n, dtype = dtype or self.dtype, device = self.device )

    def empty( self, shape, dtype = None ):
        return torch.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return tensor.unsqueeze( index )

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

    def stack( self, tensors, axis ):
        return torch.stack( tensors, dim=axis )

    # A symbolic zero: a SHAPED, TYPED, BUFFERLESS value read as 0 (see `JaxDriver.symbolic_zero`).
    # Torch has no native one, but a `meta` tensor is exactly that -- shape/dtype, no storage (any
    # materialization raises), recognizable by `is_meta`.
    def symbolic_zero( self, shape, dtype = None ):
        return torch.zeros( tuple( shape ), dtype = dtype or self.dtype, device = "meta" )

    def is_symbolic_zero( self, x ):
        return isinstance( x, torch.Tensor ) and x.is_meta

    # see `JaxDriver.is_traced`: Torch's autograd does not retrace a Python body the way
    # `lax.scan`'s differentiation rule does, so a `ComputedAttribute` cache never goes stale
    # under Torch -- always `False`, keeping today's eager-cache behavior unchanged.
    def is_traced( self, x ):
        return False

    # see `JaxDriver.checkpoint`: `func` re-evaluated in the backward instead of taping its
    # intermediates. Torch spells it as a CALL wrapper rather than a decorator, so we adapt it to
    # the same "function -> function" verb. `use_reentrant = False` is the non-deprecated
    # implementation (the only one that supports closed-over tensors and nested autograd).
    def checkpoint( self, func ):
        import torch.utils.checkpoint as torch_checkpoint
        return lambda *args: torch_checkpoint.checkpoint( func, *args, use_reentrant = False )

    # see `JaxDriver.fold`. Torch's autograd is eager: a Python loop over the leading axis IS the
    # sequential execution Jax needs a `lax.scan` to express, and it already keeps a single
    # iteration's buffers live (the rest being handed to `checkpoint`).
    def fold( self, body, init, xs ):
        leaves = xs if isinstance( xs, dict ) else { None: xs }
        n = len( next( iter( leaves.values() ) ) )
        carry = init
        for i in range( n ):
            x = { k: v[ i ] for k, v in leaves.items() } if isinstance( xs, dict ) else xs[ i ]
            carry = body( carry, x )
        return carry

    # reductions -- the backend-agnostic verbs `Tensor` reduces through (`axis` is
    # a dimension index or a tuple of them; `None` reduces everything to a scalar).
    # Torch spells the axis `dim` and rejects `dim=None`, so full reductions drop it.
    def sum( self, a, axis = None ):
        return torch.sum( a ) if axis is None else torch.sum( a, dim = axis )

    def prod( self, a, axis = None ):
        return torch.prod( a ) if axis is None else torch.prod( a, dim = axis )

    def max( self, a, axis = None ):
        return torch.max( a ) if axis is None else torch.amax( a, dim = axis )

    def min( self, a, axis = None ):
        return torch.min( a ) if axis is None else torch.amin( a, dim = axis )

    def mean( self, a, axis = None ):
        return torch.mean( a ) if axis is None else torch.mean( a, dim = axis )

    def all( self, a, axis = None ):
        return torch.all( a ) if axis is None else torch.all( a, dim = axis )

    def any( self, a, axis = None ):
        return torch.any( a ) if axis is None else torch.any( a, dim = axis )

    def where( self, cond, a, b ):
        cond = torch.as_tensor( cond, device = getattr( b, "device", None ) )
        return torch.where( cond, torch.as_tensor( a, dtype = getattr( b, "dtype", None ), device = getattr( b, "device", None ) ), b )

    # elementwise math -- the backend-agnostic verbs `Tensor` maps through (shape preserved).
    def sqrt( self, a ):
        return torch.sqrt( a )

    def arcsin( self, a ):
        return torch.asin( a )

    def clip( self, a, lo = None, hi = None ):
        return torch.clamp( a, min = lo, max = hi )

    def linalg_solve( self, A, b ):
        return torch.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return torch.moveaxis( tensor, source, destination )

    def transpose( self, a, axes = None ):
        if axes is None:
            axes = tuple( reversed( range( a.ndim ) ) )
        return a.permute( *axes )

    def hstack( self, lst ):
        return torch.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def to_nanobind_compatible_objects( self, obj ):
        if isinstance( obj, torch.Tensor ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        return None

    def forward( self, forward_func: callable, backward_func: callable, args: list, input_tensors: list, _output_args: list, output_tensors: list ):
        """Differentiable wrapper.
            forward_func ( *args ) -> None  (fills output_tensors in-place)
            backward_func( *args, *grad_inputs, *grad_outputs ) -> None  (fills grad_inputs in-place)
        """
        _driver = self
        _output = []

        class Func( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, *_ ):
                _output.append( forward_func( *args ) )
                return tuple( output_tensors )

            @staticmethod
            def backward( ctx, *grad_outputs ):
                grad_inputs = [ _driver.zeros( t.shape ) for t in input_tensors ]
                backward_func( *args, *grad_inputs, *grad_outputs )
                return tuple( grad_inputs )

        Func.apply( *input_tensors )
        return _output[ 0 ]

        # tracked = Func.apply( *input_tensors )
        # if not output_tensors:
        #     return None
        # return tracked[ 0 ] if len( output_tensors ) == 1 else tracked


    def array_conversion( self, value ):
        """ Ensure that every array of value is of the right type """
        if isinstance( value, self.array_type ):
            return value
        if isinstance( value, np.ndarray ):
            return self.tn( value )
        if isinstance( value, list ):
            return [ self.array_conversion( v ) for v in value ]
        if isinstance( value, tuple ):
            return tuple( self.array_conversion( v ) for v in value )
        if hasattr( value, "array_conversion" ):
            return value.array_conversion()
        raise NotImplementedError


    # def plan( self, bindings, f: BatchOfDistributions, g: BatchOfDistributions ):
    #     class SDOTFunction( torch.autograd.Function ):
    #         @staticmethod
    #         def forward( ctx, dirac_xs, *args ):
    #             # get constants
    #             batch_size = dirac_xs.shape[ 0 ]
    #             nb_diracs = dirac_xs.shape[ 1 ]
    #             dim = dirac_xs.shape[ 2 ]

    #             # room for the outputs
    #             barycenters = self.empty( [ batch_size, nb_diracs, dim ] )
    #             potentials = self.empty( [ batch_size, nb_diracs ] )
    #             distances = self.empty( [ batch_size ] )
    #             cuts = self.empty( [ batch_size, nb_diracs, 2 ] )

    #             # arguments as expected by the binding
    #             binding_inputs = unflatten_args( f, g, [ dirac_xs ] + list( args ) )

    #             # call the C++ procedure
    #             bindings.forward( *binding_inputs, distances, barycenters, potentials, cuts )

    #             ctx.save_for_backward( dirac_xs, *args, distances, barycenters, potentials, cuts )

    #             return distances, barycenters, potentials, cuts

    #         @staticmethod
    #         def backward( ctx, grad_distance, grad_barycenters, grad_potentials, grad_cuts ):
    #             # get room for the output gradients
    #             flat_grad_outputs = [ self.empty( arg.shape ) for arg in ctx.saved_tensors[ :-4 ] ]

    #             # arguments as expected by the binding
    #             binding_grad_outputs = unflatten_args( f, g, flat_grad_outputs )
    #             binding_inputs = unflatten_args( f, g, ctx.saved_tensors[ :-4 ] )

    #             bindings.backward( *binding_inputs, *ctx.saved_tensors[ -4: ], grad_distance, grad_barycenters, grad_potentials, grad_cuts, *binding_grad_outputs )

    #             return tuple( flat_grad_outputs )

    #     input_tensors = flat_tensor_list( f ) + flat_tensor_list( g )
    #     outputs = SDOTFunction.apply( *input_tensors )
    #     assert isinstance( outputs, tuple )
    #     return BatchOfOtPlans( *outputs )

    def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
        """ small helper to optimize `loss` wrt `params` using L-BFGS.
            - `params`  : torch tensor or list of torch tensors
            - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
            Returns the optimized params (same type as input).
        """
        is_list = isinstance( params, ( list, tuple ) )
        p_list  = list( params ) if is_list else [ params ]

        for p in p_list:
            p.requires_grad_( True )

        lbfgs = torch.optim.LBFGS( p_list, history_size=15, max_iter=10 )

        def closure():
            lbfgs.zero_grad()
            objective = loss( p_list if is_list else p_list[ 0 ] )
            objective.backward()
            return objective

        for i in range( max_iter ):
            lbfgs.step( closure )

            with torch.no_grad():
                grad_norm = float( torch.norm( torch.stack( [ torch.norm( p.grad ) for p in p_list if p.grad is not None ] ) ) )

                if on_iter:
                    on_iter( p_list if is_list else p_list[ 0 ], i, grad_norm )

                if grad_norm < tol_grad:
                    break

        return p_list if is_list else p_list[ 0 ]

    def optimize_using_sgd( self, loss, params ):
        optimizer = torch.optim.SGD( [ params ] )
        for _ in range( 2000 ):
            l = loss( params )
            print( l )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
