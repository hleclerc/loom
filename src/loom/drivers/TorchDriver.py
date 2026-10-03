import numpy as np
import torch

from .TorchFramework import TorchFramework
from .CallArgsAnalysis import CallArgsAnalysis
from ..compilation.FfiCode import FfiCode, Kernels
from ..devices.Device import Device
from ..tensor.Dtype import Dtype
from .. import env

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
        # torch spelling of the loom device (a loom `Device` knows nothing about torch)
        if device.is_cpu:
            self.torch_device = torch.device( "cpu" )
        elif device.is_apple_gpu:
            self.torch_device = torch.device( "mps" )
        else:
            self.torch_device = torch.device( "cuda", getattr( device, "device_id", 0 ) )

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
        Read from the dtype alone, so no host sync and no numpy round-trip."""
        from ..tensor.Dtype import Dtype
        dt = x.dtype
        return Dtype.from_numpy( dt )       # a torch dtype or a numpy one: `Dtype` reads both

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a `Dtype`); a no-op when it already is."""
        from ..tensor.Dtype import Dtype
        return x.to( Dtype.factory( dtype ).driver_version )


    @staticmethod
    def default_device_for( ftype ):
        # cuda -- only if we can compile for it (see `JaxDriver.default_device_for`)
        if torch.cuda.is_available():
            from ..devices.CudaGpu import CudaGpu
            gpu = CudaGpu( 0 )
            if gpu.device_is_present:
                return gpu

        # Metal (MPS) only supports FP32
        if torch.backends.mps.is_available() and ftype == "FP32":
            from ..devices.AppleGpu import AppleGpu
            return AppleGpu()

        # cpu
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









    @property
    def available_gpus( self ):
        return torch.cuda.device_count()

    # -- building blocks --
    def array( self, data, dtype = None, device = None ):
        if data is None:
            return None
        dtype = Dtype.factory( dtype or self.ftype )
        if isinstance( data, torch.Tensor ):
            return data.to( dtype = dtype.driver_version, device = self.torch_device )
        # through numpy: it reads what torch cannot (a `ShapeArray`, nested lists of numpy scalars),
        # and torch refuses a negative stride
        return torch.as_tensor( np.asarray( data, dtype = dtype.numpy_dtype, order = "C" ), device = self.torch_device )

    def pad( self, tensor, pad_width ):
        # numpy's `pad_width` ( one `( before, after )` per axis, first axis first ) -> torch's flat
        # list, LAST axis first
        flat = [ n for before_after in reversed( list( pad_width ) ) for n in before_after ]
        return torch.nn.functional.pad( tensor, flat )

    # -- transformations --
    @staticmethod
    def _leaves( tree ):
        import torch.utils._pytree as pytree
        return pytree.tree_flatten( tree )

    def _tracked( self, tree ):
        """`tree` with every floating leaf replaced by a fresh LEAF that requires grad, so the
        function sees a graph that starts here (and the caller's tensors are never modified)."""
        import torch.utils._pytree as pytree
        leaves, spec = pytree.tree_flatten( tree )
        leaves = [ torch.as_tensor( x ) if isinstance( x, np.ndarray ) else x for x in leaves ]     # numpy is a valid input, as for jax
        leaves = [ x.detach().requires_grad_( True ) if isinstance( x, torch.Tensor ) and x.is_floating_point() else x for x in leaves ]
        return pytree.tree_unflatten( leaves, spec ), leaves

    def vmap( self, func ):
        """Map `func` over a new leading axis -- by looping over it and stacking the results. Same
        values as `jax.vmap`; a `driver.call` inside is replayed item by item (no batched kernel)."""
        def mapped( *args ):
            import torch.utils._pytree as pytree
            leaves, spec = pytree.tree_flatten( args )
            n = next( x.shape[ 0 ] for x in leaves if isinstance( x, torch.Tensor ) )
            outs = [ func( *pytree.tree_unflatten( [ x[ i ] if isinstance( x, torch.Tensor ) else x for x in leaves ], spec ) ) for i in range( n ) ]
            out_leaves, out_spec = pytree.tree_flatten( outs[ 0 ] )
            columns = [ pytree.tree_flatten( o )[ 0 ] for o in outs ]
            return pytree.tree_unflatten( [ torch.stack( [ c[ k ] for c in columns ] ) for k in range( len( out_leaves ) ) ], out_spec )
        return mapped

    def grad( self, func, argnums = 0 ):
        """Gradient of a scalar-valued `func` wrt the argument(s) `argnums`. A `driver.call` inside
        reaches its backward kernel through the `torch.autograd.Function` the call registers (see
        `TorchFfi._call_with_autograd`)."""
        import torch.utils._pytree as pytree
        def gradient( *args ):
            with torch.enable_grad():
                nums = ( argnums, ) if isinstance( argnums, int ) else tuple( argnums )
                args = list( args )
                wrt = []
                for n in nums:
                    args[ n ], leaves = self._tracked( args[ n ] )
                    wrt.append( ( n, leaves ) )
                out = func( *args )
                flat = [ x for _, leaves in wrt for x in leaves if isinstance( x, torch.Tensor ) and x.requires_grad ]
                grads = iter( torch.autograd.grad( out, flat, allow_unused = True ) )
                res = []
                for n, leaves in wrt:
                    gl = []
                    for x in leaves:
                        if isinstance( x, torch.Tensor ) and x.requires_grad:
                            g = next( grads )
                            gl.append( torch.zeros_like( x ) if g is None else g )
                        else:
                            gl.append( None )
                    res.append( pytree.tree_unflatten( gl, pytree.tree_flatten( args[ n ] )[ 1 ] ) )
                return res[ 0 ] if isinstance( argnums, int ) else tuple( res )
        return gradient

    def vjp( self, func, *primals ):
        """`( func( *primals ), pullback )` -- see `JaxDriver.vjp`. `pullback( cotangent )` gives one
        gradient per primal (a symbolic-zero cotangent contributes nothing)."""
        import torch.utils._pytree as pytree
        with torch.enable_grad():
            tracked, leaves = self._tracked( primals )
            out = func( *tracked )
        out_leaves, out_spec = pytree.tree_flatten( out )

        def pullback( cotangent ):
            cts = [ torch.as_tensor( c ) if isinstance( c, np.ndarray ) else c for c in pytree.tree_flatten( cotangent )[ 0 ] ]
            pairs = [ ( o, c ) for o, c in zip( out_leaves, cts ) if o.requires_grad and not self.is_symbolic_zero( c ) ]
            flat = [ x for x in leaves if isinstance( x, torch.Tensor ) and x.requires_grad ]
            if pairs and flat:
                grads = iter( torch.autograd.grad( [ o for o, _ in pairs ], flat, [ c for _, c in pairs ], allow_unused = True, retain_graph = True ) )
            else:
                grads = iter( [ None ] * len( flat ) )
            res = []
            for x in leaves:
                if isinstance( x, torch.Tensor ) and x.requires_grad:
                    g = next( grads )
                    res.append( torch.zeros_like( x ) if g is None else g )
                else:
                    res.append( None )
            return pytree.tree_unflatten( res, pytree.tree_flatten( tracked )[ 1 ] )

        return pytree.tree_unflatten( [ o.detach() if isinstance( o, torch.Tensor ) else o for o in out_leaves ], out_spec ), pullback

    def jit( self, func ):
        """The identity: same values, only not compiled (`torch.compile` does not see the kernels)."""
        return func

    def stop_gradient( self, x ):
        return x.detach() if isinstance( x, torch.Tensor ) else x

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
            dtype = self.ftype.driver_version
        elif dtype is int:
            dtype = self.itype.driver_version
        else:
            raise NotImplementedError( f"for { dtype }" )

        res = torch.as_tensor( tensor, dtype = dtype, device = self.torch_device )

        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

        return res

    # the factories. They receive a loom `Dtype` (this is what `Tensor.zeros` & co pass them, their
    # DECLARED dtype) and translate it, like `astype`: passing it as is to torch --
    # which is what `dtype or self.dtype` did, on an attribute that does not even exist -- made
    # `RealTensor[ ... ].full( 0.0 )` fail under this driver only.
    def _dt( self, dtype, integer = False ):
        from ..tensor.Dtype import Dtype
        return Dtype.factory( dtype or ( self.itype if integer else self.ftype ) ).driver_version

    def zeros( self, shape, dtype = None ):
        return torch.zeros( tuple( shape ), dtype = self._dt( dtype ), device = self.torch_device )

    def full( self, shape, value, dtype = None ):
        return torch.full( tuple( shape ), value, dtype = self._dt( dtype ), device = self.torch_device )

    def ones( self, shape, dtype = None ):
        return torch.ones( tuple( shape ), dtype = self._dt( dtype ), device = self.torch_device )

    def arange( self, nb, dtype = None ):
        return torch.arange( nb, dtype = self._dt( dtype, integer = True ), device = self.torch_device )

    def linspace( self, a, b, n, dtype = None ):
        # numpy's formula, not torch's (they differ in the last bit, and a grid should be the same
        # grid whatever the framework)
        return torch.as_tensor( np.linspace( a, b, int( n ) ), dtype = self._dt( dtype ), device = self.torch_device )

    def reshape( self, tensor, shape ):
        return tensor.reshape( tuple( shape ) )

    def random( self, shape, dtype = None, seed = None ):
        """A uniform draw, same contract as `JaxDriver.random`: `seed = None` advances a
        process counter, an explicit seed makes the draw reproducible."""
        if seed is None:
            seed = getattr( self, "_rng_seed", 0 )
            self._rng_seed = seed + 1
        generator = torch.Generator( device = self.torch_device ).manual_seed( int( seed ) )
        return torch.rand( tuple( shape ), generator = generator, dtype = self._dt( dtype ), device = self.torch_device )

    def empty( self, shape, dtype = None ):
        return torch.zeros( tuple( shape ), dtype = self._dt( dtype ), device = self.torch_device )

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
        return torch.zeros( tuple( shape ), dtype = self._dt( dtype ), device = "meta" )

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
    def _reduced( self, fn, a, axis ):
        """`fn( a, dim = axis )` for a torch reduction that takes ONE dimension only (`prod`, `all`,
        `any`): a tuple of axes is reduced one by one, the highest first so that the others keep
        their position. A host array (a `ShapeVar` count, say) is read as a tensor."""
        a = torch.as_tensor( a )
        if axis is None:
            return fn( a )
        if isinstance( axis, int ):
            return fn( a, dim = axis )
        for d in sorted( ( d % a.ndim for d in axis ), reverse = True ):
            a = fn( a, dim = d )
        return a

    def sum( self, a, axis = None ):
        a = torch.as_tensor( a )
        return torch.sum( a ) if axis is None else torch.sum( a, dim = axis )

    def prod( self, a, axis = None ):
        return self._reduced( torch.prod, a, axis )

    # a SCAN, not a reduction: the shape is preserved, `axis` designates a single one.
    def cumsum( self, a, axis ):
        return torch.cumsum( a, dim = axis )

    # end to end along an axis.
    def concatenate( self, arrays, axis = 0 ):
        return torch.cat( list( arrays ), dim = axis )

    def max( self, a, axis = None ):
        a = torch.as_tensor( a )
        return torch.max( a ) if axis is None else torch.amax( a, dim = axis )

    def min( self, a, axis = None ):
        a = torch.as_tensor( a )
        return torch.min( a ) if axis is None else torch.amin( a, dim = axis )

    def mean( self, a, axis = None ):
        a = torch.as_tensor( a )
        return torch.mean( a ) if axis is None else torch.mean( a, dim = axis )

    def all( self, a, axis = None ):
        return self._reduced( torch.all, a, axis )

    def any( self, a, axis = None ):
        return self._reduced( torch.any, a, axis )

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
        if len( axes ) == 0:        # a scalar has nothing to permute
            return a
        return a.permute( *axes )

    def hstack( self, lst ):
        return torch.hstack( lst )

    def to_numpy( self, t ):
        return t.detach().cpu().numpy() if isinstance( t, torch.Tensor ) else np.asarray( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def to_nanobind_compatible_objects( self, obj ):
        if isinstance( obj, torch.Tensor ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        return None

    # -- the call --
    def call( self, name, *kernels, nb_items = None, batch_alignment = None, has_dynamic_capacity = True, **args ):
        """Runs an `FfiCode` on the values passed as kwargs -- the same thing as `JaxDriver.call`
        (see `loom/calls.py` for the argument vocabulary). Kernels run on the CPU (`TorchFfi`); when
        a backward kernel is given and an input requires grad, the call is a `torch.autograd.Function`.

        A capacity that turns out too small reruns the call with the requested room, exactly as
        under Jax (nothing traces here, so this always applies).
        """
        from ..calls import lower_args, returned
        from .TorchFfi import call as ffi_call
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
