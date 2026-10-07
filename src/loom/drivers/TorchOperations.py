"""The torch framework's operations (see `TorchFramework`)."""
from .CallArgsAnalysis import CallArgsAnalysis
from ..compilation.FfiCode import FfiCode, Kernels
from ..devices.Device import Device
from .. import env
import numpy as np
import torch
from ..tensor.Dtype import Dtype


class TorchOperations:
    """What the torch framework DOES with an array that already exists: its operations, its
    transforms, its readings, its way of building one from a concrete dtype and an explicit device, and of
    launching a kernel. It has no default of its own: the defaults are loom's (`Settings`), and are handed in."""

    _INT_DTYPES = frozenset( [ torch.int8, torch.int16, torch.int32, torch.int64,
                                torch.uint8, torch.uint16, torch.uint32, torch.uint64 ] )

    def is_strided( self, x ):
        """Is `x` held otherwise than dense row-major: a view the kernel reads with its own strides."""
        from .TorchFfi import _adapter
        return _adapter.is_strided( x )

    def dtype_of( self, x ):
        """The `Dtype` a torch buffer ACTUALLY has -- what a declaration is checked against.
        Read from the dtype alone, so no host sync and no numpy round-trip."""
        from ..tensor.Dtype import Dtype
        dt = x.dtype
        return Dtype.from_numpy( dt )       # a torch dtype or a numpy one: `Dtype` reads both

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

    def grad( self, func, argnums = 0 ):
        """Gradient of a scalar-valued `func` wrt the argument(s) `argnums`. A `loom.ffi_call` inside
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
        """`( func( *primals ), pullback )` -- see `JaxOperations.vjp`. `pullback( cotangent )` gives one
        gradient per primal (a symbolic-zero cotangent contributes nothing)."""
        import torch.utils._pytree as pytree
        with torch.enable_grad():
            tracked, leaves = self._tracked( primals )
            out = func( *tracked )
        out_leaves, out_spec = pytree.tree_flatten( out )

        def pullback( cotangent ):
            cts = [ torch.as_tensor( c ) if isinstance( c, np.ndarray ) else c for c in pytree.tree_flatten( cotangent )[ 0 ] ]
            pairs = [ ( o, c ) for o, c in zip( out_leaves, cts ) if o.requires_grad and not self.is_symbolic_zero( c ) ]
            # a cotangent given from the host (numpy, a list) is given to the card the output is on
            pairs = [ ( o, c.to( o.device ) if isinstance( c, torch.Tensor ) else c ) for o, c in pairs ]
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
        """`func`, compiled by `torch.compile`: the tensor code around the calls is compiled, and each
        `loom.ffi_call` is a break in the graph -- a kernel is opaque to the compiler, and runs eagerly, on the
        values it is given (nothing is traced INTO tracers, so no count or capacity ever goes missing, which
        is why `need( "trace" )` still says no).

        Falls back to `func` itself, once and with a warning, if the machine cannot compile (Inductor needs a
        C++ compiler): the values do not depend on it."""
        import warnings
        self._inductor_needs_a_system_compiler()
        compiled = torch.compile( func )
        broken = []

        def jitted( *args, **kwargs ):
            if broken:
                return func( *args, **kwargs )
            try:
                return compiled( *args, **kwargs )
            except torch._dynamo.exc.BackendCompilerFailed as e:
                broken.append( e )
                warnings.warn( f"loom: torch.compile cannot compile here, `jit` runs `{ getattr( func, '__name__', func ) }` as it is ({ str( e )[ :200 ] })" )
                return func( *args, **kwargs )
        return jitted

    @staticmethod
    def _inductor_needs_a_system_compiler():
        """Inductor compiles its C++ with the first `clang++` on the PATH. On macOS inside a conda
        environment that has the `clangxx` package (what loom's own kernels want), that is conda's, and
        it cannot build Inductor's precompiled headers (`cannot specify -o when generating multiple
        output files`). Apple's compiler does: it is the one Inductor gets, unless the user chose with
        `CXX` -- which Inductor reads itself -- or already set `torch._inductor.config.cpp.cxx`."""
        import os, shutil, sys
        import torch._inductor.config as config
        apple = "/usr/bin/clang++"
        if sys.platform != "darwin" or os.environ.get( "CXX" ) or not os.path.exists( apple ):
            return
        if config.cpp.cxx != ( None, "clang++" ):          # not the default: somebody chose
            return
        found = shutil.which( "clang++" )
        if found and os.path.realpath( found ).startswith( os.path.realpath( sys.prefix ) ):
            config.cpp.cxx = ( None, apple )

    def stop_gradient( self, x ):
        return x.detach() if isinstance( x, torch.Tensor ) else x

    @property
    def array_type( self ):
        return torch.Tensor

    @property
    def int_type( self ):
        return torch.int64

    def is_int_dtype( self, dtype ):
        return dtype in TorchOperations._INT_DTYPES

    def any_requires_grad( self, tensors ) -> bool:
        return any( t.requires_grad for t in tensors )

    def reshape( self, tensor, shape ):
        return tensor.reshape( tuple( shape ) )

    def expand_dims( self, tensor, index ):
        return tensor.unsqueeze( index )

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

    def stack( self, tensors, axis ):
        return torch.stack( tensors, dim=axis )

    def is_symbolic_zero( self, x ):
        return isinstance( x, torch.Tensor ) and x.is_meta

    # see `JaxOperations.is_traced`: Torch's autograd does not retrace a Python body the way
    # `lax.scan`'s differentiation rule does, so a `ComputedAttribute` cache never goes stale
    # under it. Only a `vmap` hands out stand-ins.
    def is_traced( self, x ):
        """A value inside a `torch.func.vmap` is a stand-in for what the kernel will see: only the
        batching rule reads its content (see `JaxOperations.is_traced`)."""
        return isinstance( x, torch.Tensor ) and torch._C._functorch.is_functorch_wrapped_tensor( x )

    # see `JaxOperations.concrete_eval`: nothing is traced here, so there is nothing to escape
    def concrete_eval( self ):
        import contextlib
        return contextlib.nullcontext()

    # see `JaxOperations.checkpoint`: `func` re-evaluated in the backward instead of taping its
    # intermediates. Torch spells it as a CALL wrapper rather than a decorator, so we adapt it to
    # the same "function -> function" verb. `use_reentrant = False` is the non-deprecated
    # implementation (the only one that supports closed-over tensors and nested autograd).
    def checkpoint( self, func ):
        import torch.utils.checkpoint as torch_checkpoint
        return lambda *args: torch_checkpoint.checkpoint( func, *args, use_reentrant = False )

    # see `JaxOperations.fold`. Torch's autograd is eager: a Python loop over the leading axis IS the
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

    def exp( self, a ):
        return torch.exp( a )

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

    def matmul( self, a, b ):
        """`a @ b`. On a card, cuBLAS has no INTEGER product: that one is contracted by hand."""
        a, b = torch.as_tensor( a ), torch.as_tensor( b )
        if not a.is_cuda or a.is_floating_point() or a.is_complex():
            return a @ b
        a2 = a.unsqueeze( 0 ) if a.ndim == 1 else a
        b2 = b.unsqueeze( -1 ) if b.ndim == 1 else b
        res = ( a2.unsqueeze( -1 ) * b2.unsqueeze( -3 ) ).sum( -2 )         # [ ..., m, k, n ] summed over k
        if a.ndim == 1:
            res = res.squeeze( -2 )
        if b.ndim == 1:
            res = res.squeeze( -1 )
        return res

    def to_numpy( self, t ):
        return t.detach().cpu().numpy() if isinstance( t, torch.Tensor ) else np.asarray( t )

    def to_nanobind_compatible_objects( self, obj ):
        if isinstance( obj, torch.Tensor ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        return None

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

    @property
    def available_gpus( self ):
        return torch.cuda.device_count()

    # ---- what a framework calls a dtype -----------------------------------------------------------------
    def dtype_version( self, dtype ):
        """The torch dtype a CONCRETE `Dtype` denotes: its size is the caller's to have filled in (the
        defaults are loom's, not a framework's)."""
        from ..tensor.Dtype import Dtype, REAL, SINT, UINT, BOOL
        dtype = Dtype.factory( dtype )
        if dtype.kind == BOOL:
            return torch.bool
        if dtype.size is None:
            raise ValueError( f"the size of { dtype.cpp_name } is not resolved: it is loom's default to fill in, not a framework's" )
        table = { REAL: { 16: torch.float16, 32: torch.float32, 64: torch.float64 },
                  SINT: { 8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64 },
                  UINT: { 8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.uint64 } }[ dtype.kind ]
        try:
            return table[ dtype.size ]
        except KeyError:
            raise ValueError( f"unsupported size for { dtype.cpp_name }" )

    def astype( self, x, dtype ):
        """`x` re-typed as `dtype` (a concrete `Dtype`); a no-op when it already is."""
        return x.to( self.dtype_version( dtype ) )

    # ---- building a value, with a concrete dtype and an explicit device (see `DefaultSizes`) -------------------
    @staticmethod
    def torch_device( device ):
        """The torch spelling of a loom `Device` (`None`: torch's own default)."""
        if device is None:
            return None
        if device.is_cpu:
            return torch.device( "cpu" )
        if device.is_apple_gpu:
            return torch.device( "mps" )
        return torch.device( "cuda", getattr( device, "device_id", 0 ) )

    def array( self, data, dtype, device = None ):
        if data is None:
            return None
        where = self.torch_device( device )
        if isinstance( data, torch.Tensor ):
            return data.to( dtype = self.dtype_version( dtype ), device = where )
        # through numpy: it reads what torch cannot (a `ShapeArray`, nested lists of numpy scalars),
        # and torch refuses a negative stride
        return torch.as_tensor( np.asarray( data, dtype = Dtype.factory( dtype ).numpy_dtype, order = "C" ), device = where )

    def zeros( self, shape, dtype, device = None ):
        return torch.zeros( tuple( shape ), dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    def full( self, shape, value, dtype, device = None ):
        return torch.full( tuple( shape ), value, dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    def ones( self, shape, dtype, device = None ):
        return torch.ones( tuple( shape ), dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    def arange( self, nb, dtype, device = None ):
        return torch.arange( nb, dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    def linspace( self, a, b, n, dtype, device = None ):
        # numpy's formula, not torch's (they differ in the last bit, and a grid should be the same
        # grid whatever the framework)
        return torch.as_tensor( np.linspace( a, b, int( n ) ), dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    def random( self, shape, dtype, seed = None, device = None ):
        """A uniform draw, same contract as `JaxOperations.random`: `seed = None` advances a
        process counter, an explicit seed makes the draw reproducible."""
        if seed is None:
            seed = getattr( self, "_rng_seed", 0 )
            self._rng_seed = seed + 1
        where = self.torch_device( device )
        generator = torch.Generator( device = where ).manual_seed( int( seed ) )
        return torch.rand( tuple( shape ), generator = generator, dtype = self.dtype_version( dtype ), device = where )

    def empty( self, shape, dtype, device = None ):
        return torch.zeros( tuple( shape ), dtype = self.dtype_version( dtype ), device = self.torch_device( device ) )

    # A symbolic zero: a SHAPED, TYPED, BUFFERLESS value read as 0 (see `JaxOperations.symbolic_zero`).
    # Torch has no native one, but a `meta` tensor is exactly that -- shape/dtype, no storage (any
    # materialization raises), recognizable by `is_meta`.
    def symbolic_zero( self, shape, dtype ):
        return torch.zeros( tuple( shape ), dtype = self.dtype_version( dtype ), device = "meta" )

    # -- the call --
    # Opaque to `torch.compile`: a call is Python bookkeeping around a C++ kernel (lowering, rendering,
    # ctypes, a compilation cache, a write-back onto the caller's objects), none of which a compiler can
    # trace -- nor should it. It is a break in the graph, and the tensors around it are compiled.
    @torch.compiler.disable
    def call( self, device, name, *kernels, nb_items = None, batch_alignment = None, has_dynamic_capacity = True, failures = None, **args ):
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
            raise ValueError( f"loom.ffi_call: expected one kernel (the forward) or two (forward, "
                              f"backward), got { len( kernels ) }" )
        code = Kernels( name, *kernels )

        prefix = name + "_"

        output_capacities = dict( output_capacities )   # ours to grow: the caller's dict is not ours to touch
        while True:
            ca = CallArgsAnalysis( kwargs, device, output_attributes, output_capacities, output_exceptions, input_exceptions, batch_alignment, scratch_attributes, groups, name, nb_items )
            ca.errors.call_name, ca.errors.failure_messages = name, dict( failures or {} )
            ffi_call( code, ca, device, prefix )

            overflows = ca.capacity_overflows()
            if not overflows:
                return returned( returns )

            if env.flag( "DEBUG_CAPACITY" ):
                print( f"[capacity] { code.name }: " + ", ".join(
                    f"{ path } wanted={ wanted } capacity={ capacity }" for path, wanted, capacity in overflows ),
                    flush = True )

            for path, wanted, capacity in overflows:
                output_capacities[ path ] = max( wanted, 2 * capacity )

    def vmap( self, func ):
        """Map `func` over a new leading axis, as `torch.func.vmap` does: a `loom.ffi_call` inside is
        BATCHED (one launch of a kernel with one more axis, see `TorchFfi._make_op`) -- as under Jax.
        Only the tensors are mapped; any other argument goes through as it is."""
        import torch.utils._pytree as pytree
        def mapped( *args ):
            args = [ torch.as_tensor( a ) if isinstance( a, np.ndarray ) else a for a in args ]
            in_dims = tuple( pytree.tree_map( lambda x: 0 if isinstance( x, torch.Tensor ) else None, a ) for a in args )
            return torch.func.vmap( func, in_dims = in_dims )( *args )
        return mapped

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
        pass
