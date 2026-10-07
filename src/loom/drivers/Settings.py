"""The defaults of loom, RESOLVED for one framework: what a value nobody said anything about is built with.

`framework` runs the operations; `device`, `ftype` and `itype` are what the caller left unsaid, completed -- by
the user's `loom.default_xxx` where they set them, by the framework's own preference where they did not. The
two concepts stay apart (`Framework` has no size): this is where they are assembled, for the callers that
build a value or launch a call.
"""
from ..tensor.Dtype import Dtype, REAL, BOOL


class Settings:
    def __init__( self, framework, device, ftype, itype ):
        ops = framework.operations
        device, ftype, itype = ops.defaults_for( device, ftype, itype )
        assert itype.floating_point == False
        assert ftype.floating_point == True

        self.framework = framework
        self.device = device
        self.ftype = ftype
        self.itype = itype

        # the framework's spelling of both sizes, filled in the `Dtype`s that carry them
        itype._driver_version = ops.dtype_version( itype )
        ftype._driver_version = ops.dtype_version( ftype )
        ops.configure( device, ftype, itype )

    # What is not a default lives with the framework (`XxxOperations`): `settings.sum( x )` is
    # `settings.framework.operations.sum( x )`.
    def __getattr__( self, name ):
        if name.startswith( "__" ) or "framework" not in self.__dict__:
            raise AttributeError( name )
        return getattr( self.framework.operations, name )

    def call( self, name, *kernels, **kwargs ):
        """`ffi_call`, on this framework and this device."""
        return self.framework.operations.call( self.device, name, *kernels, **kwargs )

    # ---- completing what a caller left open --------------------------------------------------------------------
    def concrete( self, dtype = None, integer = False ):
        """`dtype` (a `Dtype`, anything `Dtype.factory` reads, or `None`) with its size filled in from the
        defaults: a real takes `ftype`'s, an integer `itype`'s. `None` is the default real -- or integer,
        when `integer` is set."""
        d = ( self.itype if integer else self.ftype ) if dtype is None else Dtype.factory( dtype )
        if d.size is None and d.kind != BOOL:
            d = Dtype( d.kind, ( self.ftype if d.kind == REAL else self.itype ).size )
        return d

    def dtype_version( self, dtype ):
        """The dtype OF THIS FRAMEWORK that `dtype` denotes, sizes left open resolved by the defaults."""
        return self.framework.operations.dtype_version( self.concrete( dtype ) )

    def driver_dtype_version( self, kind, size ):
        return self.dtype_version( Dtype( kind, size ) )

    @property
    def _ops( self ):
        return self.framework.operations

    def astype( self, x, dtype ):
        return self._ops.astype( x, self.concrete( dtype ) )

    def array( self, data, dtype = None, device = None ):
        return self._ops.array( data, self.concrete( dtype ), device or self.device )

    def zeros( self, shape, dtype = None ):
        return self._ops.zeros( shape, self.concrete( dtype ), self.device )

    def full( self, shape, value, dtype = None ):
        return self._ops.full( shape, value, self.concrete( dtype ), self.device )

    def ones( self, shape, dtype = None ):
        return self._ops.ones( shape, self.concrete( dtype ), self.device )

    def arange( self, nb, dtype = None ):
        return self._ops.arange( nb, self.concrete( dtype, integer = True ), self.device )

    def linspace( self, a, b, nb, dtype = None ):
        return self._ops.linspace( a, b, nb, self.concrete( dtype ), self.device )

    def random( self, shape, dtype = None, seed = None ):
        return self._ops.random( shape, self.concrete( dtype ), seed, self.device )

    def symbolic_zero( self, shape, dtype = None ):
        return self._ops.symbolic_zero( shape, self.concrete( dtype ) )

    def empty( self, shape, dtype = None ):
        return self._ops.empty( shape, self.concrete( dtype ), self.device )
