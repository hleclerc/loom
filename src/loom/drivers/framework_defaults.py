"""WHAT loom BUILDS from nothing, assembled -- and WHICH framework carries an operation.

Two concepts, kept apart:

  * a FRAMEWORK is what runs the operations (`framework.operations`): it has no size and no device of
    its own to speak of;
  * the DEFAULTS (`loom.default_framework`, `default_device`, `default_dtype`, `default_itype`) are what a value
    nobody said anything about is built with.

This module is where a caller assembles the two: `framework_defaults.zeros( shape, dtype )` is the default framework's
`zeros`, given the dtype completed from the default sizes and the default device. An operation on a value that
already exists does not come here for a framework: it asks the value (`ops( x )`).
"""
from .FrameworkDefaults import FrameworkDefaults
from .Framework import Framework
_driver = FrameworkDefaults()

from ..tensor.host import to_host      # noqa: F401  (the host reading, re-exported with the rest)


# ---- the resolved defaults ------------------------------------------------------------------------------
def framework() -> Framework:
    """The default `Framework`: the one that was set, else the one already imported / available."""
    return _driver.framework


def framework_name() -> str:
    """The default framework's name, for a cheap comparison with a buffer's."""
    return _driver.framework_name


def using( framework ):
    """Run a block with `framework` as the one the defaults resolve for -- what a call does when its buffers belong to
    another framework than the default: whatever the call builds or reads on the way is of THAT framework."""
    return _driver.using( framework )


def call( name, *kernels, **kwargs ):
    """`ffi_call`, on the framework and device in force."""
    return _driver.call( name, *kernels, **kwargs )


def device():
    """The default `Device`: the one that was set, else the framework's own."""
    return _driver.device


def ftype():
    """The default real type, its size resolved."""
    return _driver.ftype


def itype():
    """The default integer type, its size resolved."""
    return _driver.itype


def concrete( dtype = None, integer = False ):
    """`dtype` (anything `Dtype.factory` reads, or `None` for the default real -- or integer) with its size
    filled in from the default sizes."""
    return _driver.concrete( dtype, integer )


def dtype_version( dtype ):
    """The default framework's spelling of `dtype` (sizes left open completed by the defaults)."""
    return _driver.dtype_version( dtype )


# ---- which framework carries an operation on a value ----------------------------------------------------
def framework_name_of( x ):
    """The name of the framework that holds `x`, or `None` when it is no array of one (a list, a number)."""
    from ..tensor.storage.buffers import buffer_class_for
    cls = buffer_class_for( x )
    if cls is not None:
        return cls.framework
    # the numpy driver's own symbolic zero
    if type( x ).__module__.startswith( "loom." ) and type( x ).__name__ == "SymbolicZero":
        return "numpy"
    return None


def ops( of = None ):
    """The OPERATIONS to apply to `of` (`framework.operations`): those of ITS framework -- or of the default
    one, for what belongs to none (nothing, a list, a number) and for numpy, which every framework reads.
    They take a concrete dtype and an explicit device: see `concrete` / `device`."""
    name = None if of is None else framework_name_of( of )
    if name is None or name == "numpy":
        return framework().operations
    return Framework.factory( name ).operations


def ops_named( name ):
    """The operations of the framework called `name`, or the default's for `None` / numpy."""
    if name is None or name == "numpy":
        return framework().operations
    return Framework.factory( name ).operations


# ---- building a value from the defaults -----------------------------------------------------------------
def array( data, dtype = None, device = None ):
    return _driver.array( data, dtype, device )


def zeros( shape, dtype = None ):
    return _driver.zeros( shape, dtype )


def ones( shape, dtype = None ):
    return _driver.ones( shape, dtype )


def full( shape, value, dtype = None ):
    return _driver.full( shape, value, dtype )


def arange( nb, dtype = None ):
    return _driver.arange( nb, dtype )


def linspace( a, b, nb, dtype = None ):
    return _driver.linspace( a, b, nb, dtype )


def random( shape, dtype = None, seed = None ):
    return _driver.random( shape, dtype, seed )


def symbolic_zero( shape, dtype = None ):
    return _driver.symbolic_zero( shape, dtype )


def astype( x, dtype ):
    return _driver.astype( x, dtype )


# ---- the transformations and queries of the default framework -------------------------------------------
# Plain functions (not the framework's bound methods): they look the default up at each call, so that
# `loom.jit` keeps following `loom.default_framework`.
def jit( *args, **kwargs ):
    return _driver.jit( *args, **kwargs )


def grad( *args, **kwargs ):
    return _driver.grad( *args, **kwargs )


def vjp( *args, **kwargs ):
    return _driver.vjp( *args, **kwargs )


def vmap( *args, **kwargs ):
    return _driver.vmap( *args, **kwargs )


def checkpoint( *args, **kwargs ):
    return _driver.checkpoint( *args, **kwargs )


def fold( *args, **kwargs ):
    return _driver.fold( *args, **kwargs )


def concrete_eval( *args, **kwargs ):
    return _driver.concrete_eval( *args, **kwargs )


def is_traced( x ):
    return ops( x ).is_traced( x )


def is_symbolic_zero( x ):
    return ops( x ).is_symbolic_zero( x )
