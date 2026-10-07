from typing import TYPE_CHECKING
import sys
import os
import contextlib
import contextvars

from .TorchFramework import TorchFramework
from .JaxFramework import JaxFramework
from .NumpyFramework import NumpyFramework
from .CupyFramework import CupyFramework
from .Framework import Framework

from ..devices.Device import Device
from .. import env

# `Dtype` is imported ONLY in the methods that use it. At module level, it pulls
# `loom/tensor/__init__.py`, which pulls `ShapeVar`, which does `from ..drivers.driver import driver`:
# a CYCLE, as soon as `loom.drivers.driver` is the first loom module imported. That is
# exactly what `from loom import driver` at the top of the file does -- and it broke a whole
# test file ( `sdot/tests/test_PowerDiagram.py` ). It only worked if something
# had imported `loom.tensor` beforehand.
if TYPE_CHECKING:
    from ..tensor.Dtype import Dtype
    from .Settings import Settings


# The driver a CALL runs on, when the values it was handed ask for another one than the default
# (see `FrameworkDefaults.using`). A context variable, so it follows the call and nothing else.
_call_driver = contextvars.ContextVar( "loom_call_driver", default = None )


# the proxies to tell when a `loom.default_xxx` changes (there is one, `framework_defaults`)
_proxies: list = []


def settings_changed():
    """A default (`loom.default_framework`, `default_device`, `default_dtype`, `default_itype`) changed: the
    driver built for the old ones is forgotten, and the next use builds the right one."""
    for proxy in _proxies:
        proxy._invalidate()


class FrameworkDefaults:
    """

    The defaults of loom -- `loom.default_framework`, `default_device`, `default_dtype`, `default_itype` --
    resolved into a `Settings` (built once, forgotten when a default changes), and the framework a call runs on.


    Type attributes:
        * `normalized_dtype`: a string like "FP32", following the TL (the Template Language) naming sheme. Used to import the right procedures
        * `user_dtype`: the string that was specified by the user
        * `dtype`: type used by the frawework, for instance `torch.float32`

        User can write sdot.driver.dtype = ... with any format ("float32", "FP32", torch.float32, ...)

    Device attributes:
        * `device`: instance used by the frawework

    Framework attributes:
        * `normalized_framework`: a string like "jax", "torch", ...
        * `user_framework`: the string that was specified by the user
        * `framework`: same thing than normalized_framework

    To find the default framework:
        * look what is imported in sys.modules
        * else, try if possible to import a module in self.prefered_frameworks ([ 'jax', 'torch', 'cupy', 'numpy' ] by default, numpy being the forward-only fallback)

    Cpu | CudaGpu | AppleGpu

    Env variables that are taken into account
        * LOOM_FRAMEWORK
        * LOOM_DEVICE
        * LOOM_FTYPE  -> float point type
        * LOOM_ITYPE  -> integer type (signed)

        * LOOM_VERBOSE
    """

    def __init__( self ):
        self.prefered_frameworks = [ JaxFramework(), TorchFramework(), CupyFramework(), NumpyFramework() ]

        self._driver_instance = None
        _proxies.append( self )

    @property
    def framework_name( self ) -> str:
        """The default framework's name, for a cheap comparison with a buffer's."""
        return self._checked_driver_instance().framework.module_name

    def ops_for( self, name = None ):
        """The driver that carries the OPERATIONS of the framework `name` (`sum`, `matmul`, ...). With
        no name -- a value that belongs to no framework -- or the default's, the default driver: no
        lookup, so an operation among values of the one framework costs one comparison."""
        current = self._checked_driver_instance()
        if name is None or name == current.framework.module_name:
            return current
        return self._instance_for( Framework.factory( name ) )

    def _instance_for( self, framework ):
        instances = self.__dict__.setdefault( "_instances", {} )
        key = str( framework )
        if key not in instances:
            instances[ key ] = framework.make_instance( *self._resolved_settings()[ 1: ] )
        return instances[ key ]

    @contextlib.contextmanager
    def using( self, framework ):
        """Run a block on the driver of `framework` instead of the default one -- what a call does
        when its buffers belong to another framework than the one `framework_defaults` was set to. The
        default stays what builds a value nobody said anything about; it does not decide what to do
        with one that already exists. Same device and sizes policy as the default."""
        framework = Framework.factory( framework )
        current = self._checked_driver_instance()
        if current.framework == framework:
            yield current
            return
        instance = self._instance_for( framework )
        token = _call_driver.set( instance )
        try:
            yield instance
        finally:
            _call_driver.reset( token )

    # ------------------------------------- the settings: `loom.default_xxx` -------------------------------------
    @staticmethod
    def _loom():
        return sys.modules[ "loom" ]

    def _invalidate( self ):
        self._driver_instance = None
        self.__dict__.pop( "_instances", None )

    def _resolved_settings( self ):
        """`( framework or None, device or None, ftype or None, itype or None )` as the drivers take them:
        the globals of `loom`, normalized. A size left open (`None`) is the driver's to pick for its device."""
        from ..tensor.Dtype import Dtype
        loom = self._loom()
        framework, device = loom.default_framework, loom.default_device
        ftype, itype = loom.default_dtype, loom.default_itype
        # a COPY: a driver writes its own spelling into the `Dtype` it is given
        return ( framework, device,
                 None if ftype.size is None else Dtype( ftype.kind, ftype.size ),
                 None if itype.size is None else Dtype( itype.kind, itype.size ) )

    def _checked_driver_instance( self ) -> 'Settings':
        override = _call_driver.get()
        if override is not None:
            return override
        if self._driver_instance is None:
            # if not specified by the user look in already imported modules
            framework, device, ftype, itype = self._resolved_settings()
            if framework is None:
                for prefered_framework in self.prefered_frameworks:
                    # numpy is in `sys.modules` of nearly every process: that says nothing about the choice
                    if prefered_framework.module_name in sys.modules and not prefered_framework.is_fallback:
                        framework = prefered_framework
                        break

            # else, try to import one
            if framework is None:
                for prefered_framework in self.prefered_frameworks:
                    if prefered_framework.can_be_imported:
                        framework = prefered_framework
                        break

            # not found :()
            if framework is None:
                raise RuntimeError( "loom: found no framework to run on (one can use 'jax', 'torch', 'cupy' or 'numpy')" )

            # use framework
            self._driver_instance = framework.make_instance( device, ftype, itype )

        return self._driver_instance


    # ------------------------------------- framework / device / sizes -------------------------------------
    # What the user may set on `framework_defaults` is a spelling of `loom.default_xxx`: the globals are the setting.
    @property
    def framework( self ) -> Framework:
        return self._checked_driver_instance().framework

    @framework.setter
    def framework( self, value ):
        self._loom().default_framework = value

    @property
    def device( self ) -> Device:
        return self._checked_driver_instance().device

    @device.setter
    def device( self, value ):
        self._loom().default_device = value

    @property
    def ftype( self ) -> "Dtype":
        return self._checked_driver_instance().ftype

    @ftype.setter
    def ftype( self, value ):
        from ..tensor.Dtype import Dtype
        self._loom().default_dtype.size = Dtype.factory( value ).size

    @property
    def itype( self ) -> "Dtype":
        return self._checked_driver_instance().itype

    @itype.setter
    def itype( self, value ):
        from ..tensor.Dtype import Dtype
        self._loom().default_itype.size = Dtype.factory( value ).size

    # ---------------------------------- __getattr__ ----------------------------------
    def __getattr__( self, name ):
        return getattr( self._checked_driver_instance(), name )
