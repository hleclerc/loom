from typing import TYPE_CHECKING
import sys
import os

from .TorchFramework import TorchFramework
from .JaxFramework import JaxFramework
from .Framework import Framework

from ..devices.Device import Device
from ..tensor.Dtype import Dtype

if TYPE_CHECKING:
    from .JaxDriver import JaxDriver


class DriverProxy:
    """

    This class is roughly a proxy over JaxDriver, TorchDriver, etc... but it allows for specifying `framework` (i.e. "Jax", "Torch", ...) `dtype` (e.g. "FP32", "double", ...), `itype` and `device` without making the instantiation.

    Furthermore when the user changer


    Type attributes:
        * `normalized_dtype`: a string like "FP32", following the TL (the Template Language) naming sheme. Used to import the right procedures
        * `user_dtype`: the string that was specified by the user
        * `dtype`: type used by the frawework, for instance `torch.float32`

        User can write sdot.driver.dtype = ... with any format ("float32", "FP32", torch.float32, ...)

    Device attributs:
        * `device`: instance used by the frawework

    Framework attributes:
        * `normalized_framework`: a string like "jax", "torch", ...
        * `user_framework`: the string that was specified by the user
        * `framework`: same thing than normalized_framework

    To find the default framework:
        * look what is imported in sys.modules
        * else, try if possible to import a module in self.prefered_frameworks ([ 'jax', 'torch' ] by default)

    Cpu | CudaGpu | AppleGpu

    Env variables that are taken into account
        * SDOT_FRAMEWORK
        * SDOT_DEVICE
        * SDOT_FTYPE  -> float point type
        * SDOT_ITYPE  -> integer type (signed)

        * SDOT_VERBOSE
    """

    def __init__( self ):
        self.prefered_frameworks = [ JaxFramework(), TorchFramework() ]

        self._driver_instance = None
        self._framework = None
        self._device = None
        self._ftype = None
        self._itype = None

        if d := os.getenv( "SDOT_FRAMEWORK" ):
            self.framework = d
        if d := os.getenv( "SDOT_DEVICE" ):
            self.device = d
        if d := os.getenv( "SDOT_FTYPE" ):
            self.ftype = d
        if d := os.getenv( "SDOT_ITYPE" ):
            self.itype = d


    def _checked_driver_instance( self ) -> 'JaxDriver':
        if self._driver_instance is None:
            # if not specified by the user look in already imported modules
            framework = self._framework
            if framework is None:
                for prefered_framework in self.prefered_frameworks:
                    if prefered_framework.module_name in sys.modules:
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
                raise RuntimeError( f"Found no valid for sdot (for now, one can use 'torch' or 'jax')" )

            # use framework
            self._driver_instance = framework.make_instance( self._device, self._ftype, self._itype )

        return self._driver_instance


    # ------------------------------------- framework -------------------------------------
    @property
    def framework( self ) -> Framework:
        return self._checked_driver_instance().framework

    @framework.setter
    def framework( self, value ):
        # cancel the driver instance if framework is not compatible
        framework = Framework.factory( value )
        if self._driver_instance is not None and not self._driver_instance.framework == framework:
            self._driver_instance = None

        # set _device
        self._framework = framework

    # ------------------------------------- device -------------------------------------
    @property
    def device( self ) -> Device:
        return self._checked_driver_instance().device

    @device.setter
    def device( self, value ):
        # cancel the driver instance if device is not compatible
        device = Device.factory( value )
        if self._driver_instance is not None and not self._driver_instance.device == device:
            self._driver_instance = None

        # set _device
        self._device = device

    # ------------------------------------- ftype -------------------------------------
    @property
    def ftype( self ) -> Dtype:
        return self._checked_driver_instance().ftype

    @ftype.setter
    def ftype( self, value ):
        # cancel the driver instance if ftype is not compatible
        ftype = Dtype.factory( value )
        if self._driver_instance is not None and not self._driver_instance.ftype == ftype:
            self._driver_instance = None

        # set _ftype
        self._ftype = ftype

    # ------------------------------------- itype -------------------------------------
    @property
    def itype( self ) -> Dtype:
        return self._checked_driver_instance().itype

    @itype.setter
    def itype( self, value ):
        # cancel the driver instance if itype is not compatible
        itype = Dtype.factory( value )
        if self._driver_instance is not None and not self._driver_instance.itype == itype:
            self._driver_instance = None

        # set _itype
        self._itype = itype


    # ---------------------------------- __getattr__ ----------------------------------
    def __getattr__( self, name ):
        return getattr( self._checked_driver_instance(), name )


    # ---------------------------------- framework ----------------------------------
    # @property
    # def normalized_framework( self ) -> str:
    #     # specified by the user ?
    #     if isinstance( self._user_framework, str ):
    #         return self._normalized_framework_for( self._user_framework )

    #     # else, look in imported modules
    #     for name in self.prefered_frameworks:
    #         if name in sys.modules:
    #             self._user_framework = name
    #             return name

    #     # try to import jax
    #     for name in self.prefered_frameworks:
    #         try:
    #             __import__( name )
    #             self._user_framework = name
    #             return name
    #         except ImportError:
    #             pass

    #     # argh
    #     raise RuntimeError( "Unable to find a driver (tested torch and jax)" )

    # @property
    # def user_framework( self ) -> str | None:
    #     return self._user_framework

    # @property
    # def framework( self ) -> str:
    #     return self.normalized_framework

    # @user_framework.setter
    # def user_framework( self, value: str ):
    #     self._user_framework = value

    #     # need an _instance update ?
    #     if value is None or ( self._driver_instance is not None and self._driver_instance.name != self._normalized_framework_for( value ) ):
    #         self._driver_instance = None

    # @framework.setter
    # def framework( self, value: str ):
    #     self.user_framework = value

    # # ---------------------------------- dtype ----------------------------------
    # @property
    # def normalized_dtype( self ) -> str:
    #     # specified by the user ?
    #     if isinstance( self.user_dtype, str ):
    #         return str( self._normalized_dtype_for( self.user_dtype ) )

    #     # metal => FP32
    #     if self.normalized_framework == "metal":
    #         return "FP32"

    #     return "FP64"

    # @property
    # def user_dtype( self ):
    #     return self._user_dtype

    # @property
    # def dtype( self ):
    #     # we need an instance to get the correct dtype object
    #     if self._driver_instance is None:
    #         self._make_driver_instance()
    #     return self._driver_instance.dtype

    # @user_dtype.setter
    # def user_dtype( self, value ):
    #     self._user_dtype = value

    #     # need an _instance update ?
    #     if value is None or ( self._driver_instance is not None and self._normalized_dtype_for( self._driver_instance.dtype ) != self._normalized_dtype_for( value ) ):
    #         self._driver_instance = None

    # @dtype.setter
    # def dtype( self, value ):
    #     self.user_dtype = value


    # # ---------------------------------- itype ----------------------------------
    # @property
    # def numpy_itype( self ):
    #     return numpy.int64

    # # ---------------------------------- device ----------------------------------
    # @property
    # def normalized_device_type( self ):
    #     from .drivers.Cpu    import Cpu
    #     from .drivers.CudaGpu import CudaGpu
    #     from .drivers.Metal  import Metal
    #     parts      = self.normalized_device.split( ":" )
    #     name       = parts[ 0 ]
    #     device_id  = int( parts[ 1 ] ) if len( parts ) > 1 else 0
    #     if name == "cpu"   : return Cpu()
    #     if name == "cuda"  : return CudaGpu( device_id )
    #     if name == "metal" : return Metal()
    #     raise NotImplementedError( f"Unknown device type: { name }" )

    # @property
    # def normalized_device( self ) -> str:
    #     # specified by the user ?
    #     if isinstance( self._user_device, str ):
    #         return self._normalized_device_for( self._user_device )

    #     # numpy => cpu
    #     if self._user_framework is not None and self._normalized_framework_for( self._user_framework ) == "numpy":
    #         return "cpu"

    #     # get a driver class in order to get a default device
    #     Driver = DriverProxy._driver_class( self.normalized_framework )
    #     return Driver.default_normalized_device_for( self._normalized_dtype_for( self._user_dtype ) if self._user_dtype else None )

    # @property
    # def user_device( self ):
    #     return self._user_device

    # @property
    # def device( self ):
    #     if self._driver_instance is None:
    #         self._make_driver_instance()
    #     return self._driver_instance.device

    # @user_device.setter
    # def user_device( self, value ):
    #     self._user_device = value

    #     # need an _instance update ?
    #     if value is None or ( self._driver_instance is not None and self._normalized_device_for( self._driver_instance.device ) != self._normalized_device_for( value ) ):
    #         self._driver_instance = None

    # @device.setter
    # def device( self, value ):
    #     self.user_device = value

    # # ---------------------------------- helpers ----------------------------------
    # def _normalized_framework_for( self, name ) -> str:
    #     # ensure lowercase str
    #     if not isinstance( name, str ):
    #         return self._normalized_framework_for( str( name ) )
    #     name = name.lower()

    #     if name in [ "torch", "pytorch" ]:
    #         return "torch"

    #     if name in [ "jax" ]:
    #         return "jax"

    #     raise RuntimeError( f"Unknown framework type '{ name }'" )

    # def _normalized_dtype_for( self, name ) -> str:
    #     # ensure lowercase str
    #     if not isinstance( name, str ):
    #         return self._normalized_dtype_for( str( name ) )
    #     name = name.lower()

    #     if ( "float32" in name ) or ( "fp32" in name ):
    #         return "FP32"

    #     if ( "float64" in name ) or ( "fp64" in name ):
    #         return "FP64"

    #     raise RuntimeError( f"TODO: find normalized dtype for { name }" )

    # def _normalized_device_for( self, name ) -> str:
    #     # ensure lowercase str
    #     if not isinstance( name, str ):
    #         return self._normalized_device_for( str( name ) )
    #     name = name.lower()

    #     if name.startswith( "cuda" ) or name.startswith( "gpu" ):
    #         if ":" in name:
    #             raise RuntimeError( "TODO: other gpus" )
    #         return "cuda:0"

    #     if name in [ "mps", "metal" ]:
    #         if ":" in name:
    #             raise RuntimeError( "TODO: other gpus" )
    #         return "metal"

    #     if "cpu" in name:
    #         return "cpu"

    #     raise RuntimeError( f"TODO { name }" )
