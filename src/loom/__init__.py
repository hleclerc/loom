"""loom — agnostic Jax/Torch → C++ kernels interface.

NB `ffi_call` lives in `calls.py`, and NOT in a module of the same name. A submodule is bound as an
attribute of its package on import: `loom/calls.py` imported from elsewhere would replace
`loom.ffi_call` (the function, cached by `__getattr__` below) with the MODULE. The error
was `'module' object is not callable`, at an unrelated place.

Lazy imports: `import loom` is instant. Heavy modules (Tensor, driver, FfiCode)
are loaded on first access, e.g. `from loom import Tensor`.
"""

import sys as _sys
import types as _types

from . import env as _env

# ---- WHAT loom BUILDS when nothing was said ----------------------------------------------------------------
# Four global settings, read at each use -- so assigning them is all it takes, nothing watches them:
#
#     loom.default_framework = "torch"      a name or a `Framework`; `None`: the one already imported / available
#     loom.default_device    = "cpu"        a name or a `Device`;    `None`: the framework's own
#     loom.default_dtype.size = 32          the real type: a `Dtype` whose `size` `None` leaves to the device
#     loom.default_itype.size = 64          the integer type, same
#
# (`LOOM_FRAMEWORK`, `LOOM_DEVICE`, `LOOM_FTYPE` and `LOOM_ITYPE` give their initial values.) They decide what a
# value built from nothing is -- `RealTensor( [ 1, 2 ] )`, `zeros`, `arange` -- and nothing else: a value that
# already exists keeps its framework and its sizes, and an operation follows its operands.
# Read them as `loom.default_xxx`: `from loom import default_xxx` would only copy what they are now.
default_framework = None
default_device    = None


class _Loom( _types.ModuleType ):
    """The module, watching its defaults: a name is checked the moment it is assigned (a typo fails THERE, not
    at some later use), the concrete `Framework` / `Device` is made once and kept, and the default driver is told,
    so that nothing is looked at again at every use."""

    def __setattr__( self, name, value ):
        if name == "default_framework" and value is not None:
            from .drivers.Framework import Framework
            value = Framework.factory( value )
        elif name == "default_device" and value is not None:
            from .devices.Device import Device
            value = Device.factory( value )
        elif name in ( "default_dtype", "default_itype" ):
            # the `Dtype` stays the same object (it is resized in place, see `DefaultDtype`)
            from .tensor.Dtype import Dtype
            getattr( self, name ).size = Dtype.factory( value ).size
            return
        super().__setattr__( name, value )
        if name in ( "default_framework", "default_device" ):
            module = _sys.modules.get( __name__ + ".drivers.FrameworkDefaults" )
            if module is not None:
                module.settings_changed()


_sys.modules[ __name__ ].__class__ = _Loom
# (through `setattr`: a plain assignment in the module body would not reach `__setattr__`)
if _env.var( "FRAMEWORK" ):
    setattr( _sys.modules[ __name__ ], "default_framework", _env.var( "FRAMEWORK" ) )
if _env.var( "DEVICE" ):
    setattr( _sys.modules[ __name__ ], "default_device", _env.var( "DEVICE" ) )


def __getattr__(name: str):
    """Lazy attribute lookup for heavy modules."""
    _lazy = {
        "Aggregate":       (".util.Aggregate",       "Aggregate"),
        "Axis":            (".tensor.Axis",           "Axis"),
        "AxisList":        (".tensor.AxisList",       "AxisList"),
        "CtShapeVar":      (".tensor.CtShapeVar",     "CtShapeVar"),
        "FfiCode":         (".compilation.FfiCode",   "FfiCode"),
        "ShapeVar":        (".tensor.ShapeVar",       "ShapeVar"),
        "ShapeArray":      (".tensor.ShapeArray",     "ShapeArray"),
        "Tensor":          (".tensor.Tensor",         "Tensor"),
        "RealTensor":      (".tensor.RealTensor",     "RealTensor"),
        "IntTensor":       (".tensor.IntTensor",      "IntTensor"),
        "BoolTensor":      (".tensor.BoolTensor",     "BoolTensor"),
        "CsrTensor":       (".tensor.CsrTensor",      "CsrTensor"),
        "dot":              (".tensor.functions",      "dot"),
        "where":            (".tensor.functions",      "where"),
        "sum":              (".tensor.functions",      "sum"),
        "prod":             (".tensor.functions",      "prod"),
        "cumsum":           (".tensor.functions",      "cumsum"),
        "min":              (".tensor.functions",      "min"),
        "max":              (".tensor.functions",      "max"),
        "mean":             (".tensor.functions",      "mean"),
        "all":              (".tensor.functions",      "all"),
        "any":              (".tensor.functions",      "any"),
        "sqrt":             (".tensor.functions",      "sqrt"),
        "arcsin":           (".tensor.functions",      "arcsin"),
        "exp":              (".tensor.functions",      "exp"),
        "abs":              (".tensor.functions",      "abs"),
        "clip":             (".tensor.functions",      "clip"),
        "stop_gradient":    (".tensor.functions",      "stop_gradient"),
        "transpose":        (".tensor.functions",      "transpose"),
        # what is built from the defaults, and the transformations of the default framework
        "array":           (".drivers.framework_defaults", "array"),
        "zeros":           (".drivers.framework_defaults", "zeros"),
        "ones":            (".drivers.framework_defaults", "ones"),
        "full":            (".drivers.framework_defaults", "full"),
        "arange":          (".drivers.framework_defaults", "arange"),
        "linspace":        (".drivers.framework_defaults", "linspace"),
        "random":          (".drivers.framework_defaults", "random"),
        "astype":          (".drivers.framework_defaults", "astype"),
        "to_numpy":        (".drivers.framework_defaults", "to_host"),
        "jit":             (".drivers.framework_defaults", "jit"),
        "grad":            (".drivers.framework_defaults", "grad"),
        "vjp":             (".drivers.framework_defaults", "vjp"),
        "vmap":            (".drivers.framework_defaults", "vmap"),
        "checkpoint":      (".drivers.framework_defaults", "checkpoint"),
        "fold":            (".drivers.framework_defaults", "fold"),
        "concrete_eval":   (".drivers.framework_defaults", "concrete_eval"),
        "is_traced":       (".drivers.framework_defaults", "is_traced"),
        "is_symbolic_zero": (".drivers.framework_defaults", "is_symbolic_zero"),
        "symbolic_zero":   (".drivers.framework_defaults", "symbolic_zero"),
        "ops":             (".drivers.framework_defaults", "ops"),
        "resolved_framework": (".drivers.framework_defaults", "framework"),
        "resolved_device": (".drivers.framework_defaults", "device"),
        "resolved_dtype":  (".drivers.framework_defaults", "ftype"),
        "resolved_itype":  (".drivers.framework_defaults", "itype"),
        "new_batch_axis":  (".tensor.batch",          "new_batch_axis"),
        "ffi_call":        (".calls",                 "ffi_call"),
        # the vocabulary of a call's arguments: the role is stated ON the value
        "out":             (".calls",                 "out"),
        "mutable":         (".calls",                 "mutable"),
        "scratch":         (".calls",                 "scratch"),
        "unbound":         (".calls",                 "unbound"),
    }

    if name in ( "default_dtype", "default_itype" ):
        # created on first use (a `Dtype` lives in the heavy part of loom), then a plain module attribute
        from .tensor.Dtype import Dtype, DefaultDtype
        env_name, kind = ( "FTYPE", Dtype.fp() ) if name == "default_dtype" else ( "ITYPE", Dtype.si() )
        size = Dtype.factory( _env.var( env_name ) ).size if _env.var( env_name ) else None
        val = DefaultDtype( kind.kind, size )
        val._armed = True
        globals()[name] = val
        return val

    if name in _lazy:
        mod_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(mod_path, package=__package__)
        val = getattr(mod, attr)
        # Cache in module globals so __getattr__ is not called again
        globals()[name] = val
        return val

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
