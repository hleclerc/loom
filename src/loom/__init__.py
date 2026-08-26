"""loom — agnostic Jax/Torch → SYCL interface.

Lazy imports: `import loom` is instant. Heavy modules (Tensor, driver, FfiCode)
are loaded on first access, e.g. `from loom import Tensor`.
"""

import sys as _sys


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
        "driver":          (".drivers.driver",        "driver"),
        "new_batch_axis":  (".tensor.batch",          "new_batch_axis"),
    }

    if name in _lazy:
        mod_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(mod_path, package=__package__)
        val = getattr(mod, attr)
        # Cache in module globals so __getattr__ is not called again
        globals()[name] = val
        return val

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
