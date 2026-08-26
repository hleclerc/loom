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
        "dot":              (".tensor.functions",      "dot"),
        "where":            (".tensor.functions",      "where"),
        "sum":              (".tensor.functions",      "sum"),
        "prod":             (".tensor.functions",      "prod"),
        "min":              (".tensor.functions",      "min"),
        "max":              (".tensor.functions",      "max"),
        "mean":             (".tensor.functions",      "mean"),
        "all":              (".tensor.functions",      "all"),
        "any":              (".tensor.functions",      "any"),
        "sqrt":             (".tensor.functions",      "sqrt"),
        "arcsin":           (".tensor.functions",      "arcsin"),
        "abs":              (".tensor.functions",      "abs"),
        "clip":             (".tensor.functions",      "clip"),
        "stop_gradient":    (".tensor.functions",      "stop_gradient"),
        "transpose":        (".tensor.functions",      "transpose"),
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
