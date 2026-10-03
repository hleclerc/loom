"""loom — agnostic Jax/Torch → C++ kernels interface.

NB `ffi_call` lives in `calls.py`, and NOT in a module of the same name. A submodule is bound as an
attribute of its package on import: `loom/calls.py` imported from elsewhere would replace
`loom.ffi_call` (the function, cached by `__getattr__` below) with the MODULE. The error
was `'module' object is not callable`, at an unrelated place.

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
        "abs":              (".tensor.functions",      "abs"),
        "clip":             (".tensor.functions",      "clip"),
        "stop_gradient":    (".tensor.functions",      "stop_gradient"),
        "transpose":        (".tensor.functions",      "transpose"),
        "driver":          (".drivers.driver",        "driver"),
        "new_batch_axis":  (".tensor.batch",          "new_batch_axis"),
        "ffi_call":        (".calls",                 "ffi_call"),
        # the vocabulary of a call's arguments: the role is stated ON the value
        "out":             (".calls",                 "out"),
        "mutable":         (".calls",                 "mutable"),
        "scratch":         (".calls",                 "scratch"),
        "unbound":         (".calls",                 "unbound"),
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
