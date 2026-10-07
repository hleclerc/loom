"""One buffer class per framework (and per way a framework holds a value that matters to a call).

`buffer_class_for( raw )` is the ONLY place a framework is recognised from a value. It reads the
module of the value's TYPE, so a framework is never imported just to be told it is not the one."""
import numpy


def buffer_class_for( raw ):
    """The buffer class that holds `raw`, or `None` when it is not an array of a framework we know."""
    module = type( raw ).__module__.split( "." )[ 0 ]
    if module in ( "jax", "jaxlib" ):
        if type( raw ).__name__.endswith( "Tracer" ):
            from .JaxTracedBuffer import JaxTracedBuffer
            return JaxTracedBuffer
        from .JaxBuffer import JaxBuffer
        return JaxBuffer
    if module == "torch":
        from .TorchBuffer import TorchBuffer
        return TorchBuffer
    if module == "cupy":
        from .CupyBuffer import CupyBuffer
        return CupyBuffer
    if isinstance( raw, numpy.ndarray ):
        from .NumpyBuffer import NumpyBuffer
        return NumpyBuffer
    return None
