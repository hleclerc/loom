"""WHICH framework wins when values of several meet -- in a call, or in an operation.

The default framework only says what to BUILD when nothing was specified; a buffer that already
exists belongs to its own framework. Two kinds of buffer are not alike:

  * a buffer that TRACES (a jax tracer, a torch tensor that wants a gradient) belongs to a
    machinery -- the result must be a node of THAT one, so they must all be of the same framework;
  * any other buffer is memory, and gives a preference, nothing more. numpy is memory every
    framework reads: it asks for nothing.
"""

ORDER = ( "jax", "torch", "cupy", "numpy" )


def promote( tracing, plain, default, for_call = False ):
    """The framework name to move to, or `None` when the default one already wins. `tracing` and
    `plain` are the sets of framework names of the buffers that trace / that do not.

    `for_call`: a KERNEL call, where a buffer of any framework is read in place by its address (see
    `PointerFfi.foreign_operand`) -- except by XLA, which needs jax's own buffers and would copy the
    others. So when nothing traces, jax gives way to a framework that reads by address."""
    if len( tracing ) > 1:
        raise TypeError( f"tensors that trace ({ ', '.join( sorted( tracing ) ) }) cannot be mixed: "
                         f"the result is a node of ONE framework's graph. Convert them to the same one." )
    if tracing:
        chosen = next( iter( tracing ) )
    else:
        plain = set( plain ) - { "numpy" }
        if not plain:
            return None
        if for_call and len( plain ) > 1:
            plain.discard( "jax" )
        chosen = default if default in plain else min( plain, key = ORDER.index )
    return None if chosen == default else chosen
