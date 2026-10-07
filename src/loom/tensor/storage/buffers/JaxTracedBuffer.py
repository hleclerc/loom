from .JaxBuffer import JaxBuffer


class JaxTracedBuffer( JaxBuffer ):
    """A jax tracer: a value that only exists inside a `jit` / `vmap` / `grad` trace. It traces, so a
    call over it is a node of the XLA program, and it is never read back to the host."""

    @property
    def traces( self ) -> bool:
        return True

    def device( self ):
        return None

    def to_numpy( self ):
        raise TypeError( "a traced jax value has no host value: it only exists inside the trace" )
