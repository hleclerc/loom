"""Moving a buffer to another framework -- WITHOUT a copy when the memory can be shared.

Between two frameworks that hold real memory the exchange is DLPack: the memory is shared when the
receiving framework can read it where it is, and copied by that framework when it cannot (see the
alignment note below). Either way the value arrives -- loom never refuses what it can do."""
import numpy


def convert( buffer, target ):
    """`buffer.raw` as an array of the framework named `target`."""
    raw = buffer.raw
    if buffer.framework == "numpy":
        return _from_numpy( raw, target )
    if target == "numpy":
        return buffer.to_numpy()
    if buffer.framework == "torch":
        raw = raw.detach()           # what crosses is a value: a tape cannot follow it into another framework
    if target == "jax":
        _hint_if_copied( buffer )
        import jax.numpy as jnp
        return jnp.from_dlpack( raw )
    if target == "torch":
        import torch
        return torch.from_dlpack( raw )
    if target == "cupy":
        import cupy
        return cupy.from_dlpack( raw )
    raise ValueError( f"cannot convert a { buffer.framework } buffer to { target }" )


def _from_numpy( raw, target ):
    if target == "jax":
        import jax.numpy as jnp
        return jnp.asarray( raw )
    if target == "torch":
        import torch
        return torch.from_numpy( raw )
    if target == "cupy":
        import cupy
        return cupy.asarray( raw )
    raise ValueError( f"cannot convert a numpy buffer to { target }" )


# XLA shares a host buffer only when it is aligned this way; otherwise jax COPIES it. That is fine --
# the value arrives either way -- but it is the user's to know if they want to optimize.
_XLA_ALIGNMENT = 64


def _hint_if_copied( buffer ):
    raw = buffer.raw
    if buffer.framework == "torch" and raw.device.type == "cpu" and raw.data_ptr() % _XLA_ALIGNMENT:
        from ....env import hint
        hint( f"a torch tensor is not { _XLA_ALIGNMENT }-byte aligned, so jax copies it instead of sharing its "
              f"memory (allocate it with torch itself -- `t.clone()` -- to avoid the copy)" )
