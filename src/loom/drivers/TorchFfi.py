"""Run C++ kernels on torch tensors -- the Torch twin of `NumpyFfi` / `JaxFfi`.

The kernels are the very same generated source, compiled against the plain-buffer FFI shim of
`NumpyFfi` (see there): a torch CPU tensor IS a host buffer, so it crosses by pointer, with no copy
(`Tensor.numpy()` / `torch.from_numpy` share the memory). The compiled handlers are shared with the
numpy driver (same `_load` cache).

Differentiation is Torch's own: a call whose code has a backward goes through a
`torch.autograd.Function` whose `backward` is an ordinary kernel call (`BackwardCall.call_backward`,
the same one Jax uses through its `custom_vjp`).

CPU only for now: a kernel dereferences its buffers where it runs, and the GPU queues are not wired
to this FFI.
"""
from __future__ import annotations

import ctypes

import numpy
import torch

from .BackwardCall import call_backward
from .FfiSource import call_signature, _render_call
from .NumpyFfi import _load, _as_buffer, _Buffer, _CallFrame


def _host( x ) -> numpy.ndarray:
    """A contiguous numpy VIEW of a torch CPU tensor (a copy only if it is not contiguous)."""
    if isinstance( x, torch.Tensor ):
        if x.device.type != "cpu":
            raise NotImplementedError( f"the torch driver runs kernels on CPU buffers only (got a tensor on { x.device })" )
        return numpy.ascontiguousarray( x.detach().numpy() )
    return numpy.ascontiguousarray( x )


def _run( code, ca, device, prefix ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result tensors )`, nothing written back."""
    if not device.is_cpu:
        raise NotImplementedError( f"the torch driver has no kernel launch on { device } yet (CPU only)" )

    source, inputs, outputs, attrs, sources, headers = _render_call( code, ca, device )
    entry = _load( source, device, prefix, sources, code.name, call_signature( ca ), headers )

    # kept alive by these lists until the kernel is done: the frame only holds addresses
    in_arrays = [ _host( b.jax_input_array() ) for b in inputs ]
    out_arrays = [ numpy.zeros( shape, dtype ) for shape, dtype in ( b.out_shape_dtype() for b in outputs ) ]

    args = ( _Buffer * max( 1, len( in_arrays ) ) )( *[ _as_buffer( a ) for a in in_arrays ] )
    rets = ( _Buffer * max( 1, len( out_arrays ) ) )( *[ _as_buffer( a ) for a in out_arrays ] )
    values = ( ctypes.c_int64 * max( 1, len( attrs ) ) )( *[ int( v ) for _, _, v in attrs ] )
    frame = _CallFrame( args, rets, values )

    error = entry( ctypes.byref( frame ) )
    if error:
        raise RuntimeError( ctypes.string_at( error ).decode( errors = "replace" ) )
    return outputs, [ torch.from_numpy( a ) for a in out_arrays ]


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`). Differentiable when `code` has a backward."""
    if code.has_backward:
        outputs, results = _call_with_autograd( code, ca, device, prefix )
    else:
        outputs, results = _run( code, ca, device, prefix )

    for buffer, tensor in zip( outputs, results ):
        buffer.jax_write_back( tensor )


def _call_with_autograd( code, ca, device, prefix ):
    """The forward call as a `torch.autograd.Function`: `loss.backward()` / `torch.autograd.grad`
    reach the backward kernel. Returns the same `( outputs, results )` as `_run`."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    in_tensors = [ b.jax_input_array() for b in inputs ]
    in_tensors = [ x if isinstance( x, torch.Tensor ) else torch.as_tensor( x ) for x in in_tensors ]

    # nothing wants a gradient: the plain forward is enough (and cheaper)
    if not any( x.requires_grad for x in in_tensors ) or not torch.is_grad_enabled():
        return _run( code, ca, device, prefix )

    class Op( torch.autograd.Function ):
        @staticmethod
        def forward( ctx, *values ):
            _, outs = _run( code, ca, device, prefix )
            ctx.perturbed = [ bool( needs ) and t.is_differentiable for needs, t in zip( ctx.needs_input_grad, inputs ) ]
            ctx.save_for_backward( *values, *outs )
            ctx.mark_non_differentiable( *[ o for o in outs if not o.is_floating_point() ] )
            return tuple( outs )

        @staticmethod
        def backward( ctx, *cotangents ):
            saved = ctx.saved_tensors
            full_in, out_values = saved[ : len( inputs ) ], saved[ len( inputs ) : ]
            # a cotangent the autograd did not materialize is a zero of the output's shape
            cotangents = [ torch.zeros_like( o ) if c is None else c for c, o in zip( cotangents, out_values ) ]
            grads = call_backward( code, ca, device, prefix, inputs, outputs,
                                   full_in, out_values, ctx.perturbed, cotangents, _run )
            return tuple( grads )

    results = Op.apply( *in_tensors )
    return outputs, list( results )
