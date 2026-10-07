"""Run C++ kernels on torch tensors -- the Torch twin of `NumpyFfi` / `JaxFfi`.

The engine is `PointerFfi`: the kernels are the very same generated source, compiled against the
plain-buffer FFI shim (see there), and a torch CPU tensor crosses by ADDRESS -- `data_ptr()`, no numpy
view in between, no copy. The compiled handlers are shared with the numpy driver (same `_load` cache).

Differentiation is Torch's own: a call whose code has a backward goes through a
`torch.autograd.Function` whose `backward` is an ordinary kernel call (`BackwardCall.call_backward`,
the same one Jax uses through its `custom_vjp`).

On a CUDA card the kernel is launched on torch's current stream and takes its scratch from torch's
allocator (see `PointerFfi`): the same call, the same ordering, as under Jax.
"""
from __future__ import annotations

import numpy
import torch

from . import PointerFfi
from .BackwardCall import call_backward
from .PointerFfi import Operand, PointerAdapter


_torch_dtypes: dict = {}


def _torch_dtype( dtype ):
    """The torch dtype of a numpy one, asked of torch itself (no table to keep in step with it)."""
    dtype = numpy.dtype( dtype )
    if dtype not in _torch_dtypes:
        _torch_dtypes[ dtype ] = torch.from_numpy( numpy.empty( 0, dtype ) ).dtype
    return _torch_dtypes[ dtype ]


class TorchAdapter( PointerAdapter ):
    """Torch tensors, on the CPU or on a CUDA card: a tensor crosses by ADDRESS wherever it lives (the
    kernel's memory space is the device's, see `Device.cpp_memory_space`), and the kernel is launched on
    the stream torch itself is working on, so no synchronization is needed on either side."""

    framework = "torch"

    @staticmethod
    def torch_device( device ):
        if device.is_cpu:
            return torch.device( "cpu" )
        if device.is_cuda_gpu:
            return torch.device( "cuda", device.device_id )
        raise NotImplementedError( f"the torch driver has no kernel launch on { device } (CPU and CUDA only)" )

    def is_strided( self, x ) -> bool:
        return isinstance( x, torch.Tensor ) and x.dim() > 0 and not x.is_contiguous()

    def operand( self, x, device, strided = False ) -> Operand:
        wanted = self.torch_device( device )
        if not isinstance( x, torch.Tensor ):
            x = torch.as_tensor( x, device = wanted )
        # a tensor on another device than the kernel's is taken where it is: the view's memory space says so, and
        # the queue transfers (see `CallArg_Tensor._memory_space_of`)
        if x.is_contiguous():
            return Operand( x.data_ptr(), x.shape, x, itemsize = x.element_size() )
        if strided:
            # as it is: the kernel reads these strides
            return Operand( x.data_ptr(), x.shape, x, strides_bytes = [ s * x.element_size() for s in x.stride() ] )
        # the kernel was generated for a dense buffer: the one case that costs a copy
        x = x.contiguous()
        return Operand( x.data_ptr(), x.shape, x, itemsize = x.element_size() )

    def zeros( self, shape, dtype, device ):
        return torch.zeros( tuple( shape ), dtype = _torch_dtype( dtype ), device = self.torch_device( device ) )

    def empty( self, shape, dtype, device ):
        return torch.empty( tuple( shape ), dtype = _torch_dtype( dtype ), device = self.torch_device( device ) )

    def operand_of_output( self, x ) -> Operand:
        return Operand( x.data_ptr(), x.shape, x, itemsize = x.element_size() )

    _side_streams: dict = {}

    def _streams( self, device ):
        """`( torch's current stream, ours or None )`: ours only when torch's is the legacy default one."""
        card = self.torch_device( device )
        current = torch.cuda.current_stream( card )
        if current.cuda_stream != 0:
            return current, None
        if card.index not in self._side_streams:
            self._side_streams[ card.index ] = torch.cuda.Stream( device = card )
        return current, self._side_streams[ card.index ]

    def stream( self, device ):
        if device.is_cpu:
            return None
        current, side = self._streams( device )
        return ( side or current ).cuda_stream

    def order_before( self, device ):
        if not device.is_cpu:
            current, side = self._streams( device )
            if side is not None:
                side.wait_stream( current )

    def order_after( self, device ):
        if not device.is_cpu:
            current, side = self._streams( device )
            if side is not None:
                current.wait_stream( side )


_adapter = TorchAdapter()


def _run( code, ca, device, prefix, values = None ):
    """Run `code` on the buffers of `ca`: `( output CallArgs, result tensors )`, nothing written back.
    `values`: the inputs to read instead of the ones `ca` holds -- what a batched call is given (see `vmap`)."""
    return PointerFfi.run( code, ca, device, prefix, _adapter, values )


def _values_of( ca, device ):
    """The inputs of the call, as tensors of the card the kernel runs on."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    values = [ b.jax_input_array() for b in inputs ]
    return [ x if isinstance( x, torch.Tensor ) else torch.as_tensor( x, device = _adapter.torch_device( device ) ) for x in values ]


def _run_op( code, ca, device, prefix ):
    """`_run`, as an operation `torch.func` can see: `( output CallArgs, result tensors )`, nothing written
    back. What the backward calls (`call_backward` expects exactly this), so that a kernel launched from
    inside a `vmap` is batched, like any other."""
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]
    return outputs, list( _make_op( code, ca, device, prefix ).apply( *_values_of( ca, device ) ) )


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects the
    caller handed us (same contract as `JaxFfi.call`). Differentiable when `code` has a backward, and
    mappable by `torch.func.vmap` into ONE launch of a batched kernel, as under Jax."""
    outputs, results = _run_op( code, ca, device, prefix )
    for buffer, tensor in zip( outputs, results ):
        buffer.jax_write_back( tensor )


def _make_op( code, ca, device, prefix, item = None ):
    """The call as a `torch.autograd.Function`: `loss.backward()` / `torch.autograd.grad` reach the backward
    kernel, and `torch.func.vmap` its batching rule (`Op.vmap`) -- the twin of `JaxFfi._make_op`.

    Written the way `torch.func` wants it: a `forward` without a context, and a `setup_context`.

    `item`: for the call a `vmap` derived, `( code, ca, batched inputs, batched outputs )` of the call it
    was derived FROM. The backward of a batched call is the backward of ONE item, mapped: what `call_backward`
    describes is a call over the values of one item (its objects are the user's, unbatched), and it is the
    framework that maps it -- as Jax does with the `custom_vjp` of the call."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    class Op( torch.autograd.Function ):
        @staticmethod
        def forward( *values ):
            _, outs = _run( code, ca, device, prefix, values )
            return tuple( outs )

        @staticmethod
        def setup_context( ctx, values, outs ):
            if not code.has_backward:
                # nothing differentiates through a kernel that has no backward: its outputs are constants
                ctx.mark_non_differentiable( *outs )
                return
            ctx.perturbed = [ bool( needs ) and t.is_differentiable for needs, t in zip( ctx.needs_input_grad, inputs ) ]
            ctx.save_for_backward( *values, *outs )
            ctx.mark_non_differentiable( *[ o for o in outs if not o.is_floating_point() ] )

        @staticmethod
        def backward( ctx, *cotangents ):
            saved = ctx.saved_tensors
            full_in, out_values = saved[ : len( inputs ) ], saved[ len( inputs ) : ]
            # a cotangent the autograd did not materialize is a zero of the output's shape
            cotangents = [ torch.zeros_like( o ) if c is None else c for c, o in zip( cotangents, out_values ) ]
            if item is None:
                grads = call_backward( code, ca, device, prefix, inputs, outputs,
                                       full_in, out_values, ctx.perturbed, cotangents, _run_op )
            else:
                grads = _mapped_backward( item, device, prefix, full_in, out_values, ctx.perturbed, cotangents )
            return tuple( grads )

        @staticmethod
        def vmap( info, in_dims, *values ):
            """What `torch.func.vmap` calls instead of looping: RECOMPILE. The code gains one batch axis and
            the lowering the buffers that gained a leading dimension, and the kernel that comes out runs the
            N items in one launch (`JaxFfi._make_op` says the same, with Jax's names)."""
            values = [ v if d is None else v.movedim( d, 0 ) for v, d in zip( values, in_dims ) ]
            in_batched = [ d is not None for d in in_dims ]
            out_batched = [ t.takes_batch_axis() for t in outputs ]

            axis_name, batched_code = code.with_batch_axis()
            batched_ca = ca.batched( axis_name, info.batch_size, { t.ffi_name for t, b in zip( inputs, in_batched ) if b } )

            results = _make_op( batched_code, batched_ca, device, prefix,
                                ( code, ca, in_batched, out_batched ) if item is None else item ).apply( *values )

            # ... one output per item, save what belongs to the CALL rather than to an item (the error buffer)
            return tuple( results ), tuple( 0 if b else None for b in out_batched )

    return Op


def _mapped_backward( item, device, prefix, full_in, out_values, perturbed, cotangents ):
    """The cotangents of a BATCHED call: the backward of one item, mapped over the batch (see `_make_op`).

    An input the mapping did not give a batch axis is shared by every item: its cotangent is the SUM of what each
    item contributes -- which the mapping returns item by item."""
    code, ca, in_batched, out_batched = item
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]
    wanted = [ k for k, p in enumerate( perturbed ) if p ]

    def one_item( full_in, out_values, cotangents ):
        grads = call_backward( code, ca, device, prefix, inputs, outputs,
                               full_in, out_values, perturbed, cotangents, _run_op )
        return [ grads[ k ] for k in wanted ]       # tensors only: a mapped function does not return `None`

    dim = lambda batched: 0 if batched else None
    mapped = torch.func.vmap( one_item, in_dims = ( [ dim( b ) for b in in_batched ],
                                                    [ dim( b ) for b in out_batched ],
                                                    [ dim( b ) for b in out_batched ] ) )
    grads = iter( mapped( list( full_in ), list( out_values ), list( cotangents ) ) )

    res = [ None ] * len( inputs )
    for k in wanted:
        g = next( grads )
        res[ k ] = g if in_batched[ k ] else g.sum( 0 )
    return res
