import math

from sdot.tensor.PhysicalLayout import PhysicalLayout, items_per_alignment
from sdot.testing import test


def _contiguous( shape ):
    s, acc = [ 0 ] * len( shape ), 1
    for i in range( len( shape ) - 1, -1, -1 ):
        s[ i ] = acc
        acc *= shape[ i ]
    return s


if test( "items_per_alignment_depends_on_itemsize" ):
    # a BYTE alignment maps to a different ITEM granularity per dtype
    assert items_per_alignment( 128, 4 ) == 32          # fp32
    assert items_per_alignment( 128, 8 ) == 16          # fp64
    assert items_per_alignment( 64, 8 ) == 8
    assert items_per_alignment( 1, 8 ) == 1             # no meaningful alignment -> no padding


if test( "no_batch_is_identity" ):
    # no batch axis -> exactly the plain contiguous layout
    L = PhysicalLayout.of( caps = [ 3, 5 ], is_batch = [ False, False ], alignment_bytes = 128, itemsize = 8 )
    assert L.is_identity
    assert L.buffer_shape == [ 3, 5 ]
    assert L.strides == _contiguous( [ 3, 5 ] )   # [ 5, 1 ]


if test( "batch_alignment_1_is_identity" ):
    # a leading batch axis, no meaningful byte alignment: byte-identical to the plain layout
    L = PhysicalLayout.of( caps = [ 3, 5 ], is_batch = [ True, False ], alignment_bytes = 1, itemsize = 8 )
    assert L.is_identity
    assert L.buffer_shape == [ 3, 5 ]
    assert L.strides == [ 5, 1 ]


if test( "batch_padded" ):
    # one batch axis (3), 32-byte alignment in fp64 -> 4 items -> capacity 4; strides UNCHANGED
    # (padding is extra rows at the end; the used region is the first 3).
    L = PhysicalLayout.of( caps = [ 3, 5 ], is_batch = [ True, False ], alignment_bytes = 32, itemsize = 8 )
    assert not L.is_identity
    assert L.buffer_shape == [ 4, 5 ]
    assert L.strides == [ 5, 1 ]
    assert L.element_count == 20


if test( "padding_differs_fp32_fp64" ):
    # same 128-byte hardware alignment pads the batch (10) differently by dtype
    caps, mask = [ 10, 3 ], [ True, False ]
    L32 = PhysicalLayout.of( caps, mask, alignment_bytes = 128, itemsize = 4 )
    L64 = PhysicalLayout.of( caps, mask, alignment_bytes = 128, itemsize = 8 )
    assert L32.buffer_shape[ 0 ] == 32          # round_up( 10, 32 )
    assert L64.buffer_shape[ 0 ] == 16          # round_up( 10, 16 )


if test( "two_batch_axes_no_pad_stays_logical" ):
    # two batch axes, no alignment -> nothing moves, so we keep the LOGICAL shape (no flatten): the
    # buffer stays same-rank and contiguous, and the layout is identity (lowering unchanged).
    L = PhysicalLayout.of( caps = [ 2, 3, 5 ], is_batch = [ True, True, False ], alignment_bytes = 1, itemsize = 8 )
    assert L.is_identity
    assert L.buffer_shape == [ 2, 3, 5 ]
    assert L.strides == [ 15, 5, 1 ]   # contiguous of [2,3,5]


if test( "two_batch_axes_padded" ):
    # product of batch sizes 2*3 = 6, padded up to 16 items (128-byte alignment, fp64)
    L = PhysicalLayout.of( caps = [ 2, 3, 4 ], is_batch = [ True, True, False ], alignment_bytes = 128, itemsize = 8 )
    assert not L.is_identity
    assert L.buffer_shape == [ 16, 4 ]           # padded flat batch (16) x inner (4)
    assert L.strides == [ 12, 4, 1 ]
    assert L.element_count == 64
    assert 1 * 12 + 2 * 4 + 3 < L.element_count   # every used logical index stays inside the buffer


if test( "strides_bytes" ):
    L = PhysicalLayout.of( caps = [ 3, 5 ], is_batch = [ True, False ], alignment_bytes = 32, itemsize = 8 )
    assert L.strides_bytes( 8 ) == [ 40, 8 ]


if test( "phys_num_reorders_non_batch_axes" ):
    # hardware physical-order numbers REORDER the non-batch axes (lower = more leading). A pure
    # performance move: the strides keep the logical view, so the logical caps never change.
    # here logical [a=3, b=5], but phys_num asks b before a -> physical buffer is [5, 3].
    L = PhysicalLayout.of( [ 3, 5 ], [ False, False ], phys_num = [ 1, 0 ] )
    assert not L.is_identity
    assert L.caps == [ 3, 5 ]           # logical extents unchanged
    assert L.buffer_shape == [ 5, 3 ]   # physical: b (5) leads, then a (3)
    assert L.strides == [ 1, 3 ]        # a steps by 1, b steps by 3 -> b is the outer physical axis
    assert L.element_count == 15


if test( "phys_num_ties_keep_logical_order" ):
    # EQUAL numbers are tolerated: axes of the same phys_num keep their logical relative order, so
    # an all-equal policy is inert (identity), exactly like passing None.
    assert PhysicalLayout.of( [ 3, 5 ], [ False, False ], phys_num = [ 7, 7 ] ).is_identity
    assert PhysicalLayout.of( [ 3, 5 ], [ False, False ], phys_num = [ 0, 1 ] ).is_identity   # already sorted


if test( "phys_num_reorders_after_batch_group" ):
    # the batch group always leads physically (flattened+padded); phys_num only orders what follows.
    # logical [batch=2, a=3, b=4], b before a -> physical [batch(padded 16), b=4, a=3].
    L = PhysicalLayout.of( [ 2, 3, 4 ], [ True, False, False ], alignment_bytes = 128, itemsize = 8,
                           phys_num = [ 0, 2, 1 ] )
    assert not L.is_identity
    assert L.buffer_shape == [ 16, 4, 3 ]         # flat batch padded to 16, then b(4), then a(3)
    assert L.strides == [ 12, 1, 3 ]              # batch steps 12 (=4*3); a steps 1; b steps 3
    assert L.element_count == 16 * 12


if test( "device_batch_alignment" ):
    # the factor (BYTES) lives on the Device, and a per-call value overrides it
    from sdot.devices.Cpu import Cpu
    from sdot.devices.CudaGpu import CudaGpu

    assert Cpu().batch_alignment == 1
    assert Cpu().resolve_batch_alignment() == 1
    assert Cpu().resolve_batch_alignment( 128 ) == 128     # per-call override wins over the default
    assert CudaGpu( 0 ).batch_alignment == 128
    assert CudaGpu( 0 ).resolve_batch_alignment() == 128
    assert CudaGpu( 0 ).resolve_batch_alignment( 64 ) == 64

    # the physical-order policy is INERT by default: no device reorders anything until it overrides
    # the hook (so today every output keeps its logical axis order -> identity layout).
    assert Cpu().physical_axis_num( [ "row", "col" ] ) is None
    assert CudaGpu( 0 ).physical_axis_num( [ "row", "col" ] ) is None
