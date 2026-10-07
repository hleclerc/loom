"""`Machine`: what a kernel can know about the machine, under names that do not refer to any
particular machine.

This was missing for writing a portable kernel: the launch geometry was either hard-coded
( `const int block = 128` in `CudaQueue.h` ), or guessed in Python. A kernel that wants to choose its
group size or its shared-memory budget must be able to ASK for them.

Four fields, and each has a meaning on both sides -- that is what allows writing the kernel once.
The test checks that they make it all the way into a kernel and that they are plausible, then the
constraints specific to each device.
"""
import loom
import loom
from loom import Axis, ShapeVar, IntTensor
from loom.compilation.FfiCode import FfiCode
from errand import test

_FIELDS = ( "nb_workers", "sub_group_width", "local_mem_bytes", "suggested_group" )


# all the C++ is here: `Machine` is read in the kernel under the same names as on the host.
_CODE = """
    struct StoreMachine {
        HD void operator()( auto coords, auto &&args ) const {
            const SI k = coords[ num_field ];
            args.outputs.fields( k ) = k == 0 ? args.machine.nb_workers
                             : k == 1 ? args.machine.sub_group_width
                             : k == 2 ? args.machine.local_mem_bytes
                             :          args.machine.suggested_group;
        }
    };

    void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
        queue.run_parallel( StoreMachine(), batch_axes + args.outputs.fields.domain(), args );
    }
"""


def _machine():
    """The four fields, as the KERNEL sees them ( and not as Python would guess them )."""
    fields = IntTensor[ Axis( ShapeVar( len( _FIELDS ) ), name = "num_field" ) ]()
    loom.ffi_call( "test_machine", FfiCode( code = _CODE ), fields = loom.out( fields ) )
    return dict( zip( _FIELDS, ( int( v ) for v in fields.raw.tolist() ) ) )


if test( "the_four_fields_come_through" ):
    m = _machine()
    print( f"{ loom.resolved_device() } : " + "  ".join( f"{ k }={ v }" for k, v in m.items() ) )

    # nothing must be zero: a portable kernel divides by `sub_group_width` or sizes on
    # `local_mem_bytes`, and a zero would blow up otherwise correct code.
    for name, value in m.items():
        assert value >= 1, ( name, value )

    # a group cannot exceed what can be in flight
    assert m[ "suggested_group" ] <= m[ "nb_workers" ], m


if test( "what_each_device_promises" ):
    m = _machine()

    if getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        # the warp width is 32 on everything that exists; the test pins it so that a change
        # gets noticed rather than going by silently.
        assert m[ "sub_group_width" ] == 32, m
        assert m[ "suggested_group" ] == m[ "sub_group_width" ], m
        # 48 KiB per block is the floor since Fermi
        assert m[ "local_mem_bytes" ] >= 48 * 1024, m
        # SMs x threads per SM: at least a few thousand on a card that exists
        assert m[ "nb_workers" ] >= 1024, m
    else:
        # no lanes on a CPU, hence no sub-group: everything is 1, and the shared-memory
        # budget is NOTIONAL ( `CpuQueue` takes a `std::vector` on the heap ).
        assert m[ "sub_group_width" ] == 1, m
        assert m[ "suggested_group" ] == 1, m
        assert m[ "local_mem_bytes" ] == 64 * 1024, m
        # the thread pool: at least one, and no more than the machine's number of logical cores
        import os
        assert 1 <= m[ "nb_workers" ] <= ( os.cpu_count() or 1 ), m
    print( f"{ loom.resolved_device() } : the device keeps its promises" )
