"""The "GROUP kernel" facility (`FfiCode.per_item( group_size = ... )`), for its own sake.

It used to be exercised only by `OtPlan1d`, through a cooperative radix sort and an optimal
transport sweep: when it breaks, the symptom is a wrong transport cost of 3%, which
points to nothing. These tests check the CONTRACT, in three separable assertions:

    1. every lane of the group runs (`local_index` does cover `0..local_size-1`);
    2. `local_scratch` is SHARED by the group -- what one lane writes there, another reads;
    3. `group_barrier` orders those writes before those reads.

A backend that makes `local_scratch` private per lane, or that only runs one lane out of two, fails
here, at a spot that names it.
"""
import numpy

import loom
from loom import driver
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, IntTensor, ShapeVar
from errand import test


def _sum_over_lanes( group_size ):
    """Each lane drops its rank into `local_scratch`, then lane 0 sums its contents.

    The sum equals `0+1+...+(local_size-1)` IF AND ONLY IF the three guarantees hold: a missing
    lane removes its term, a private scratch leaves only one, a missing barrier loses
    some at random. The expected result is therefore a single value, not an interval.
    """
    num_group = Axis( ShapeVar( 2 ), name = "num_group" )
    res = IntTensor[ num_group ]()

    loom.ffi_call(
        f"test_group_kernel_{ group_size }",
        FfiCode.per_item( code = """
                local_scratch[ local_index ] = local_index;
                group_barrier( group );
                if ( local_index == 0 ) {
                    int s = 0;
                    for ( int k = 0; k < local_size; ++k )
                        s += local_scratch[ k ];
                    outputs.res( group_index ) = s;
                }
            """,
            max_nb_threads = "return outputs.res.shape( 0 );",
            group_size = f"return { group_size };",
            local_mem_elems = f"return { group_size };" ),
        res = loom.out( res ),
    )
    return int( numpy.asarray( res.value ).reshape( -1 )[ 0 ] )


if test( "a_group_of_one_degenerates_to_the_plain_kernel" ):
    # `group_size == 1` is the PRODUCTION case on CPU (`Cpu.group_size`): the cooperative path
    # must reduce exactly to the ordinary kernel there. One lane, its rank is 0, zero sum.
    assert _sum_over_lanes( 1 ) == 0


if test( "every_lane_of_a_group_runs_and_shares_its_scratch" ):
    # the real test: beyond one lane, the three guarantees become observable.
    for gs in ( 2, 4, 8 ):
        got = _sum_over_lanes( gs )
        assert got == gs * ( gs - 1 ) // 2, f"group_size={ gs }: sum={ got }, expected { gs * ( gs - 1 ) // 2 }"


def _runtime_subgroup_width( group_size ):
    """The sub-group width that the BACKEND reports at execution time, for this `group_size`."""
    num_group = Axis( ShapeVar( 2 ), name = "num_group" )
    res = IntTensor[ num_group ]()

    loom.ffi_call(
        f"test_group_sgw_{ group_size }",
        FfiCode.per_item( code = """
                if ( local_index == 0 )
                    outputs.res( group_index ) = SI( sub_group.get_local_linear_range() );
            """,
            max_nb_threads = "return outputs.res.shape( 0 );",
            group_size = f"return { group_size };",
            local_mem_elems = f"return { group_size };" ),
        res = loom.out( res ),
    )
    return int( numpy.asarray( res.value ).reshape( -1 )[ 0 ] )


if test( "the_runtime_subgroup_width_matches_what_the_device_claims" ):
    # `Device.subgroup_size` is ENGRAVED on the Python side in the local memory that `OtPlan1d` reserves
    # (`local_mem_elems`, "kept in sync by hand"). If the backend reports another one at
    # execution time, the number of reserved ranks and the number of used ranks diverge -- and the
    # cooperative sort writes outside its reservation, with nothing to say so. Hence this test: it is
    # an invariant maintained BY HAND, so exactly the kind that rots silently.
    # `Device.subgroup_size` is the HARDWARE width (32 on CUDA), hence an UPPER BOUND: a
    # work-group of 4 items does not have a sub-group of 32. The expectation is `min( group_size, claimed )`
    # -- which is indeed what `OtPlan1d`'s sizing assumes, `num_sg = ceil( gs / sgs )`.
    claimed = driver.device.subgroup_size
    for gs in ( 1, 2, 4, 8 ):
        got = _runtime_subgroup_width( gs )
        expected = min( gs, claimed )
        assert got == expected, ( f"group_size={ gs }: the backend reports a sub-group of { got }, "
                                  f"expected min( { gs }, { claimed } ) = { expected }" )


def _probe( group_size, expr, tag ):
    num_lane = Axis( ShapeVar( group_size ), name = "num_lane" )
    res = IntTensor[ num_lane ]()
    loom.ffi_call(
        f"probe_sg_{ group_size }_{ tag }",
        FfiCode.per_item( code = f"outputs.res( local_index ) = SI( { expr } );",
            max_nb_threads = "return 1;",
            group_size = f"return { group_size };",
            local_mem_elems = f"return { group_size };" ),
        res = loom.out( res ),
    )
    return numpy.asarray( res.value ).reshape( -1 ).tolist()


if test( "the_lane_to_subgroup_mapping_is_linear" ):
    # What `OtPlan1d`'s cooperative radix sort needs: that each lane knows WHICH
    # sub-group it is in, in order to receive a distinct piece of the input.
    #
    # Asking the backend does not work. `omp.library-only` returns `get_group_linear_id() == 0`
    # for EVERY lane, while giving distinct `local_index`es -- the split then collapsed
    # onto the same piece for all of them, which sorted the same slice towards the same
    # destinations. The symptom was a wrong transport cost of 3%, with no out-of-bounds
    # access to flag it: nothing pointed to the sub-groups.
    #
    # The identifier is therefore DERIVED from `local_index` (see `OtPlan1d.cxx::sort_diracs`), and it is
    # this derivation that this test locks down: distinct per sub-group, covering 0..num_sg-1.
    for gs in ( 1, 2, 4, 8 ):
        widths = _probe( gs, "sub_group.get_local_linear_range()", "rng" )
        assert len( set( widths ) ) == 1, f"non-uniform sub-group width within the group: { widths }"
        w = widths[ 0 ]
        derived = sorted( set( li // w for li in _probe( gs, "local_index", "li" ) ) )
        num_sg = ( gs + w - 1 ) // w
        assert derived == list( range( num_sg ) ), \
            f"group_size={ gs }, sub-group={ w }: derived sub-groups { derived }, expected 0..{ num_sg - 1 }"
