"""The RAGGED tensor in CSR ( `loom.CsrTensor` ): rows of different lengths, laid out
end to end.

What needs checking fits in one sentence: IN `csr( i, j )`, `i` LOOKS ITSELF UP IN THE
OFFSETS. The rest -- the prefix sum, the total, the exact allocation -- follows from it.
"""
import numpy

import loom
from loom.testing import host
from loom.compilation.FfiCode import FfiCode
from errand import test


_FILL = FfiCode.per_item( code = """
    const SI i = flat_index;
    for ( SI j = 0; j < outputs.csr.row_size( i ); ++j )
        outputs.csr( i, j ) = 10 * i + j;
""" )


if test( "the_i_is_looked_up_in_the_offsets" ):
    # three rows of lengths 2, 0 and 3 -- one of them EMPTY, which is the case where padding
    # wastes the most and where offset arithmetic is the easiest to get wrong.
    counts = loom.IntTensor[ 3 ]( [ 2, 0, 3 ] )
    csr = loom.CsrTensor.from_counts( counts )

    assert host( csr.offsets.value ).tolist() == [ 0, 2, 2, 5 ]
    assert csr.total == 5          # the last bound IS the total
    assert csr.nb_rows_value == 3

    loom.ffi_call( "csr_fill", _FILL, csr = loom.out( csr, writes = ( "values", ) ), nb_items = 3 )

    # row 0 -> 0, 1 ; row 1 -> nothing ; row 2 -> 20, 21, 22
    assert host( csr.values.value ).reshape( -1 ).tolist() == [ 0, 1, 20, 21, 22 ]
    print( f"3 rows ( 2, 0, 3 ) -> { csr.total } slots, exactly the total" )


if test( "an_empty_row_has_a_zero_size" ):
    # `row_size` is a SUBTRACTION of two neighbouring bounds: an empty row gives 0, and there is
    # no counts array to keep consistent with the offsets.
    sizes = loom.IntTensor[ 4 ]()
    counts = loom.IntTensor[ 4 ]( [ 0, 3, 0, 1 ] )
    csr = loom.CsrTensor.from_counts( counts )

    loom.ffi_call(
        "csr_sizes",
        FfiCode.per_item( code = "outputs.sizes( flat_index ) = inputs.csr.row_size( flat_index );" ),
        csr = csr,
        sizes = loom.out( sizes ),
        nb_items = 4,
    )
    assert host( sizes.value ).reshape( -1 ).tolist() == [ 0, 3, 0, 1 ]


if test( "the_csr_pads_nothing" ):
    # THE POINT: `values` is allocated at the TOTAL, not at `rows x longest_row`. A long row
    # in the middle of short rows is exactly what makes the two diverge.
    counts = loom.IntTensor[ 5 ]( [ 1, 1, 60, 1, 1 ] )
    csr = loom.CsrTensor.from_counts( counts )

    padded = 5 * 60
    assert csr.total == 64
    assert tuple( csr.values.shape ) == ( 64, )
    print( f"padded { padded } slots, csr { csr.total } -- x{ padded / csr.total:.1f}" )
