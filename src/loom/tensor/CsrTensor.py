"""The RAGGED tensor as CSR: rows of different lengths, stored end to end."""

from ..util.Aggregate import Aggregate
from .CtShapeVar import CtShapeVar
from .IntTensor import IntTensor
from .ShapeVar import ShapeVar
from .Tensor import Tensor
from .Axis import Axis


class CsrTensor( Aggregate ):
    """Rows of different lengths, stored END TO END -- no padding.

        offsets  [ 0, 2, 2, 5 ]        four bounds for three rows
        values   [ a, b, c, d, e ]
        => row 0 = { a, b }, row 1 = {}, row 2 = { c, d, e }

    On the C++ side ( `loom/include/sdot/CsrTensor.h` ), the only gesture to know is:

        csr( i, j )        the `i` looks up the OFFSETS, the `j` indexes into the row
        csr.row_size( i )  the length of row `i`

    WHAT IT TRADES AGAINST loom's PADDED RAGGED ( `ShapeVar[ "axis" ]`, one count per
    row, a `rows x longest_row` buffer ): MEMORY for a PASS. The padded form
    is registered in a single sweep -- reserve a slot and write -- whereas the CSR asks to
    COUNT first, then fill once the offsets are known. Measured on `examples/splats`:
    the padded form costs x2.38 to x5.08 the memory of the CSR there.

    AND A HOST READ, which is the real price: the TOTAL sizes `values`, and it is a count
    that a kernel has just written. A shape built by `from_counts` is therefore not usable under
    `jit` -- exactly what a JIT cannot pay, and what `loom.ffi_call` knows how to do.

    `offsets` carries `nb_rows + 1` BOUNDS and not `nb_rows` counts: the size of a row is then
    a subtraction of two neighbours, with no extra array, and the last bound IS the total.
    """

    offsets   : IntTensor[ "num_bound" ]
    values    : Tensor[ "num_slot" ]

    num_bound : Axis[ "nb_rows + 1" ]
    num_slot  : Axis[ "nb_slots" ]

    nb_rows   : ShapeVar
    nb_slots  : ShapeVar

    @classmethod
    def from_counts( cls, counts, **template_kwargs ):
        """The CSR that a tensor of COUNTS describes: `offsets` is their exclusive prefix sum,
        plus the total as the last bound, and `values` is allocated at EXACTLY that total.

            counts   [ 2, 0, 3 ]   ->   offsets [ 0, 2, 2, 5 ]   and  values of 5 slots

        The prefix sum is done ON THE DEVICE ( `Tensor.cumsum` ). The total, for its part, comes back to
        the host: it is what sizes the allocation, and a size cannot be a device
        value. It is the only thing here that a `jit` cannot go through, and it is what we
        buy in exchange for exactness.

        THE COUNTS CAN HAVE ANY RANK, and are read in the tensor's order: a
        GRID of counts ( a grid of tiles ) gives as many rows, taken row by row.
        This is the only place where a multi-axis structure gets flattened, and that is what a
        CSR is -- cumulative bounds along ONE order, so a sequence of rows and not a grid.

        `template_kwargs` goes to `values` ( `dtype`, `size`, `device` ): what the rows CARRY
        is not decided by the counts.
        """
        from ..drivers import framework_defaults
        from .functions import cumsum

        raw = counts.value if isinstance( counts, Tensor ) else counts
        flat = Tensor.wrap( raw.reshape( -1 ) )

        total = int( flat.sum() )
        nb = int( flat.shape[ 0 ] )

        res = cls( nb_rows = nb, nb_slots = max( total, 1 ),
                   values = dict( template_kwargs ) if template_kwargs else {} )
        # the bounds: the exclusive prefix sum, THEN the total -- `nb + 1` entries. The
        # `concatenate` goes through the driver, so it stays where the data lives.
        starts = cumsum( flat, exclusive = True )
        res.offsets = framework_defaults.ops( starts.value ).concatenate( [ starts.value, framework_defaults.array( [ total ], dtype = starts.dtype ) ] )
        return res

    @property
    def nb_rows_value( self ) -> int:
        """How many rows -- the last bound is not one."""
        return int( self.nb_rows.value )

    @property
    def total( self ) -> int:
        """How many elements in all: the last bound."""
        return int( self.nb_slots.value )
