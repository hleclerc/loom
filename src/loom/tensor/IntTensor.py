from .Dtype import Dtype, SINT, UINT
from .Tensor import Tensor


class IntTensor( Tensor ):
    """A tensor of INTEGERS -- indices, offsets, item maps, counts.

    Not differentiable: no gradient flows through an index, so the FFI never seeks a primal
    for one and a backward never seeds a cotangent buffer for it (see `is_differentiable`).

    Its size is the driver's integer policy (`driver.itype`, `SDOT_ITYPE`) unless pinned with
    `dict( size = 32 )`; `dict( dtype = "pi32" )` makes it unsigned.
    """

    dtype_kinds = ( SINT, UINT )
    is_differentiable = False

    @classmethod
    def default_dtype( cls, size = None ) -> Dtype:
        return Dtype.si( size )
