from .Dtype import Dtype, REAL
from .Tensor import Tensor


class RealTensor( Tensor ):
    """A tensor of REAL numbers -- the differentiable kind, and the default one.

    Its size (FP32 / FP64) is not part of the declaration: it is the driver's policy
    (`driver.ftype`, `SDOT_FTYPE`), resolved late. Pass `dict( size = 32 )` to pin one.
    """

    dtype_kinds = ( REAL, )

    @classmethod
    def default_dtype( cls, size = None ) -> Dtype:
        return Dtype.fp( size )
