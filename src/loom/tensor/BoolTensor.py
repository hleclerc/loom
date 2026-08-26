from .Dtype import Dtype, BOOL
from .Tensor import Tensor


class BoolTensor( Tensor ):
    """A tensor of BOOLEANS -- what a comparison produces, and what a mask is.

    Rarely declared on a field; mostly it is the type an op RESULT turns out to have, which is
    why it exists as a class at all: `t > 0` must be able to say what it is (see `_result_dtype`).
    Not differentiable, and it has no size to choose.
    """

    dtype_kinds = ( BOOL, )
    is_differentiable = False

    @classmethod
    def default_dtype( cls, size = None ) -> Dtype:
        assert size is None, "a boolean tensor has no size"
        return Dtype.bo()
