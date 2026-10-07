from .Storage import Storage
from .cpp_spellings import absent_type


class Zero( Storage ):
    """A shaped, typed, STORAGELESS value that reads as 0 -- the framework's own symbolic-zero
    cotangent, or one we mint. It lowers to a `ZeroTensor`, dropped at compile time.

    It holds a value (so it is not `Unbound`) but backs no buffer (so `buffer` is `None`, and it
    binds nothing across the FFI). That is the whole distinction, and it needs no flag anywhere
    else."""

    holds_value      = True
    is_symbolic_zero = True

    def absent_cpp_type( self, arg ):
        # like a `NoneTensor`, a TYPE with no data -- except it READS AS 0 wherever indexed, so the
        # kernel needs no branch and the compiler drops the arithmetic it feeds. Only this side is
        # overridden: backing no buffer, a symbolic zero is never on the bound one.
        return absent_type( "ZeroTensor", arg )
