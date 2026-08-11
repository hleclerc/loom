from .IoCategory import IoCategory
from .CallArg import CallArg


class CallArg_Attr( CallArg ):
    """A plain Python `int` handed to the call: a RUNTIME value, uniform across the call, that
    crosses as an XLA FFI ATTRIBUTE rather than through a buffer.

    An attribute is baked into the CALL, not into the kernel, so the SAME compiled kernel serves
    every value -- the contrast with a `CtShapeVar`, whose value lives in the type and recompiles
    the kernel for each distinct one. It reaches the handler as an extra `int64_t` parameter, the
    body reads it as an `SI`, and it enters `run_parallel` as a read-only input (`InpList()`),
    made available by value (an arithmetic value crosses untouched, see make_avaiable.h).

    It is never an output: a kernel cannot write into an attribute (there is nowhere for a result
    to go). Being named as one is a contradiction, and is refused.
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( IoCategory.INPUT, name )

        call_args_analysis.io_category( path, True )   # so a covering declaration counts as used
        if call_args_analysis.is_exact_output( path ):
            raise ValueError( f"attribute '{ name }' is a runtime value: a kernel cannot write it" )

        self.value = int( inst )

    # -- driver-agnostic C++ (the same for every driver) --
    def cpp_root_decl( self, var_name ):
        # the attribute reaches the handler under its OWN name, the body reads it under the
        # argument's -- two names, so the local does not read itself in its own initializer.
        return f"    auto { var_name } = SI( { self._jax_attr_name() } );"

    # `cpp_run_parallel_pair` (base) already gives `InpList(), <name>`: a read-only input.

    # -- Jax FFI ABI --
    def _jax_attr_name( self ):
        # the FFI attributes share one flat namespace with the buffers (`ffi_*`) and the capacity
        # bounds (`max_ffi_*`); a prefix of our own keeps an attribute apart from both.
        return f"attr_{ self.name }"

    def jax_attrs( self ):
        """The scalars this node needs at run time, but NOT through a buffer: an XLA FFI attribute
        is baked into the call, so the same compiled kernel serves every value."""
        return [ ( self._jax_attr_name(), "int64_t", self.value ) ]
