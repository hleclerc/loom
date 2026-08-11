from .IoCategory import IoCategory
from .CallArg import CallArg

class CallArg_CtShapeVar( CallArg ):
    """A compile-time `ShapeVar`: its count is carried by the C++ TYPE, not by an FFI buffer.

    It lowers to a `Ct<SI, value>` member -- the value lives in the type, so C++ code can tell
    it is compile-time known (and specialize, unroll, ...). Being part of the generated source,
    it is also part of the library-name hash.

    It is therefore never an output: a kernel cannot write into a type. Being swept in by an
    aggregate declared as an output is fine (we are just skipped); being named as one is a
    contradiction, and is refused.
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( IoCategory.INPUT, name )

        call_args_analysis.io_category( path, True )   # so a covering declaration counts as used
        if call_args_analysis.is_exact_output( path ):
            raise ValueError( f"CtShapeVar '{ name }' is compile-time: a kernel cannot write it" )

        self.value = int( inst.raw )

    # -- driver-agnostic C++ (the same for every driver) --
    def cpp_type( self ):
        return f"Ct<SI, { self.value }>"

    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    # -- Jax FFI ABI --
    def jax_cpp_init( self ):
        return f"{ self.cpp_type() }{{}}"
