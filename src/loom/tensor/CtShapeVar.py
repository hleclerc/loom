from typing import TYPE_CHECKING
from .ShapeVar import ShapeVar


class CtShapeVar( ShapeVar ):
    """A `ShapeVar` whose value is known at C++ *compile* time.

    Emitted as a compile-time constant (a literal `static constexpr`) in the generated source
    rather than crossing the FFI as a runtime buffer. Because the value is baked into the
    source, it is part of the library-name hash -- each distinct value yields a distinct
    compiled library (enabling template specialization, loop unrolling, ...).

    It must therefore be *prescribed* (`Cell( nb_dims = 2 )`), and it is the one case where
    count and capacity coincide: the value is in the type, so a buffer sized on it is sized on
    exactly what the kernel will see. Nothing to allocate for, nothing for a kernel to write.
    """

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_CtShapeVar import CallArg_CtShapeVar
        return CallArg_CtShapeVar( caa, path, name, inst )

    def accept_capacity( self, capacity ):
        raise ValueError(
            f"CtShapeVar '{ self.name }' is compile-time: its count IS its capacity, so a call "
            f"cannot allocate a different one. Prescribe it instead: { self.name } = ..."
        )

    def set_count( self, value ):
        raise ValueError( f"CtShapeVar '{ self.name }' is compile-time: a kernel cannot write it" )

    if TYPE_CHECKING:
         def __lt__( self, that ) -> bool: ...
         def __le__( self, that ) -> bool: ...
         def __gt__( self, that ) -> bool: ...
         def __ge__( self, that ) -> bool: ...
         def __eq__( self, that ) -> bool: ...
