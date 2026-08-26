import numpy

# The KINDS a `Dtype` can have. Semantic, per-field, machine-independent -- unlike the size.
REAL = "real"       # floating point
SINT = "sint"       # signed integer
UINT = "uint"       # unsigned integer
BOOL = "bool"       # boolean (what a comparison produces)


class Dtype:
    """The ELEMENT contract of a tensor: a KIND (real / signed int / unsigned int / bool)
    and a SIZE in bits.

    The two are not the same kind of decision, which is why they live in one object but are
    read separately:

    * the KIND is semantic and per-field -- an index tensor is integer whatever the machine.
    * the SIZE is a global policy (`driver.ftype` / `driver.itype`, `SDOT_FTYPE` / `SDOT_ITYPE`).
      `None` means "whatever the driver runs with", resolved LATE (see `driver_version`), so a
      declaration written at import time does not freeze a size the user has not chosen yet.

    A `Dtype` is a DECLARATION. `__eq__` compares declarations (an unresolved size is not 64);
    `same_as` compares what they DENOTE, sizes resolved through the driver -- which is what a
    check against a real buffer needs.
    """

    def __init__( self, kind: str = REAL, size: int | None = None, driver_version = None ) -> None:
        assert kind in ( REAL, SINT, UINT, BOOL ), f"unknown dtype kind: { kind }"
        # a bool has no size to choose: it is what the framework spells `bool`.
        assert not ( kind == BOOL and size is not None ), "a boolean dtype has no size"
        self._driver_version = driver_version # updated during driver instantiation is some cases
        self.kind = kind
        self.size = size

    @staticmethod
    def factory( value ) -> 'Dtype':
        if isinstance( value, Dtype ):
            # `_driver_version`, NOT the property: resolving it here would instantiate the driver
            # merely to COPY a declaration (a `Tensor` field is built long before any kernel runs).
            return Dtype( value.kind, value.size, value._driver_version )

        if value is float or value is None:
            return Dtype.fp()

        if value is int:
            return Dtype.si()

        if value is bool:
            return Dtype.bo()

        # -------------- numpy / framework dtype objects --------------
        # a `numpy.dtype`, or anything numpy can read as one (`jnp.float32`, an array's `.dtype`).
        # Tried BEFORE the string parsing below, which would otherwise mis-read `str( dtype )`.
        if isinstance( value, numpy.dtype ) or ( isinstance( value, type ) and issubclass( value, numpy.generic ) ):
            return Dtype.from_numpy( value )

        # -------------- str --------------
        sv = str( value ).lower()

        if sv == "bool":
            return Dtype.bo()

        if sv == "int":
            return Dtype.si()

        if sv.startswith( "fp" ):
            return Dtype.fp( size = int( sv[ 2: ] ) )

        if sv.startswith( "float" ):
            return Dtype.fp( size = int( sv[ 5: ] ) )

        if sv.startswith( "si" ):
            return Dtype.si( size = int( sv[ 2: ] ) )

        if sv.startswith( "int" ):
            return Dtype.si( size = int( sv[ 3: ] ) )

        if sv.startswith( "pi" ):
            return Dtype.pi( size = int( sv[ 2: ] ) )

        if sv.startswith( "unsigned" ):
            return Dtype.pi( size = int( sv[ 8: ] ) )

        raise ValueError( f"unsupported type name: { str( value ) }" )

    @staticmethod
    def from_numpy( value ) -> 'Dtype':
        """The `Dtype` a numpy (or numpy-readable) dtype denotes -- how a real BUFFER answers what
        it actually is. Sizes are CONCRETE here: this describes storage, not a declaration."""
        dt = numpy.dtype( value )
        if dt.kind == "b":
            return Dtype.bo()
        if dt.kind == "f":
            return Dtype.fp( size = 8 * dt.itemsize )
        if dt.kind == "i":
            return Dtype.si( size = 8 * dt.itemsize )
        if dt.kind == "u":
            return Dtype.pi( size = 8 * dt.itemsize )
        raise ValueError( f"unsupported numpy dtype: { dt }" )

    @staticmethod
    def of( raw ) -> 'Dtype':
        """The dtype a backend buffer ACTUALLY has (via the driver, so Jax and Torch answer the
        same way). This is the truthful direction: a buffer knows its type, a declaration only
        claims one."""
        from ..drivers.driver import driver
        return driver.dtype_of( raw )

    @staticmethod
    def fp( size: int | None = None ):
        """ make a floating point type """
        return Dtype( REAL, size )

    @staticmethod
    def si( size: int | None = None ):
        """ make a signed integer type """
        return Dtype( SINT, size )

    @staticmethod
    def pi( size: int | None = None ):
        """ make an unsigned integer type """
        return Dtype( UINT, size )

    @staticmethod
    def bo():
        """ make a boolean type (what a comparison produces) """
        return Dtype( BOOL )

    # ---- kind predicates: what the rest of the code actually asks ----
    @property
    def floating_point( self ) -> bool:
        return self.kind == REAL

    @property
    def integer( self ) -> bool:
        return self.kind in ( SINT, UINT )

    @property
    def boolean( self ) -> bool:
        return self.kind == BOOL

    @property
    def signed( self ) -> bool:
        return self.kind in ( REAL, SINT )

    @property
    def differentiable( self ) -> bool:
        """Whether a gradient can flow through a value of this type: only a real one can. This is
        the predicate the FFI uses to decide what is a primal (see `JaxFfi`)."""
        return self.kind == REAL

    @property
    def name( self ):
        return self.cpp_name

    @property
    def signature( self ):
        return self.cpp_name

    @property
    def cpp_name( self ):
        """The C++ spelling (see `support/common_types.h`): `FP64`, `SI32`, `PI32`, `bool` -- or
        the driver-resolved aliases `TF` / `TI` when the size is left to the driver."""
        if self.kind == BOOL:
            return "bool"
        if self.size is None:
            return { REAL: "TF", SINT: "TI", UINT: "TU" }[ self.kind ]
        return { REAL: "FP", SINT: "SI", UINT: "PI" }[ self.kind ] + str( self.size )

    @property
    def driver_version( self ):
        if self._driver_version:
            return self._driver_version
        from ..drivers.driver import driver
        return driver.driver_dtype_version( self.kind, self.size )

    def resolved( self ) -> 'Dtype':
        """This dtype with its size FILLED IN from the driver -- what it will really be on the
        machine. A declaration left open (`size is None`) only becomes concrete here."""
        return Dtype.from_numpy( numpy.dtype( self.driver_version ) )

    def same_as( self, other ) -> bool:
        """Whether both denote the same MACHINE type, sizes resolved through the driver -- so a
        declared `fp` (size left open) matches a concrete FP64 when that is what the driver runs.
        This is what a check against a real buffer must use, not `__eq__`."""
        return numpy.dtype( self.driver_version ) == numpy.dtype( Dtype.factory( other ).driver_version )

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Dtype ):
            value = Dtype.factory( value )
        return self.kind == value.kind and self.size == value.size

    def __hash__( self ) -> int:
        return hash( ( self.kind, self.size ) )

    def __repr__( self ) -> str:
        return f"Dtype( { self.cpp_name } )"
