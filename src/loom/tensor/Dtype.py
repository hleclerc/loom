class Dtype:
    """
    """

    def __init__( self, floating_point: bool = True, size: int | None = None, signed = True, driver_version = None ) -> None:
        self._driver_version = driver_version # updated during driver instantiation is some cases
        self.floating_point = floating_point
        self.signed = signed
        self.size = size

    @staticmethod
    def factory( value ) -> 'Dtype':
        if isinstance( value, Dtype ):
            return Dtype( value.floating_point, value.size, value.signed, value.driver_version )

        if value is float or value is None:
            return Dtype.fp()

        if value is int:
            return Dtype.si()

        # -------------- str --------------
        sv = str( value ).lower()

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

        # -------------- numpy --------------
        import numpy
        if value is numpy.float16:
            return Dtype.fp( size = 16 )
        if value is numpy.float32:
            return Dtype.fp( size = 32 )
        if value is numpy.float64:
            return Dtype.fp( size = 64 )

        if value is numpy.uint8:
            return Dtype.pi( size = 8 )
        if value is numpy.uint16:
            return Dtype.pi( size = 16 )
        if value is numpy.uint32:
            return Dtype.pi( size = 32 )
        if value is numpy.uint64:
            return Dtype.pi( size = 64 )

        if value is numpy.int8:
            return Dtype.si( size = 8 )
        if value is numpy.int16:
            return Dtype.si( size = 16 )
        if value is numpy.int32:
            return Dtype.si( size = 32 )
        if value is numpy.int64:
            return Dtype.si( size = 64 )

        raise ValueError( f"unsupported type name: { str( value ) }" )

    @staticmethod
    def fp( size: int | None = None ):
        """ make a floating point type """
        return Dtype( floating_point = True, size = size )

    @staticmethod
    def si( size: int | None = None ):
        """ make a signed integer type """
        return Dtype( floating_point = False, size = size, signed = True )

    @staticmethod
    def pi( size: int | None = None ):
        """ make a signed integer type """
        return Dtype( floating_point = False, size = size, signed = False )

    @property
    def name( self ):
        return self.cpp_name

    @property
    def signature( self ):
        return self.cpp_name

    @property
    def cpp_name( self ):
        if self.floating_point:
            if self.size is None:
                return "TF"
            return f"FP{ self.size }"

        if self.size is None:
            return "TI"
        return f"SI{ self.size }"

    @property
    def driver_version( self ):
        if self._driver_version:
            return self._driver_version
        from ..drivers.driver import driver
        return driver.driver_dtype_version( self.floating_point, self.signed, self.size )

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Dtype ):
            value = Dtype.factory( value )
        assert isinstance( value, Dtype )
        return self.floating_point == value.floating_point and self.size == value.size

    def __neq__( self, value, / ) -> bool:
        return not self.__eq__( value )

    def jax_ffi_tensor_type( self ) -> str:
        from .driver import driver
        if self.floating_point:
            return f"xla::ffi::F{ self.size or driver.ftype.size }"
        if self.signed:
            return f"xla::ffi::S{ self.size or driver.itype.size }"
        return f"xla::ffi::U{ self.size or driver.itype.size }"

    def msl_name( self ) -> str:
        """Metal Shading Language scalar type for this dtype (size resolved via the driver)."""
        from .driver import driver
        size = self.size or ( driver.ftype.size if self.floating_point else driver.itype.size )
        if self.floating_point:
            # MSL has no `double`; the Metal binding runs in FP32 anyway.
            return { 16: "half", 32: "float", 64: "float" }[ size ]
        if self.signed:
            return { 8: "char", 16: "short", 32: "int", 64: "long" }[ size ]
        return { 8: "uchar", 16: "ushort", 32: "uint", 64: "ulong" }[ size ]
