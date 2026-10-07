
class Framework:
    """
    """

    @staticmethod
    def factory( value ) -> 'Framework':
        if isinstance( value, Framework ):
            return value

        value = str( value ).lower()
        if value in Framework._named:
            return Framework._named[ value ]
        res = Framework._build( value )
        Framework._named[ value ] = res
        return res

    _named: dict = {}

    @staticmethod
    def _build( value ) -> 'Framework':
        match value:
            case "torch" | "pytorch":
                from .TorchFramework import TorchFramework
                return TorchFramework()
            case "jax":
                from .JaxFramework import JaxFramework
                return JaxFramework()
            case "cupy":
                from .CupyFramework import CupyFramework
                return CupyFramework()
            case "numpy" | "np":
                from .NumpyFramework import NumpyFramework
                return NumpyFramework()
            case _:
                raise ValueError( f"unsupported framework name: { value }" )

    def __repr__( self ) -> str:
        return self.module_name

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Framework ):
            value = Framework.factory( value )
        return str( self ) == str( value )

    def __neq__( self, value, / ) -> bool:
        return not self.__eq__( value )

    # a framework that is only ever the default when nothing else can be imported (see `NumpyFramework`)
    is_fallback = False

    @property
    def module_name( self ):
        raise NotImplementedError

    @property
    def can_be_imported( self ):
        raise NotImplementedError

    def make_instance( self, device, ftype, itype ):
        raise NotImplementedError

    @property
    def operations( self ):
        """What this framework DOES with an array that already exists (`sum`, `matmul`, `jit`, a
        reading as numpy...). Stateless -- it needs no device nor default size -- and built on first use,
        since it imports the framework itself."""
        res = self.__dict__.get( "_operations" )
        if res is None:
            res = self.__dict__[ "_operations" ] = self._make_operations()
        return res

    def _make_operations( self ):
        raise NotImplementedError
