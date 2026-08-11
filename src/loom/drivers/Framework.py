
class Framework:
    """
    """

    @staticmethod
    def factory( value ) -> 'Framework':
        if isinstance( value, Framework ):
            return value

        value = str( value ).lower()
        match value:
            case "torch" | "pytorch":
                from .TorchFramework import TorchFramework
                return TorchFramework()
            case "jax":
                from .JaxFramework import JaxFramework
                return JaxFramework()
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

    @property
    def module_name( self ):
        raise NotImplementedError

    @property
    def can_be_imported( self ):
        raise NotImplementedError

    def make_instance( self, device, ftype, itype ):
        raise NotImplementedError
