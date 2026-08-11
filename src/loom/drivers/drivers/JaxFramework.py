from .Framework import Framework


class JaxFramework( Framework ):
    @property
    def module_name( self ):
        return "jax"

    @property
    def can_be_imported( self ):
        try:
            import jax as jax
            return True
        except:
            return False

    def make_instance( self, device, ftype, itype ):
        from .JaxDriver import JaxDriver
        return JaxDriver( self, device, ftype, itype )
