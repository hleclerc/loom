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
        from .Settings import Settings
        return Settings( self, device, ftype, itype )

    def _make_operations( self ):
        from .JaxOperations import JaxOperations
        return JaxOperations()
