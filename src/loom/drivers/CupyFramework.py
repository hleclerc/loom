from .Framework import Framework


class CupyFramework( Framework ):
    """CuPy: numpy's interface on a CUDA card, forward only, like numpy (see `CupyDriver`)."""

    @property
    def module_name( self ):
        return "cupy"

    @property
    def can_be_imported( self ):
        # a module that imports on a machine with no card is of no use: there is nothing for it to run on
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def make_instance( self, device, ftype, itype ):
        from .Settings import Settings
        return Settings( self, device, ftype, itype )

    def _make_operations( self ):
        from .CupyOperations import CupyOperations
        return CupyOperations()
