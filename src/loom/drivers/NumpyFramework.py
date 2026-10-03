from .Framework import Framework


class NumpyFramework( Framework ):
    """Plain numpy, CPU only, forward only: what is left when neither jax nor torch is installed
    (see `NumpyDriver`). It is the LAST resort of the default choice -- numpy is imported by nearly
    everything, so its presence in `sys.modules` says nothing about what the user wants."""

    is_fallback = True

    @property
    def module_name( self ):
        return "numpy"

    @property
    def can_be_imported( self ):
        try:
            import numpy as numpy
            return True
        except ImportError:
            return False

    def make_instance( self, device, ftype, itype ):
        from .NumpyDriver import NumpyDriver
        return NumpyDriver( self, device, ftype, itype )
