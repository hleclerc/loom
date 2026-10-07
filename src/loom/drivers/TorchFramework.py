from .Framework import Framework


class TorchFramework( Framework ):
    @property
    def module_name( self ):
        return "torch"

    @property
    def can_be_imported( self ):
        try:
            import torch as torch
            return True
        except:
            return False

    def make_instance( self, device, ftype, itype ):
        from .Settings import Settings
        return Settings( self, device, ftype, itype )

    def _make_operations( self ):
        from .TorchOperations import TorchOperations
        return TorchOperations()
