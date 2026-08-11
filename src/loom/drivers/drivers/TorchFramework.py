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
        from .TorchDriver import TorchDriver
        return TorchDriver( self, device, ftype, itype )
