from .Parametrized import Parametrized

def annotations( cls ):
    if isinstance( cls, Parametrized ):
        cls = cls.cls

    res = {}
    for klass in reversed( cls.__mro__ ):
        for name, attr in getattr( klass, '__annotations__', {} ).items():
            res[ name ] = attr
    return res
