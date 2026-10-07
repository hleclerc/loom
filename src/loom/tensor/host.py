import numpy


def to_host( x ):
    """`x` as a host numpy array. A value that already is one -- the common case for a count, which the host
    holds -- needs no framework to be asked."""
    t = type( x )
    if t is numpy.ndarray:
        return x
    if t is int or t is float or isinstance( x, numpy.generic ):
        return numpy.asarray( x )
    from ..drivers import framework_defaults
    return framework_defaults.ops( x ).to_numpy( x )
