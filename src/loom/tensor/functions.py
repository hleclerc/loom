"""Free-function form of the tensor operations.

Every one of these is the method of the same name -- `loom.dot( a, b, "i" )` IS `a.dot( b, "i" )`.
Neither form is the "real" one: some expressions read better as a chain (`t.sum( "i" ).sqrt()`),
others as a call (`dot( normals, points, "xy" )`), and a pipeline of free functions composes where
a method chain does not. They are kept in step deliberately -- anything reachable one way is
reachable the other.

The reduction names shadow python builtins (`sum`, `min`, `max`, `all`, `any`, `abs`), exactly as
numpy's do, so reach them through the module (`loom.sum( t, "i" )`) rather than importing them bare.
"""


def dot( a, b, over ):
    """Contract `a` and `b` over the SHARED axis `over` -- `( a * b ).sum( over )`. The axis is
    matched by identity, so this assumes no axis order (unlike `@`)."""
    return a.dot( b, over )


def where( cond, a, b ):
    """`a` where `cond` is true, `b` elsewhere. The three are aligned by axis identity, and either
    branch may be a plain scalar."""
    return cond.where( a, b )


# ---- reductions: `axis` is None (everything), an axis name, a position, or a tuple of those ----
def sum( t, axis = None ):
    return t.sum( axis )


def prod( t, axis = None ):
    return t.prod( axis )


def min( t, axis = None ):
    return t.min( axis )


def max( t, axis = None ):
    return t.max( axis )


def mean( t, axis = None ):
    """Divided by the count of REAL cells, not by the bounding box -- so it is right on a ragged
    tensor, whose box holds padding."""
    return t.mean( axis )


def all( t, axis = None ):
    return t.all( axis )


def any( t, axis = None ):
    return t.any( axis )


# ---- elementwise maps: the shape, hence every axis, is preserved ----
def sqrt( t ):
    return t.sqrt()


def arcsin( t ):
    return t.arcsin()


def abs( t ):
    return t.__abs__()


def clip( t, lo = None, hi = None ):
    """Values clamped to `[ lo, hi ]`; either bound may be `None` (unbounded)."""
    return t.clip( lo, hi )


def stop_gradient( t ):
    """Same values, detached from the gradient tape -- for a quantity needed for its VALUE only,
    whose derivative is supplied by another (better conditioned) route."""
    return t.stop_gradient()


# ---- structure ----
def transpose( t, *axes ):
    """The dimensions permuted; no argument reverses them. Each entry is a position or an axis
    name, and the names follow the move."""
    return t.transpose( *axes )
