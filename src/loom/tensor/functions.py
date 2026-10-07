"""Free-function form of the tensor operations.

Every one of these is the method of the same name -- `loom.dot( a, b, "i" )` IS `a.dot( b, "i" )`.
Neither form is the "real" one: some expressions read better as a chain (`t.sum( "i" ).sqrt()`),
others as a call (`dot( normals, points, "xy" )`), and a pipeline of free functions composes where
a method chain does not. They are kept in step deliberately -- anything reachable one way is
reachable the other.

The reduction names shadow python builtins (`sum`, `min`, `max`, `all`, `any`, `abs`), exactly as
numpy's do, so reach them through the module (`loom.sum( t, "i" )`) rather than importing them bare.
"""
import copy

from ..util.Aggregate import Aggregate
from .Tensor import Tensor


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


def cumsum( t, axis = None, *, exclusive = False ):
    """The prefix sum along `axis` -- a SCAN, not a reduction: see `Tensor.cumsum`."""
    return t.cumsum( axis, exclusive = exclusive )


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


def exp( t ):
    return t.exp()


def abs( t ):
    return t.__abs__()


def clip( t, lo = None, hi = None ):
    """Values clamped to `[ lo, hi ]`; either bound may be `None` (unbounded)."""
    return t.clip( lo, hi )


def stop_gradient( x ):
    """Same values, detached from the gradient tape -- for a quantity needed for its VALUE only,
    whose derivative is supplied by another (better conditioned) route.

    AN AGGREGATE ANSWERS WITH A TWIN OF ITSELF: every tensor it holds detached, and everything else
    -- the counts, the axes -- the very SAME objects. Sharing them is not a saving, it is the only
    correct answer: a count restated is a count that can disagree, and an axis is a REFERENCE, so a
    twin with axes of its own would be a second geometry that merely happens to match. It also
    spares the caller the field-by-field copy, which is the code that goes stale the day a field is
    added."""
    if isinstance( x, Aggregate ):
        return _detached_aggregate( x )
    return x.stop_gradient()


def _detached_aggregate( agg ):
    # A SHALLOW COPY, then the only fields that have a derivative are replaced. Share by default and
    # detach by exception, rather than rebuild and re-share: what the schema gains tomorrow arrives
    # shared, which is right for everything that is not a tensor -- and a custom `__init__` (a
    # `CsrTensor` has one) is never called, so this works on any aggregate.
    res = copy.copy( agg )
    for name, attr in agg.__dict__.items():
        if isinstance( attr, Aggregate ):
            res.__dict__[ name ] = _detached_aggregate( attr )
        elif isinstance( attr, Tensor ) and attr.is_defined:
            # the field NAME rides along: a detached tensor is still the member the C++ spells.
            detached = attr.stop_gradient()
            detached.name = attr.name
            res.__dict__[ name ] = detached
    return res


# ---- structure ----
def transpose( t, *axes ):
    """The dimensions permuted; no argument reverses them. Each entry is a position or an axis
    name, and the names follow the move."""
    return t.transpose( *axes )
