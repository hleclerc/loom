from .Storage import Storage


class Unbound( Storage ):
    """No value: the attribute exists but holds nothing, and lowers to a `NoneTensor` -- a TYPE
    carrying the declared scalar/shape/axis names and no data, discriminated at compile time.

    It still carries a `reference_shape`: unbinding a buffer (invalidating a computed cache, say)
    does not unlearn the sizes that were observed from it, and a `ShapeVar` may still be resolving
    its count through them.

    Its lowering needs no override: bound means this call ALLOCATES our buffer (we are its
    output), so the inherited view is exactly right; unbound means there is genuinely nothing, and
    the inherited `NoneTensor` says so. Both spell the same axes, so the aggregate still
    `DEFINE_AXIS`es them either way."""
