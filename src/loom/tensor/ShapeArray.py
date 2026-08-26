import numpy


class ShapeArray:
    """A COUNT, on the HOST -- what a `ShapeVar` reads back (`c.nb_dims.value`).

    Not a `Tensor`, and deliberately so. A count is what SIZES things: an allocation, a python
    loop, a `numpy.arange`, an XLA shape. All of those need a value Python actually holds, and a
    `Tensor` is the opposite of that -- it is a backend buffer, so under a `jit` it is a tracer,
    and a tracer sizes nothing. Handing counts back as tensors is what used to force call sites
    to keep a private host duplicate of a count they had already declared.

    So residency, not element type, is what separates the two:

    * `ShapeArray` -- host, numpy-backed, known to Python, never traced, never differentiable.
    * `IntTensor`  -- device, a real buffer, what a kernel reads and writes (`sv.as_tensor()`).

    "Do not trace this" cannot be a flag: in Jax, traced-vs-static is decided at the `jit`
    boundary, and nothing can un-trace a value that already is a tracer. What a type CAN do is
    refuse to be built from one -- which is why the check lives in `__init__`, so the error lands
    where the host value was expected rather than 40 frames downstream in some `arange`.

    Its algebra is deliberately small: it stays a `ShapeArray` only under operations that keep it
    a count. A true division does not, so it hands back plain numpy, and anything more elaborate
    goes through `numpy.asarray( ... )` explicitly.
    """

    __slots__ = ( "raw", "names" )

    def __init__( self, value, names = None ) -> None:
        from ..drivers.driver import driver

        if driver.is_traced( value ):
            raise TypeError(
                "a count cannot be read on the host here: it lives on the device (it is traced), "
                "which is what happens to a count a KERNEL wrote once you are under a `jit`. "
                "Nothing can un-trace it -- either take it as a device value (`shape_var"
                ".as_tensor()`) and compute with it there, or prescribe the count in Python so it "
                "is known before the trace." )
        # a count is an integer, always: `dtype = int` also turns a 0-d device array into a plain
        # host one, which is the whole point of this type.
        self.raw = numpy.asarray( value, dtype = int )
        # one name per dimension, for display -- the `dep_axes` a ragged count varies along.
        self.names = list( names ) if names is not None else [ None ] * self.raw.ndim

    # ---- reading it as what it is: a number, or a small array of numbers ----
    def __int__( self ) -> int:
        return int( self.raw )

    def __float__( self ) -> float:
        return float( self.raw )

    def __index__( self ) -> int:
        """Makes a rank-0 count usable wherever python wants an index or a size: `range( n )`,
        `[ 0 ] * n`, a slice bound. This is the whole reason a count must be a host value."""
        return int( self.raw )

    def __array__( self, dtype = None ):
        return self.raw.astype( dtype ) if dtype is not None else self.raw

    def __bool__( self ) -> bool:
        return bool( self.raw )

    def __len__( self ) -> int:
        return len( self.raw )

    def __iter__( self ):
        for v in self.raw:
            yield ShapeArray( v )

    def __getitem__( self, key ):
        return ShapeArray( self.raw[ key ] )

    @property
    def shape( self ):
        return tuple( self.raw.shape )

    @property
    def ndim( self ) -> int:
        return self.raw.ndim

    @property
    def dtype( self ):
        return self.raw.dtype

    def tolist( self ):
        return self.raw.tolist()

    def max( self ) -> "ShapeArray":
        return ShapeArray( self.raw.max() )

    def min( self ) -> "ShapeArray":
        return ShapeArray( self.raw.min() )

    def sum( self ) -> "ShapeArray":
        return ShapeArray( self.raw.sum() )

    # ---- arithmetic: it stays a COUNT only while the operation keeps it one ----
    def _int_op( self, other, op ):
        return ShapeArray( op( self.raw, numpy.asarray( other, dtype = int ) ), self.names )

    def __add__     ( self, o ): return self._int_op( o, lambda a, b: a +  b )
    def __radd__    ( self, o ): return self._int_op( o, lambda a, b: b +  a )
    def __sub__     ( self, o ): return self._int_op( o, lambda a, b: a -  b )
    def __rsub__    ( self, o ): return self._int_op( o, lambda a, b: b -  a )
    def __mul__     ( self, o ): return self._int_op( o, lambda a, b: a *  b )
    def __rmul__    ( self, o ): return self._int_op( o, lambda a, b: b *  a )
    def __floordiv__( self, o ): return self._int_op( o, lambda a, b: a // b )
    def __mod__     ( self, o ): return self._int_op( o, lambda a, b: a %  b )

    # a ratio is no longer a count, so it leaves this type rather than pretending otherwise
    def __truediv__ ( self, o ): return self.raw / numpy.asarray( o )
    def __rtruediv__( self, o ): return numpy.asarray( o ) / self.raw

    def __eq__( self, o ): return bool( numpy.all( self.raw == numpy.asarray( o ) ) )
    def __ne__( self, o ): return not self.__eq__( o )
    def __lt__( self, o ): return bool( numpy.all( self.raw <  numpy.asarray( o ) ) )
    def __le__( self, o ): return bool( numpy.all( self.raw <= numpy.asarray( o ) ) )
    def __gt__( self, o ): return bool( numpy.all( self.raw >  numpy.asarray( o ) ) )
    def __ge__( self, o ): return bool( numpy.all( self.raw >= numpy.asarray( o ) ) )

    def __hash__( self ):
        return hash( ( self.raw.shape, self.raw.tobytes() ) )

    def __repr__( self ) -> str:
        named = "" if all( n is None for n in self.names ) else f", axes={ self.names }"
        return f"ShapeArray( { self.raw.tolist() }{ named } )"
