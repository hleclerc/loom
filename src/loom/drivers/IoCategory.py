from enum import Enum


class IoCategory( Enum ):
    """How one attribute of a call argument crosses (or does not cross) the FFI boundary.

    Inputs and outputs are DISJOINT: a kernel never writes what it reads. What looks like an
    in-place update is a Python-side rebinding, done by the caller between two calls -- so
    there is no aliasing to arrange, and no reconciling of an input capacity with an output
    one under a single name.

    The category is not a free declaration: `OUTPUT` is declared (`output_attributes`), the
    other two are *observed*. An attribute that holds data is an input; an empty, undeclared
    one is simply not bound -- it may just be an optional field this kernel does not use, and
    the C++ side sees a null `TensorView` it can test.
    """

    INPUT = "input"      # holds data: bound at the size that data actually has
    OUTPUT = "output"    # declared: a fresh buffer, allocated at the capacity declared in Python
    UNBOUND = "unbound"  # empty and undeclared: nothing crosses the FFI

    @property
    def is_input( self ):
        return self is IoCategory.INPUT

    @property
    def is_output( self ):
        return self is IoCategory.OUTPUT

    @property
    def is_bound( self ):
        return self is not IoCategory.UNBOUND
