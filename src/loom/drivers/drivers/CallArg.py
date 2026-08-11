class CallArg:
    """One node of the lowering tree: how a Python attribute reaches the C++ kernel.

    Codegen is split by concern, and a node opts in by simply DEFINING the method (the parent
    filters on `hasattr`, see `CallArg_Aggregate._fields`): `cpp_*` emits the driver-agnostic
    C++ (the struct is the same for Jax or Torch), `jax_*` carries the Jax FFI ABI. A node with
    no buffer (a `CtShapeVar`) just has no `jax_*` buffer methods.

    As a member of an aggregate, a node is ONE template parameter of it -- its own type, spelled
    out at instantiation (`TensorView<...>`, `NoneTensor<...>`, `ShapeVarView<...>`, `Ct<SI,2>`).
    So what a member IS is decided per call, and the C++ body reads its scalar type, rank, axis
    names or compile-time value straight off it.
    """

    def __init__( self, io_category, name ) -> None:
        self.io_category = io_category
        self.name = name

    def _clone( self, mapping ):
        """A copy of this node, for a lowering of the SAME objects under different conditions (a
        `vmap` derives one that carries a batch axis). The Python objects are shared, not copied:
        the write-back must reach the very tensors the caller handed us."""
        import copy
        res = copy.copy( self )
        mapping[ id( self ) ] = res
        return res

    def takes_batch_axis( self ):
        """Whether a `vmap` gives us one more (leading) axis. True of anything that holds ONE
        value per batch item -- which is everything the kernel writes, save the buffers that
        belong to the call itself rather than to an item (the error buffer, see
        `CallArg_Errors`)."""
        return True

    def batch_dim_expr( self, name ):
        """Where the SIZE of a batch axis can be read at run time, if we carry it: an extent of
        our buffer, like any other (a node that carries no batch axis answers None)."""
        return None

    def cpp_tpl_name( self ):
        """Our template parameter, named after the field: unique within the struct that holds
        us, which is the only scope it lives in."""
        return f"T_{ self.name }"

    def children( self ):
        """The nodes directly below us in the argument tree: none for a leaf, its members for an
        aggregate. This is the ONLY structure a generic traversal needs -- every collection the
        analysis derives (axes, attributes, ...) is a FOLD over `CallArgsAnalysis.nodes`, not a
        typed list a node pushes itself onto during construction."""
        return []

    def cpp_axis_names( self ):
        """The axis names our C++ TYPE spells (`_num_vertex`, a batch axis), each needing a
        `DEFINE_AXIS` in the generated source. Empty for a node that names none."""
        return []

    def is_ffi_buffer( self ):
        """Whether we bind an FFI buffer -- a slot in the handler's arg/result list, moved to the
        device and back. True of a bound tensor or count, and of the error buffer; false of a
        value that crosses as an attribute (a `CtShapeVar`, a bare `int`) and of an aggregate (its
        members bind buffers, it does not). The analysis pulls the buffer list by folding the tree
        on this, so nothing pushes itself onto a list as it is built."""
        return False

    def cpp_includes( self ):
        """C++ headers the call must `#include` for us. Empty for most nodes; an aggregate names
        the struct header it is built from. The caller collects these blind -- it never needs to
        know which node brought what, nor whether a header is hand-written or generated."""
        return []

    def cpp_run_parallel_pair( self ):
        """How we enter a `run_parallel` list, as an `<io>, <value>` pair: an io tag saying how
        much the kernel moves (an input in, an output back), then our C++ value. An aggregate
        overrides this to hand its per-member POLICY instead of one blanket tag."""
        return f"{ self.cpp_io_list() }, { self.name }"

    def cpp_io_list( self ):
        """This member's io category, as the tag `run_parallel` speaks (see kernels/IoCategory.h).

        It is what tells `make_available` how much to move: an input is copied to the device but
        not back, an output back but not in. Python already knows it attribute by attribute, so a
        kernel argument is made available MEMBER BY MEMBER -- nothing crosses that has no reason
        to."""
        from .IoCategory import IoCategory
        return { IoCategory.INPUT: "InpList()",
                 IoCategory.OUTPUT: "OutList()",
                 IoCategory.UNBOUND: "UndefList()" }[ self.io_category ]

    def _jax_buffer( self ):
        """How to reach the FFI buffer: an arg is passed by value, a result is pointer-like."""
        return f"{ self.ffi_name }->" if self.io_category.is_output else f"{ self.ffi_name }."

    def jax_data_ptr( self ):
        return f"{ self._jax_buffer() }typed_data()"

    def jax_dim( self, d ):
        """Extent `d` of this buffer, READ FROM THE BUFFER at run time.

        A capacity must never be a literal in the source: it changes from call to call, and
        every distinct value would mean another compilation of the same kernel. XLA already
        carries the extents next to the data, so the generated code just asks for them -- what
        stays in the C++ type is what is genuinely structural (scalar, rank, axis names, and the
        compile-time counts one WANTS in the type)."""
        return f"SI( { self._jax_buffer() }dimensions()[ { d } ] )"
