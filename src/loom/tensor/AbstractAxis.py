from ..util.Attribute import Attribute, resolve_attribute
import numpy
import weakref

from .Affine import Affine, parse_terms


def count_name_for( axis_name ):
    """The name of the count an axis mints for itself when it is given only a name: `nb_<thing>`,
    with a leading `num_` -- the convention for an axis that INDEXES over something -- dropped.
    `num_cell` -> `nb_cell`, `i` -> `nb_i`.

    Deliberately NOT pluralized. English plurals are irregular (`vertex`/`vertices`,
    `index`/`indices`), so guessing one would make the count's name unpredictable from the axis it
    came from -- and being predictable is the only thing this name has to be: it is what shows up
    in an error message about an unresolved count."""
    return "nb_" + ( axis_name[ 4: ] if axis_name.startswith( "num_" ) else axis_name )


class AxisId:
    """WHICH dimension. A pure identity: a name, and nothing else.

    Deliberately sizeless. A size belongs to a WINDOW on a dimension (`Axis`), and a dimension has
    no privileged window -- `num_vertex` and `num_vertex+2` are two views of the same thing, and
    neither is "the" one. Before this existed, identity was represented by whichever `Axis` happened
    to be created first, which meant the identity dragged an extent it had no business carrying.

    Compared by reference, never by name: two dimensions both called `i` are two dimensions."""

    __slots__ = ( "name", )

    def __init__( self, name = None ) -> None:
        self.name = name

    def __repr__( self ) -> str:
        return f"AxisId( { self.name } )"


class AbstractAxis( Attribute ):
    """Common base for `Axis` and `AxisList`.

    Both name a tensor dimension whose extent is an `Affine` expression over
    `ShapeVar`s (see `Affine.py`). The shared part -- factored here -- is the
    *parsing* of that expression, plus the single-variable inversion used to
    solve a `ShapeVar` from an observed size.

    `coeffs` / `offset` remain readable as the two halves of that expression, so
    everything that walks an extent term by term still can.

    The expression is given either as a string (`Axis[ "2 * nb_dims + 1" ]`,
    whose names an aggregate resolves) or, outside of any aggregate, as the
    `ShapeVar` itself (`Axis( nb_dims )`).

    They differ in what is known at declaration time (virtual `register_in`):
    an `Axis` is a single dimension (ragged when one of its `ShapeVar`s has
    rank > 0); an `AxisList` is a *family* of dimensions to be unrolled, whose
    count (`nb_dims`) is unknown when the class is declared.
    """

    def __init__( self, *exprs, template_args = (), template_kwargs = {}, scope = None, name = None ) -> None:
        from .ShapeVar import ShapeVar

        # our WINDOW into the dimension: our index k is its position `lo + k * step`, for k while
        # `lo + k * step` stays short of `hi`. Both bounds are POSITIONS in the dimension's own
        # space, and both are affine -- which is what lets `v[ 2: ]` say "ends where the dimension
        # ends" (`hi = nb_vertices`) instead of freezing the size it happened to have.
        #
        # The extent is therefore DERIVED, never stored: `( hi - lo ) / step`. A declaration is the
        # window `( 0, <the declared extent>, 1 )`, so `Axis[ "nb_vertices" ]` means exactly what it
        # always did.
        self.lo   = Affine.constant( 0 )
        self.hi   = Affine.constant( 0 )
        self.step = 1

        # WHICH dimension we are a view of. Minted here, so nobody ever has to declare one; a
        # window into us (a slice) points at the same one.
        self.axis_id = AxisId( name )

        # Two questions, and conflating them is what makes slicing unsafe:
        #   * `identity`   -- "which dimension is this?" -> SELECTION (`sum( "num_vertex" )`)
        #   * `coordinate` -- "do our positions correspond?" -> ALIGNMENT (an elementwise op)
        # So `v[ 2:4 ] * w[ 2:4 ]` maps elementwise (same window) while `v[ 2:4 ] * w[ 0:2 ]` is
        # refused (index 2 is not index 0), and both remain findable as `num_vertex`.

        # a batch axis sorts BEFORE ordinary ones in the logical layout an elementwise op produces
        # (see `Tensor._binary`). A plain axis is not one; `new_batch_axis` sets this. Physical axis
        # order (e.g. a leading batch on GPU) is a separate, later concern -- it will not touch this.
        self.is_batch = False

        # declared (`Axis[ "nb_dims + 1" ]`) or built directly (`Axis( nb_dims )`): same args,
        # two ways in.
        self._init_axis( list( template_args ) + list( exprs ), scope )

    @property
    def extent( self ):
        """Our extent AS AN EXPRESSION -- `hi - lo` -- or `None` when it is not one: a STEPPED
        window's size is a ceiling division, which no affine can be. Use `numeric_extent` to get
        the number in every case; this is for the cases that need the expression (inverting it)."""
        return ( self.hi - self.lo ) if self.step == 1 else None

    def numeric_extent( self, of ):
        """How many items we hold, `of( symbol )` giving each symbol's value: `( hi - lo ) / step`,
        rounded UP (a stepped window covers a partial last stride) and never negative (an empty
        slice holds nothing, it does not hold a negative amount). `None` while a bound is unknown.

        Follows the shape of the values, so a RAGGED bound gives one extent per segment."""
        lo, hi = self.lo.value( of ), self.hi.value( of )
        if lo is None or hi is None:
            return None
        span = hi - lo
        return numpy.maximum( span if self.step == 1 else -( -span // self.step ), 0 )

    # the two halves of the extent EXPRESSION, kept readable: it is walked term by term in a few
    # places (a ragged display, an unrolled `AxisList`). Exact at step 1, which is every declared
    # axis; a stepped window only reports which symbols its size involves.
    @property
    def coeffs( self ):
        return ( self.hi - self.lo ).coeffs

    @property
    def offset( self ):
        return ( self.hi - self.lo ).offset

    # our NAME is the dimension's: a window is a view of it, not another thing to name.
    @property
    def name( self ):
        return self.axis_id.name

    @name.setter
    def name( self, value ):
        self.axis_id.name = value

    @property
    def identity( self ):
        """WHICH dimension this is -- an `AxisId`, shared with every window into it. What SELECTION
        matches on: `m[ 2:4 ].sum( num_vertex )` still finds its dimension, wherever it starts."""
        return self.axis_id

    @property
    def coordinate( self ):
        """WHERE our indices sit: `( which dimension, origin, step )`, the origin being an affine
        POSITION. What ALIGNMENT matches on -- two dimensions map elementwise only if position k
        means the same item in both. `hi` is deliberately absent: it decides how MANY items we
        hold, not which item each index is, so two windows that start alike but end differently
        align and then fail on the shape, which is the honest error."""
        return ( self.axis_id, self.lo, self.step )

    @property
    def display_name( self ):
        """How this axis reads back: `num_vertex` for the dimension itself, `num_vertex+10` for a
        window into it -- where a window starts is part of what it IS, so it should be visible."""
        if self.name is None or ( self.lo == 0 and self.step == 1 ):
            return self.name
        res = self.name if self.lo == 0 else f"{ self.name }+{ self.lo!r }"
        return res if self.step == 1 else f"{ res }*{ self.step }"

    def windowed( self, key, extent = None ):
        """The axis a slice of us produces: the SAME dimension (still selectable by name), read
        between new bounds. `None` when the slice cannot be expressed as an affine window.

        The bounds stay SYMBOLIC wherever python's own rules allow it, which is the whole point:
        `v[ 2: ]` ends where the dimension ends (`hi` unchanged), `v[ :-1 ]` one short of it
        (`hi - 1`). Only a REVERSAL has to be resolved to numbers -- counting backwards from an
        unknown end is not an affine bound -- and it then needs `extent`, our own item count."""
        step = 1 if key.step is None else int( key.step )

        if step > 0 and self.step > 0:
            lo = self.lo if key.start is None else self._position( key.start )
            hi = self.hi if key.stop  is None else self._position( key.stop )
            return self._window( lo, hi, self.step * step )

        if extent is None:
            return None
        start, stop, step = key.indices( int( extent ) )
        return self._window( self.lo + start * self.step, self.lo + stop * self.step,
                             self.step * step )

    def _position( self, k ):
        """The position our index `k` designates in the dimension's own space. A negative `k`
        counts back from our END, exactly as python does -- and stays affine, since our end is
        itself an affine bound."""
        k = int( k )
        return ( self.lo if k >= 0 else self.hi ) + k * self.step

    def _window( self, lo, hi, step ):
        """A sibling axis over the same dimension, between those bounds. Same name and identity, so
        it is still the `num_vertex` dimension; different coordinate, so it only maps elementwise
        onto the same window."""
        res = type( self )( Affine.constant( 0 ) )
        res.axis_id = self.axis_id                  # the same dimension, not a namesake
        res.lo, res.hi, res.step = lo, hi, step
        res.is_batch = self.is_batch
        return res

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        # An axis lowers to NOTHING: it is a declaration, not data. Its extent is already baked
        # into the shape of every tensor that uses it, and its name is registered by those
        # tensors (a tensor may borrow an axis from an object that is not even an argument).
        return None

    def cpp_axis_names( self ):
        """The axis name(s) this declaration needs `DEFINE_AXIS`'d in C++. An aggregate collects
        these so every axis it declares is spelled in its header -- even one no tensor of the call
        references (`num_edge`): a body may still name it, and the C++ type must exist for it."""
        return [ self.name ]

    def cpp_dim_names( self, index ):
        """The C++ name(s) for the ARRAY dimension(s) this axis expands into: the NAME analogue of
        `max_list` (which does the same for extents). One entry for a plain `Axis`; several for an
        unrolled `AxisList`. Keeping the unrolling HERE (and in the overrides) lets a caller merely
        concatenate over a tensor's axes -- it needs no notion of how many `AxisList`s there are or
        how wide each unrolls. `index` is the axis' position, used only for the nameless fallback."""
        return [ self.name or f"a{ index }" ]

    @staticmethod
    def cpp_shared_header( name ):
        """The shared header that DECLARES the axis `name`: `DEFINE_AXIS` (behind AxisNames.h),
        which spells the type `_name` a tensor references and the `name` object a body indexes
        with. Returns its include path. It lives here because the C++ facet of an axis is the
        axis's business, not the call's -- the call only asks for it by name."""
        from ..compilation.generated_headers import shared_header
        content = ( "#pragma once\n\n"
                    '#include <loom/support/containers/AxisNames.h>\n\n'
                    f"DEFINE_AXIS( { name } );\n" )
        return shared_header( f"sdot/generated/axes/{ name }.h", content )

    # ---- shared affine parser ----
    @staticmethod
    def parse_affine( expr ):
        """Pure TEXT parse of `"2 * nb_dims + 3 * nb_xs + 1"` into `( { var_name: coeff }, offset )`
        -- names stay strings, so this is usable with no scope to resolve them in."""
        return parse_terms( expr )

    def _parse_expr( self, expr, scope ):
        """Instance-side parse: fill `coeffs` (each name resolved to its `ShapeVar` in `scope`)
        and `offset`. A `ShapeVar` handed over directly is the degenerate expression `1 * var`,
        and needs no scope; so does a plain INTEGER extent, which mints its own."""
        from .ShapeVar import ShapeVar
        if isinstance( expr, Affine ):
            self.hi = self.hi + expr
            return

        if isinstance( expr, ShapeVar ):
            self._add_symbol( expr, 1, scope )
            return

        if isinstance( expr, str ) and scope is None and expr.isidentifier():
            # Standalone there is no scope a NAME could be resolved in -- so a bare name is not a
            # REFERENCE to a count, it IS the axis's name, and the count is minted for it. This is
            # the short spelling for ad-hoc work: `RealTensor[ Axis( "num_cell" ), Axis( "dim" ) ]`,
            # with the extent then solved from whatever tensor uses the axis.
            #
            # Inside an aggregate the string keeps its usual meaning (a count declared as a field,
            # possibly inside an affine expression) -- that is what the scope is for.
            from .ShapeVar import ShapeVar
            shape_var = ShapeVar()
            shape_var.name = count_name_for( expr )
            if self.name is None:
                self.name = expr
            self._add_symbol( shape_var, 1, scope )
            return

        if isinstance( expr, int ):
            # `Axis( 3 )` -- a literal extent, the standalone case. It still gets a `ShapeVar` of
            # its own (prescribed to 3) rather than becoming a bare offset: an extent that is a
            # count, even a known one, is what everything else inverts, shares and displays. The
            # short spelling is a shortcut, not a second kind of axis.
            self._add_symbol( ShapeVar( expr ), 1, scope )
            return

        names, offset = parse_terms( expr )
        self.hi = self.hi + offset
        for var_name, coeff in names.items():
            self._add_symbol( var_name, coeff, scope )

    def _add_symbol( self, var, coeff, scope ):
        """Add `coeff * var` to our extent, `var` being a `ShapeVar` or a NAME to resolve."""
        from .ShapeVar import ShapeVar
        shape_var = resolve_attribute( var, scope, ShapeVar )
        self.hi = self.hi + Affine.of( shape_var ) * coeff

    def solve_single( self, shape_var, size ):
        """Invert the single-variable affine `size = coeff * shape_var + offset`, i.e.
        `( size - offset ) // coeff`, following the shape of `size` (a scalar, or one per segment).
        `None` when `shape_var` is not our sole variable -- a multi-variable solve is not
        attempted, and an extent that depends on a POSITION cannot be inverted at all (see
        `Affine.solve`). A STEPPED window has no affine extent at all, so it never inverts -- a
        sampled view genuinely does not determine what it was sampled from."""
        extent = self.extent
        return None if extent is None else extent.solve( shape_var, size )

    def set( self, value ):
        raise RuntimeError( "An axis cannot be set" )

    # ---- extents (virtual) ----
    def max_list( self ):
        """The member's extents as a list to be concatenated into a tensor shape:
        one entry for an `Axis`, `nb_dims` entries for an unrolled `AxisList`."""
        raise NotImplementedError

    def array_dims( self, tensor ):
        """How many ARRAY dimensions this axis occupies on `tensor`: one for a plain `Axis`, its
        unroll width for an `AxisList` (which overrides). The count lives on the AXIS, so a tensor
        never has to do `ndim - n_plain` arithmetic itself, nor know how many lists it holds."""
        return 1

    # ---- usage registration (virtual) ----
    def register_in( self, tensor ):
        """Record on our `ShapeVar`s that `tensor` (which declares us among its axes) constrains them,
        so they can be SOLVED (pulled) from it on demand -- from its logical sizes for the count, from
        its buffer for the allocated capacity. We find our OWN position in `tensor` (`_dim_index`) so
        the caller passes no index. A plain axis is one dimension; an `AxisList` overrides (it spans
        several)."""
        self._register_dense( tensor )

    def _register_dense( self, tensor ):
        """One declared axis <-> its own array dimension. Two resolvers per ShapeVar, both mapping our
        axis to its array dimension (`_spec_dims`, which accounts for an unrolled sibling): `logical`
        inverts our affine on the LOGICAL size there (`reference_shape.sizes(...)`, a 0-d scalar for a dense
        axis or a per-segment array for a ragged one -- unpadded); `capacity` on the ALLOCATED buffer
        size (`allocated_sizes` at that dimension). Our position is resolved at PULL time (`_dim_index`),
        by then `tensor.axes` is complete.

        The axis and the ShapeVar are captured WEAKLY, and that is not a detail: a resolver is
        stored ON the ShapeVar (`add_usage`), so capturing them strongly closes the ring
        `Axis -> coeffs -> ShapeVar -> usages -> resolver -> Axis` -- and a ring is freed only by
        the cyclic collector, never by refcounting. The tensor weakref right below is there for the
        same reason; it just did not go far enough.

        Weak is SAFE by construction, not by luck: `_pull` only ever calls a resolver whose tensor
        is still alive, a live tensor holds its axes, and an axis holds its ShapeVars. So neither
        referent can be gone while the entry is usable -- and if one somehow is, the resolver
        answers `None`, which `_pull` already means as "cannot say"."""
        axis_ref = weakref.ref( self )
        for shape_var in self.coeffs:
            sv_ref = weakref.ref( shape_var )
            def logical( t, axis_ref = axis_ref, sv_ref = sv_ref ):
                axis, shape_var = axis_ref(), sv_ref()
                if axis is None or shape_var is None or t.reference_shape is None:
                    return None
                dim = t._spec_dims()[ t._dim_index( axis ) ]
                return axis.solve_single( shape_var, t.reference_shape.sizes( dim ) )
            def capacity( t, axis_ref = axis_ref, sv_ref = sv_ref ):
                axis, shape_var = axis_ref(), sv_ref()
                sizes = t.allocated_sizes
                if axis is None or shape_var is None or sizes is None:
                    return None
                dim = t._spec_dims()[ t._dim_index( axis ) ]
                return axis.solve_single( shape_var, sizes[ dim ] )
            shape_var.add_usage( tensor, logical, capacity )

    # ---- extents (virtual) ----
    def capacity_list( self, capacity_of ):
        """The member's extents for an allocation, as a list to be concatenated into a tensor
        shape -- `capacity_of( shape_var )` being what the CALL decided to allocate for each of
        our variables (a capacity is never our own state, see `ShapeVar`)."""
        raise NotImplementedError

    # ---- virtual ----
    def _init_axis( self, args, scope ):
        raise NotImplementedError
