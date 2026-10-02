# from typing_extensions import Sequence, overload
from ..util.Attribute import Attribute, resolve_attribute
from typing import TYPE_CHECKING, Iterator
from ..drivers.driver import driver
from numpy._typing import ArrayLike
import weakref
import numpy


class ShapeVar( Attribute ):
    """An integer (possibly rank > 0) variable used to build shapes.

    Rank 0 is a plain scalar count; rank > 0 holds the per-row/per-segment counts
    of a RAGGED structure (`dep_axes` records which axes it varies along).

    Axis extents are affine *expressions* over `ShapeVar`s (e.g. `nb_dims + 1`).
    A `ShapeVar` is either prescribed (`c.nb_dims = 1222`) or solved from the
    shapes observed on the declared tensors that use it; a prescribed value wins.
    `c.nb_dims` reads back the value.

    Sharing: pass a `ShapeVar` to several constructors (`Cell( nb_dims = n )`);
    they then reference the same object, so the value is solved from the union of
    their tensors.

    `value` reads the count back on the HOST (a `ShapeArray`), because sizing is what a count
    is for; `as_tensor()` is the way to ask for it as a device value instead.

    The value a `ShapeVar` holds is a COUNT: how many items are used. A kernel
    reads it and/or writes it, so after a call it lives in a DEVICE buffer -- under
    `jit`, Python does not know it, and it can therefore never size anything.

    What sizes a buffer is a CAPACITY, and a capacity is deliberately NOT state
    kept here: it is a decision about ONE allocation, so it belongs to the call
    that allocates (`driver.call( ..., capacities = { "cell.nb_vertices": 8 } )`).
    Storing it on the object would open a window where the object contradicts
    itself -- a capacity of 64 next to a buffer of 8. What we can offer is what we
    KNOW: `allocated_capacity` (read off the buffers our tensors already have) and
    `static_count` (a count Python actually holds). See `CallArgsAnalysis`.
    """

    # au niveau de la CLASSE, comme `Attribute.name` : tout `ShapeVar` n'est pas bâti par notre
    # `__init__` ( une sous-classe peut avoir le sien, un clone peut reconstruire l'objet ), et un
    # compte sans formule est le cas NORMAL -- il ne doit pas dépendre d'un champ posé par un ctor.
    _formula = None

    if TYPE_CHECKING:
        def __set__( self, obj, value: int ) -> None: ...
        def __iter__( self ) -> Iterator: ...

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_ShapeVar import CallArg_ShapeVar
        return CallArg_ShapeVar( caa, path, name, inst )

    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        from .AbstractAxis import AbstractAxis

        # (weakref(tensor), logical, capacity): two resolvers per using tensor. `logical(t)` inverts
        # our affine on the tensor's LOGICAL sizes (`t.reference_shape`, the unpadded truth); `capacity(t)`
        # on its ALLOCATED buffer (padded). Either returns None when it cannot yet. Weak
        # refs so a shared ShapeVar does not keep dropped instances' tensors alive.
        self.usages = []
        self._compact_usages_at = 8   # see `add_usage`
        self.dep_axes = [ resolve_attribute( d, scope, AbstractAxis ) for d in template_args ]
        # the axes the AGGREGATE we belong to is batched over (`Aggregate.apply_batch_axes`), and
        # nothing to do with `dep_axes`: those are the segments of a ragged structure, these are
        # items co-iterated by the kernel. They matter only for a count a KERNEL WRITES, which is
        # then one count PER ITEM -- a host-known count is uniform over the batch by construction,
        # and stays the single value it is (see `CallArg_ShapeVar`).
        self.batch_axes = []

        self.prescribed_value = None
        self._count = None     # count produced by a kernel: a driver tensor, possibly traced
        # UNE FORMULE QUELCONQUE ( `Axis[ "nb_dim * ( nb_dim + 1 ) / 2" ]` ) : `( texte,
        # { nom: ShapeVar } )`, evaluee a la demande. Voir `set_formula`.
        self._formula = None

        if value is not None:
            self.set( value )

    def add_usage( self, tensor, logical, capacity ):
        # DERIVED tensors register too (see `Tensor._wrap_axes`), and a loop makes a great many of
        # them -- so dead entries are dropped as they accumulate instead of piling up for every
        # `_pull` to walk. Amortized: compact only once the list has doubled since the last time.
        if len( self.usages ) >= self._compact_usages_at:
            self.usages = [ u for u in self.usages if u[ 0 ]() is not None ]
            self._compact_usages_at = max( 8, 2 * len( self.usages ) )
        # appended LAST, so a declared witness is still the first `_pull` consults.
        self.usages.append( ( weakref.ref( tensor ), logical, capacity ) )

    def set_count( self, value ):
        """Rebind the count to what a kernel produced (a driver tensor). Nothing else moves:
        the buffers keep the size they were allocated with."""
        self._count = value

    def accept_batch_axes( self, batch_axes ):
        """Told which axes the aggregate holding us is batched over. See `batch_axes`."""
        self.batch_axes = list( batch_axes )

    def is_kernel_written( self ):
        """Whether a kernel is what put the count there -- as opposed to the host prescribing it
        or solving it off a tensor's shape. This is what decides whether the count is PER BATCH
        ITEM: only a kernel can make two items disagree."""
        return self._count is not None

    def _axis_names_for( self, raw ):
        """The names of the dimensions `raw` actually has: the ragged ones (`dep_axes`), preceded
        by the batch ones when the count really did materialize one per item."""
        names = [ ax.name for ax in self.batch_axes ] + [ ax.name for ax in self.dep_axes ]
        return names[ len( names ) - numpy.ndim( raw ) : ]

    def accept_capacity( self, capacity ):
        """Called when a call is given a capacity for us: our chance to refuse one that would
        contradict what we are (see `CtShapeVar`). Nothing is stored -- a capacity belongs to
        the call that allocates."""
        pass

    def set( self, value ):
        if isinstance( value, ShapeVar ):
            value = value.value
        self.prescribed_value = numpy.array( value, dtype = int )

    def set_formula( self, expr, symbols ):
        """Notre compte est une FORMULE d'autres comptes : `"nb_dim * ( nb_dim + 1 ) / 2"`, avec
        `symbols` qui dit quel `ShapeVar` chaque nom désigne.

        POURQUOI CE N'EST PAS UNE AFFINE. Une étendue d'axe est normalement affine ( `nb_dims + 1` )
        parce que c'est elle qu'on INVERSE : voir une taille de 4 sur un tampon, et en déduire que
        `nb_dims` vaut 3. Une formule quelconque ne s'inverse pas -- et n'a pas à l'être, parce
        qu'elle ne décrit pas un compte inconnu : elle le CALCULE à partir de comptes déjà connus.
        Les deux coexistent donc sans se gêner, et le partage est net :

            affine    ce qu'on RÉSOUT  ( `Axis[ "nb_rows + 1" ]` )
            formule   ce qu'on CALCULE ( `Axis[ "nb_dim * ( nb_dim + 1 ) / 2" ]` )

        Évaluée À LA DEMANDE, et c'est nécessaire : dans un agrégat les champs sont construits dans
        l'ordre des annotations, donc l'axe est bâti AVANT que `Splats( nb_dim = 2 )` ait prescrit
        quoi que ce soit. Tant qu'un symbole n'a pas de compte HÔTE ( `static_count` ), on rend
        `None` -- exactement comme une affine non résolue."""
        self._formula = ( str( expr ), dict( symbols ) )

    def _formula_value( self ):
        """La formule évaluée, `None` tant qu'un de ses symboles n'a pas de compte hôte.

        PAR SEGMENT SI SES SYMBOLES LE SONT. Les symboles entrent TELS QUELS ( `static_raw`, et non
        `static_count` qui en prend le maximum ), donc `( nb_pixels + 15 ) // 16` sur un `nb_pixels`
        qui vaut `[ 256, 300 ]` rend `[ 16, 19 ]` : une étendue par dimension, ce qu'une `AxisList`
        déroule. C'est l'arithmétique de numpy qui travaille, donc la FORME suit les symboles sans
        qu'on ait à la dire -- un compte simple reste un scalaire.

        Le résultat doit être ENTIER : `2 * 3 / 2` vaut `3.0`, ce qui est bien le compte 3 ; `5 / 2`
        ne vaut aucun compte, et c'est une erreur de déclaration, pas un arrondi à faire en
        silence."""
        if self._formula is None:
            return None
        expr, symbols = self._formula

        env = {}
        for nom, shape_var in symbols.items():
            compte = shape_var.static_raw()
            if compte is None:
                return None
            env[ nom ] = numpy.asarray( compte )

        res = numpy.asarray( eval( compile( expr, f"<taille d'axe { self.name }>", "eval" ),
                                   { "__builtins__": {} }, env ) )
        entier = numpy.rint( res ).astype( int )
        if not numpy.all( res == entier ):
            raise ValueError(
                f"la taille d'axe '{ expr }' vaut { res }, qui n'est pas un entier -- un compte en "
                f"est un ( avec { ', '.join( f'{ n } = { v }' for n, v in env.items() ) } )" )
        return entier

    def _pull( self, kind ):
        """Solve our count from the tensors that use us: the FIRST usage able to invert one of its
        sizes, `None` if none can. `kind` picks which sizes -- the `"logical"` ones (`t.reference_shape`,
        the count) or the `"capacity"` ones (the allocation a chained call must reuse).
        First-that-answers, so a witness carrying the RIGHT rank (a ragged tensor holding per-segment
        counts) is reached the same way as a scalar one -- both invert this ShapeVar's own affine."""
        for tensor_ref, logical, capacity in self.usages:
            tensor = tensor_ref()
            if tensor is None:
                continue
            solved = ( logical if kind == "logical" else capacity )( tensor )
            if solved is not None:
                return numpy.asarray( solved )
        return None

    @property
    def raw( self ) -> ArrayLike:
        """The raw COUNT as a backend / numpy array, or `None` while UNRESOLVED -- the same role
        `Tensor.raw` plays: the backend buffer behind the nice object. A kernel-written count wins,
        being the freshest truth (and it is then a DEVICE value, never size a buffer with it); what
        the HOST holds otherwise, in the order `static_raw` says.

        Users read `value` (a host `ShapeArray`); `raw` is the escape hatch to the backend array
        (what the shape math and the FFI read, without wrapping) -- and it is where a count that
        only lives on the device is still reachable, `as_tensor()` being its tidy form."""
        if self._count is not None:
            return self._count

        brut = self.static_raw()
        return None if brut is None else numpy.asarray( brut )

    @property
    def value( self ):
        """The count on the HOST, as a `ShapeArray` -- rank 0 for a plain scalar count, rank > 0
        for a ragged one (its `dep_axes` name the dimensions). `None` while unresolved.

        Host, not device, and that is the point: a count is what SIZES things (an allocation, a
        `range`, an XLA shape), and only a value Python actually holds can do that. Reading one
        that lives on the device is refused HERE, with a message saying so, rather than handed
        back as a tracer that fails much later in whatever tried to size something with it.

        When the device value IS what you want -- to compute with it in a kernel or in backend
        algebra -- ask for it: `as_tensor()`."""
        raw = self.raw
        if raw is None:
            return None
        from .ShapeArray import ShapeArray
        return ShapeArray( raw, names = self._axis_names_for( raw ) )

    @value.setter
    def value( self, value ):
        self.set( value )

    def as_tensor( self ):
        """The count as a DEVICE value (an `IntTensor`), for the cases that genuinely compute with
        it on the backend rather than size something with it. `None` while unresolved.

        No dtype is claimed: a count buffer knows what it is, and it is NOT uniformly the driver's
        itype -- a kernel-written one is `int32` by design (see `CallArg_ShapeVar`), a prescribed
        one is numpy's int. `wrap` reads it off the buffer instead of labelling it."""
        raw = self.raw
        if raw is None:
            return None
        from .Tensor import Tensor
        return Tensor.wrap( raw, names = self._axis_names_for( raw ) )

    # UN COMPTE EST UN NOMBRE, et l'hôte doit pouvoir s'en servir comme tel. Sans ça, tout site qui
    # dimensionne quelque chose écrivait `int( splats.nb_splats.value )` -- un détour par la lecture
    # hôte, puis une conversion, pour un entier que l'objet connaît. `__index__` est ce qui le rend
    # utilisable partout où une taille est attendue : `range( n )`, une borne de tranche, `[ 0 ] * n`.
    #
    # C'est `value` qui travaille, donc le refus est le sien : un compte qui vit sur le DEVICE
    # ( écrit par un noyau, lu sous `jit` ) dit pourquoi il ne peut pas être lu ici, au lieu de
    # rendre un tracer qui échouera quarante cadres plus loin.
    def __int__( self ) -> int:
        valeur = self.value
        if valeur is None:
            raise ValueError( f"le compte '{ self.name }' n'est pas résolu : rien ne le prescrit et "
                              f"aucun tenseur déclaré dessus n'a encore de forme" )
        return int( valeur )

    def __index__( self ) -> int:
        return self.__int__()

    @property
    def max( self ) -> int:
        return driver.max( self.raw )

    def allocated_capacity( self ):
        """The capacity our tensors were ALLOCATED with, read off their buffers -- a fact, not
        a decision, so a chained call needs no restating of a capacity already materialized.
        `None` when we have no allocated tensor to read it from."""
        solved = self._pull( "capacity" )
        return int( numpy.max( solved ) ) if solved is not None else None

    def static_raw( self ):
        """Our count when PYTHON holds it, AS IT IS: a scalar for a plain count, a VECTOR for one
        count per segment (`dep_axes` -- an image side per dimension). `None` when it only lives on
        the device, which is the whole meaning of "Python holds it".

        Three sources, in this order: what a user PRESCRIBED, what a FORMULA over other counts
        CALCULATES (`set_formula` -- it computes, so it comes before anything that tries to deduce),
        then what the LOGICAL sizes of a using tensor solve to (`_pull`).

        This is the raw count; `static_count` is its maximum. The split is what each is FOR: a size
        to allocate wants one number (the max), an extent to compute with wants the count itself."""
        if self.prescribed_value is not None:
            return self.prescribed_value

        calcule = self._formula_value()
        if calcule is not None:
            return calcule

        return self._pull( "logical" )

    def static_count( self ):
        """Our count when PYTHON holds it, as ONE number -- the MAXIMUM when there is one per
        segment, because that is what sizes things. See `static_raw`."""
        brut = self.static_raw()
        return None if brut is None else int( numpy.max( brut ) )
