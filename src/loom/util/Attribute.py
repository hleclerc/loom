from .Parametrized import Parametrized


class Attribute:
    """Base protocol for `@aggregate` field declarations.

    This is the *only* thing `@aggregate` knows about: it operates on classes
    whose fields are `Attribute`s, with no knowledge of any concrete field type
    (`ShapeVar`, `Tensor`, ... are just examples).

    A field annotation is a `Parametrized` (`Attribute[...]`) that acts as the
    class-level *schema*. `get_attribute` calls it once per parent instance to
    build a fresh per-instance `Attribute` that holds that instance's state, kept
    under its field name in the instance `__dict__`. `@aggregate` installs one
    SET-only descriptor per field: `c.field = value` routes to `set`, while
    `c.field` reads the `Attribute` object straight back from `__dict__`.

    The dependency stops there: an `Attribute` knows NOTHING about `@aggregate`.
    Every ctor takes the same two keyword-only channels, and an aggregate is only
    ever one possible caller:

        Attr( *args, template_args = (), template_kwargs = {}, scope = None )

    * `template_args` / `template_kwargs` are the declaration (`Attr[ ... ]`), i.e.
      what a `Parametrized` schema carries;
    * `scope` is what resolves a template arg given as a NAME (see
      `resolve_attribute`) -- an `@aggregate` instance, or `None` outside of one;
    * the POSITIONALS are the concrete type's own business (a value for `Tensor`,
      an affine expression for `Axis`, ...), which is what lets each type be built
      on its own: `Tensor( 17 )`, `Axis( nb_dims )`, `Tensor[ x, y ]( [ ... ] )`.

    `name` is the declared field name, stamped by `get_attribute`. An `Attribute`
    that outlives its aggregate keeps it (a `Tensor` built on a borrowed `Axis`
    still knows that axis is called `num_vertex`, which is what the C++ side needs
    to name it).
    """

    name = None

    def __class_getitem__( cls, item ):
        if isinstance( item, tuple ):
            return Parametrized( cls, *item )
        else:
            return Parametrized( cls, item )

    def set( self, value ):
        raise NotImplementedError

    def get( self ):
        """Per-instance read view exposed by `c.field`; default: the object itself."""
        return self

    @property
    def value( self ):
        """`attr.value` is to a standalone `Attribute` what `c.field` is to an aggregate's:
        the same `get`/`set` pair, reachable without a parent to route through."""
        return self.get()

    @value.setter
    def value( self, value ):
        self.set( value )


def resolve_attribute( entry, scope, expected = Attribute ):
    """Resolve one attribute reference found in a declaration.

    A reference is either the `Attribute` OBJECT itself -- which needs no scope at all, and is
    what lets a `Tensor` be declared on axes borrowed from anywhere (or on none) -- or a NAME,
    which only the scope that declares it can resolve. The scope's whole protocol is
    `scope.get_attribute( name )`; `@aggregate` provides it, nothing else has to.
    """
    if isinstance( entry, Attribute ):
        res = entry
    elif scope is None:
        raise TypeError(
            f"'{ entry }' is a name, and a name can only be resolved inside an @aggregate. "
            f"Outside of one, pass the { expected.__name__ } object itself."
        )
    else:
        res = scope.get_attribute( entry )

    assert isinstance( res, expected ), f"'{ entry }' should be a { expected.__name__ }"
    return res
