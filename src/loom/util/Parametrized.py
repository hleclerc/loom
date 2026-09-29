import inspect


class Parametrized:
    def __init__( self, cls, *args ) -> None:
        self.kwargs = {}
        self.args = []
        self.cls = cls

        for arg in args:
            if isinstance( arg, tuple ) and len( arg ) == 2 and isinstance( arg[ 0 ], str ):
                self.kwargs[ arg[ 0 ] ] = arg[ 1 ]
            elif isinstance( arg, dict ):
                self.kwargs.update( arg )
            else:
                self.args.append( arg )

    def __call__( self, *args, scope = None, **kwargs ):
        # positionals belong to the wrapped type (a value, an expression, ...); the schema only
        # adds the template args/kwargs it carries, plus the scope names are to be resolved in.
        merged_kwargs = { **self.kwargs, **kwargs }
        return self.cls( *args, template_args = self.args, template_kwargs = merged_kwargs, scope = scope )

    def make_CallArg( self, caa, io_category, name, value, ctor_args ):
        # forward the decomposition to the wrapped type, handing it this schema so it can read
        # its template args (axes, dep_axes, ...).
        return self.cls.make_CallArg( caa, io_category, name, value, ctor_args, schema = self )

    def __getattr__( self, name ):
        # any other attribute is a FACTORY on the wrapped type (`Tensor.full`, ...), given this
        # schema's template args/kwargs the same way a plain instantiation would get them -- so
        # `RealTensor[ axis ].full( v )` builds straight from the axis, with no size to repeat.
        attr = getattr( self.cls, name )
        if not callable( attr ):
            return attr

        # A KEYWORD the factory itself declares goes to the FACTORY; anything else is a template
        # kwarg, as on a plain instantiation. Without the split, everything landed in the template
        # kwargs -- so `RealTensor[ x ].random( seed = 7 )` drew a DIFFERENT value each time, in
        # silence, while `full( v )` worked only because its argument is positional. The signature
        # is what separates them; a name that is both is resolved in favour of the factory.
        try:
            declared = set( inspect.signature( attr ).parameters ) - { "template_args", "template_kwargs", "scope" }
        except ( TypeError, ValueError ):
            declared = set()

        # UNE FABRIQUE QUI DÉCLARE `**kwargs` DIT QU'ELLE PREND DES NOMS ARBITRAIRES, et la règle
        # ci-dessus s'inverse alors : ce qui n'est pas un kwarg de TEMPLATE lui revient. Sans ça,
        # `RealTensor[ n, n ].expr( "...", c = centre )` envoyait `c` dans les template kwargs, où
        # il était ignoré -- et le C++ engendré ne compilait pas, faute de `c`. Les kwargs de
        # template sont une liste FERMÉE ( ce que `Tensor.__init__` et `_declared_dtype` lisent ),
        # donc on peut la nommer ici sans risquer d'en oublier un en silence.
        prend_tout = any( p.kind == inspect.Parameter.VAR_KEYWORD
                          for p in inspect.signature( attr ).parameters.values() ) if declared else False

        def pour_la_fabrique( k ):
            return k in declared or ( prend_tout and k not in _TEMPLATE_KWARGS )

        def method( *args, scope = None, **kwargs ):
            own = { k: v for k, v in kwargs.items() if pour_la_fabrique( k ) }
            merged_kwargs = { **self.kwargs, **{ k: v for k, v in kwargs.items() if not pour_la_fabrique( k ) } }
            return attr( *args, **own, template_args = self.args, template_kwargs = merged_kwargs, scope = scope )
        return method


# LES KWARGS DE TEMPLATE, liste fermée : ce que `Tensor.__init__` et `_declared_dtype` lisent.
# Tout le reste, sur une fabrique qui déclare `**kwargs`, appartient à la fabrique.
_TEMPLATE_KWARGS = ( "dtype", "size", "device" )


def constructor_of_subclass_of( klass, parents ):
    if isinstance( klass, Parametrized ):
        return issubclass( klass.cls, parents )
    return inspect.isclass( klass ) and issubclass( klass, parents )
