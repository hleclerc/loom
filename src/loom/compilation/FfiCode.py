from pathlib import Path


class AbstractFfiCode:
    """Le C++ qu'un appel exécute, derrière deux questions : `code_for( call_args_analysis )`, les
    instructions à l'intérieur du handler, et `preamble_for( ... )`, ce qui doit exister au niveau
    du namespace avant lui (un foncteur).

    Il n'y a pas de direction dans ces deux questions : un BACKWARD est un noyau à part entière,
    que l'appel obtient par `for_backward()` et lance ensuite comme n'importe quel forward. Le
    reste du pipeline ne connaît donc qu'un seul sens."""

    def code_for( self, call_args_analysis ) -> str:
        raise NotImplementedError

    def preamble_for( self, call_args_analysis ) -> str:
        """C++ de niveau namespace émis avant le handler. Rien par défaut (un corps verbatim
        apporte le sien)."""
        return ""

    @property
    def wants_allocator( self ) -> bool:
        """Si le handler doit recevoir l'allocateur d'XLA (`Scratch`). Faux par défaut : c'est la
        SOURCE qui est hachée pour nommer un noyau, donc lier ce contexte sans qu'on le demande
        recompilerait tout le dépôt pour une capacité que personne n'utilise."""
        return False

    @property
    def is_handler( self ) -> bool:
        """Si le corps EST le handler -- donc si c'est LUI qui lance. Faux par défaut."""
        return False


class FfiCode( AbstractFfiCode ):
    """UN NOYAU : le C++ que l'appel exécute, et ce qu'il faut pour le compiler.

    = Ce que loom écrit, et ce qu'il n'écrit pas

    `code` est le corps DU HANDLER, recopié tel quel. Loom écrit tout ce qui l'entoure -- et c'est
    tout ce qu'il a à écrire, parce que c'est là qu'est la douleur : l'enrobage FFI, la liaison des
    tampons aux vues, la construction des agrégats, le semis des sorties, l'adjoint côté Jax. Les
    interfaces de Jax et de Torch sont lourdes ET différentes ; c'est ça qu'on ne veut pas écrire
    deux fois.

    Dans le corps, loom met à disposition `queue` et les arguments de l'appel sous leurs noms
    Python. Ce qu'on en fait ne le regarde pas :

        FfiCode(
            include_roots = [ ma_racine ],
            includes = [ "diffusion/noyaux.h" ],
            code = "diffusion::pas( queue, grille, coef, suivant );",
        )

    Le parcours, le choix du parallélisme, la géométrie de lancement vivent alors dans NOTRE C++ --
    un fichier ordinaire, qui se compile et se teste sans loom, et qu'on remplace par du Kokkos, du
    SYCL, de l'OpenMP ou une simple boucle sans toucher à Python. `run_parallel` est l'outil que
    loom propose, pas une obligation qu'il impose.

    = Un noyau ne se contient pas lui-même

    L'adjoint n'est pas un champ de l'aller : c'est un autre noyau, et c'est l'APPEL qui en prend
    plusieurs.

        driver.call(
            FfiCode( code = "monpaquet::avant( queue, ... );" ),
            FfiCode( code = "monpaquet::arriere( queue, ... );" ),   # optionnel
            name = "un_pas",
            ... )

    Il tourne sur d'autres tampons (les cotangentes), fait un autre travail, et n'a aucune raison de
    vouloir la même géométrie que l'aller. Le NOM est sur l'appel : il identifie le couple, préfixe
    la cible compilée et groupe le journal des compilations.

    = Où vit le C++

    `include_roots` : les racines `-I` de CE noyau. Un noyau sait où sont ses en-têtes -- ça n'a pas
    à être une incantation de module (`compilation.register_include_root`) prononcée avant tout le
    reste et sans rapport visible avec lui.

    `includes` : les en-têtes dont le corps a besoin, émis après ceux du runtime. `sources` : les
    unités C++ qu'il LIE (`"sdot/x.cpp"` ou `( "sdot/x.cpp", { "DEF": "1" } )`), compilées une fois
    par (source, defines, compilateur) et partagées par tous les noyaux qui les nomment.

    `prologue` : une instruction C++ émise avant le corps, dans la même portée. Vestige de l'époque
    où le corps était par item et ne pouvait pas exprimer une pré-passe ; un corps qui est le
    handler n'en a plus besoin.

    `allocator = True` fait lier l'allocateur d'XLA, donc `args.allocator.view<T>( n )` dans le corps : de la
    mémoire dimensionnée à l'exécution, à une taille que seul le noyau connaît, sous `jit` comme en
    eager. Voir `support/kernels/Scratch.h` et `tests/test_scratch_gpu.py`.

    = LE SUCRE : un corps par item ( `FfiCode.per_item` )

    Quand le parallélisme du noyau EST celui de l'appel -- un item par multi-indice de batch, ce
    qu'un `vmap` fabrique -- loom peut écrire le foncteur ET son lancement, et le corps n'est plus
    que ce qui se passe pour un item. Trois noms sont alors réservés : `batch_index`,
    `flat_index`, `thread_index`, `nb_threads` (ou, en coopératif, `flat_index`, `group_index`,
    `local_index`, `local_size`,
    `group`, `local_scratch`, `sub_group`). La géométrie se déclare en corps de méthode du foncteur
    engendré : `max_nb_threads`, `group_size`, `local_mem_elems` -- `group_size` exige
    `local_mem_elems`, sans quoi le chemin coopératif serait silencieusement ignoré.

    Le piège, et c'est pour ça que ce n'est plus le défaut : le domaine engendré est
    `global_batch_indices` et RIEN D'AUTRE. Un noyau dont le parallélisme n'est pas un axe de `vmap`
    -- une grille cartésienne, parcourue en (j, i) -- devait s'en déguiser un, matérialiser un rang
    plat et le redécouper en C++. Voir `examples/diffusion/README.md`, friction 3.
    """

    def __init__( self, code = "", prologue = "", includes = (), sources = (), include_roots = (),
                  max_nb_threads = "", group_size = "", local_mem_elems = "",
                  allocator = False, _scaffold = False ) -> None:
        if group_size and not local_mem_elems:
            raise ValueError( "FfiCode: `group_size` without `local_mem_elems` -- `run_parallel` "
                              "only takes the cooperative path when BOTH hooks exist, so this "
                              "would be silently ignored" )
        if local_mem_elems and not group_size:
            raise ValueError( "FfiCode: `local_mem_elems` without `group_size` -- there is no "
                              "work-group to share it" )
        if ( max_nb_threads or group_size or local_mem_elems ) and not _scaffold:
            raise ValueError( "FfiCode: une géométrie de lancement (`max_nb_threads`, "
                              "`group_size`, `local_mem_elems`) est faite de MÉTHODES du foncteur "
                              "engendré -- un corps qui lance lui-même n'en a pas. Passez-la à "
                              "`run_parallel` là où vous lancez, ou utilisez `FfiCode.per_item`." )

        # OÙ VIT LE C++ DE CE NOYAU. C'était un appel de module à part
        # (`compilation.register_include_root( ... )`), donc une incantation avant toute chose et
        # sans rapport visible avec le noyau qui en a besoin. Un noyau sait où sont ses en-têtes :
        # il le dit ici.
        from . import register_include_root
        if not include_roots and not _scaffold:
            # LA RACINE PAR DÉFAUT : le répertoire du `.py` qui construit ce noyau. Une règle, zéro
            # exception -- `#include "mon_noyau.h"` marche pour un fichier posé à côté, ce qui est
            # tout ce qu'un tutoriel doit expliquer. Une disposition différente se dit
            # explicitement.
            import inspect
            for cadre in inspect.stack()[ 1: ]:
                chemin = Path( cadre.filename )
                if chemin.is_file() and "loom/compilation" not in chemin.as_posix():
                    include_roots = [ chemin.resolve().parent ]
                    break
        for root in include_roots:
            register_include_root( root )

        self.code = code
        self.prologue = prologue
        self.sources = tuple( sources )
        self.includes = tuple( includes )
        self.include_roots = tuple( include_roots )
        self.allocator = bool( allocator )
        self._scaffold = _scaffold

        # les corps des hooks que `run_parallel` détecte sur le foncteur -- du C++, pas des
        # expressions évaluées dans une portée que l'appelant ne voit pas. L'ordre compte :
        # `local_mem_elems` peut appeler `group_size`.
        self.hooks = { hook: body for hook, body in
                       ( ( "max_nb_threads", max_nb_threads ), ( "group_size", group_size ),
                         ( "local_mem_elems", local_mem_elems ) ) if body }

    @classmethod
    def handler( cls, code = "", **kwargs ):
        """Redondant : c'est ce que fait `FfiCode` tout court depuis qu'un corps qui lance lui-même
        est la forme NORMALE. Gardé parce que des appels existants le nomment."""
        return cls( code, **kwargs )

    @classmethod
    def per_item( cls, code = "", **kwargs ):
        """LE SUCRE : un corps par ITEM, et loom engendre pour vous le foncteur ET son lancement.

        C'est commode quand le parallélisme du noyau EST celui de l'appel -- un item par
        multi-indice de batch, ce qu'un `vmap` fabrique. Ça ne l'est pas sinon : le domaine engendré
        est `global_batch_indices` et rien d'autre, donc un noyau qui veut parcourir une grille en
        (j, i) devait se déguiser un axe de batch plat ( voir `examples/diffusion/README.md`,
        friction 3 ). Dans ce cas, écrivez le lancement -- c'est `FfiCode` tout court."""
        return cls( code, _scaffold = True, **kwargs )

    # ---- ce que l'appel demande ( il passe le nom du foncteur, qu'il est seul à connaître ) ----

    @property
    def cooperative( self ):
        return "group_size" in self.hooks

    @property
    def wants_allocator( self ):
        return self.allocator

    @property
    def is_handler( self ):
        """Le corps EST le handler : c'est LUI qui lance. Voir `FfiCode.handler`."""
        return not self._scaffold

    def _params( self, names ):
        """Les paramètres de l'`operator()` : les réservés, puis un par argument de l'appel --
        chacun avec son propre paramètre de template, puisque leur type côté kernel est décidé en
        C++."""
        # `flat_index` : LE RANG PLAT DE L'ITEM dans le domaine parcouru -- « qui suis-je ? ». Il
        # vient de la variable de boucle du lancement ( voir `CpuQueue::call` ), donc il ne coûte
        # rien ; avant, un noyau qui en avait besoin se fabriquait un agrégat-prétexte portant un
        # `iota` ( `examples/splats::Rangs` ), soit un tenseur entier écrit puis lu pour un nombre
        # que la boucle connaissait déjà.
        if self.cooperative:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "SI", "flat_index" ), ( "int", "group_index" ),
                         ( "int", "local_index" ), ( "int", "local_size" ), ( "Group", "group" ),
                         ( "LocalScratch", "local_scratch" ), ( "SubGroup", "sub_group" ) ]
        else:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "SI", "flat_index" ),
                         ( "int", "thread_index" ), ( "int", "nb_threads" ) ]
        params = reserved + [ ( f"T_{ n }", n ) for n in names ]
        tparams = [ t for t, _ in params if t not in ( "int", "SI" ) ]
        return tparams, params

    def _hook_methods( self, names ):
        """Les hooks de lancement, en méthodes du foncteur. `run_parallel` les appelle avec les
        arguments de l'appel (`func.max_nb_threads( args... )`, voir `run_parallel.cxx`), donc ils
        prennent la même liste que l'`operator()`, sans les noms réservés -- et ils tournent CÔTÉ
        HÔTE, avant le lancement, donc pas de `HD`."""
        if not self.hooks:
            return ""
        tparams = ", ".join( f"class T_{ n }" for n in names )
        params = ", ".join( f"T_{ n } { n }" for n in names )
        res = ""
        for hook, body in self.hooks.items():
            res += ( f"    template<{ tparams }>\n"
                     f"    int { hook }( { params } ) const {{\n"
                     f"        { body }\n"
                     f"    }}\n" )
        return res

    def preamble_for( self, call_args_analysis, functor ) -> str:
        if not self._scaffold:
            # TOUT le C++ de l'usager, VERBATIM, au niveau du namespace : ses `#include`, ses
            # foncteurs, sa fonction `kernel`. Loom n'y touche pas -- y compris pour le namespace
            # anonyme, que l'usager écrit lui-même : sinon ses `#include` se retrouveraient DANS ce
            # namespace, et loom avec eux.
            #
            # Ce namespace n'est pas décoratif : `compilation/catalogue.py` compile chaque noyau
            # dans son propre objet puis les lie dans UNE bibliothèque, donc deux `kernel` à nom
            # fixe sans liaison interne seraient une violation d'ODR.
            return self.code
        names = list( call_args_analysis.args )
        tparams, params = self._params( names )
        return ( f"struct { functor } {{\n"
                 f"{ self._hook_methods( names ) }"
                 f"    template<{ ', '.join( 'class ' + t for t in tparams ) }>\n"
                 f"    HD void operator()( { ', '.join( f'{ t } { n }' for t, n in params ) } ) const {{\n"
                 f"        { self.code }\n"
                 f"    }}\n"
                 f"}};\n" )

    def code_for( self, call_args_analysis, functor ) -> str:
        # émis AVANT le lancement, dans la portée du handler (voir la docstring) -- une pré-passe
        # unique, pas un morceau du foncteur.
        prologue = ( self.prologue + "\n" ) if self.prologue else ""

        if not self._scaffold:
            # L'APPEL, et il est fixe : `kernel( queue, batch_axes, args )`. Trois choses, et la
            # deuxième est ce qui rend cette forme aussi capable que l'échafaudage -- les axes de
            # batch de l'appel sont une VALEUR que le noyau compose avec les siens
            # (`batch_axes + args.<tenseur>.domain()`), au lieu d'un domaine qu'on lui impose.
            return prologue + "kernel( queue, global_batch_indices, args );"

        names = list( call_args_analysis.args )
        mapped = ", ".join( call_args_analysis.args[ n ].cpp_run_parallel_pair() for n in names )
        return ( f"{ prologue }"
                 "run_parallel(\n"
                 "    queue,\n"
                 "    global_batch_indices,\n"
                 f"    { functor }{{}},\n"
                 f"    { mapped }\n"
                 ");" )

    def inheriting( self, other ):
        """Nous, mais en prenant de `other` ce que nous n'avons pas dit : `includes` et `sources`
        sont ce que l'APPEL compile et lie, identique dans les deux sens. La géométrie, elle, n'est
        jamais héritée -- c'est tout l'intérêt de l'avoir sortie."""
        if self.includes and self.sources:
            return self
        res = FfiCode( self.code, self.prologue, self.includes or other.includes,
                       self.sources or other.sources, self.include_roots or other.include_roots,
                       allocator = self.allocator,
                       _scaffold = self._scaffold,
                       **{ hook: self.hooks.get( hook, "" ) for hook in
                           ( "max_nb_threads", "group_size", "local_mem_elems" ) } )
        return res


class Kernels( AbstractFfiCode ):
    """CE QU'UN APPEL LANCE : un noyau aller, un noyau retour optionnel, et le nom qui identifie
    les deux.

    Ce n'est pas une classe qu'on écrit soi-même : `driver.call` la construit à partir des
    `FfiCode` qu'on lui passe. Elle existe parce que le pipeline a besoin d'UN objet à qui
    demander « ton préambule », « ton corps », « ton adjoint », « toi avec un axe de plus » -- et
    parce que ces trois dernières questions portent sur le COUPLE, pas sur un noyau.
    """

    def __init__( self, name, forward, backward = None, batch_axes = () ) -> None:
        if not name:
            raise ValueError( "driver.call: `name` is required -- it names the functors, prefixes "
                              "the compiled target and groups the compilation journal" )
        self.name = name
        self.forward = forward
        self.backward = backward
        self.batch_axes = tuple( batch_axes )

    # ce que le rendu de la source lit sur nous, directement
    @property
    def includes( self ):
        return self.forward.includes

    @property
    def sources( self ):
        return self.forward.sources

    @property
    def wants_allocator( self ):
        return self.forward.wants_allocator

    @property
    def is_handler( self ):
        return self.forward.is_handler

    def cpp_base_name( self ) -> str:
        """Le nom de l'appel, rendu identifiant C++. Tout ce qui est engendré pour cet appel en
        dérive, et c'est ce qui les garde distincts quand un catalogue lie plusieurs noyaux dans une
        seule bibliothèque."""
        base = "".join( c if c.isalnum() or c == "_" else "_" for c in self.name )
        return "_" + base if base[ 0 ].isdigit() else base

    def functor_name( self ) -> str:
        """L'identifiant C++ du foncteur engendré ( forme `per_item` )."""
        return f"{ self.cpp_base_name() }_kernel"

    def args_name( self ) -> str:
        """L'identifiant C++ de l'agrégat d'arguments ( forme générale )."""
        return f"{ self.cpp_base_name() }_args"

    def preamble_for( self, call_args_analysis ) -> str:
        return self.forward.preamble_for( call_args_analysis, self.functor_name() )

    def code_for( self, call_args_analysis ) -> str:
        return self.forward.code_for( call_args_analysis, self.functor_name() )

    @property
    def has_backward( self ):
        """Avoir un adjoint est ce qui rend un appel différentiable."""
        return self.backward is not None

    def for_backward( self ):
        """L'adjoint, prêt à être lancé comme un aller ordinaire : le retour devient le noyau
        d'un couple sans retour, sous un nom dérivé. Il garde NOS axes de batch -- ce qu'un `vmap`
        a ajouté à l'appel vaut pour les deux sens."""
        return Kernels( self.name + "_bwd", self.backward.inheriting( self.forward ),
                        batch_axes = self.batch_axes )

    def with_batch_axis( self ):
        """Le même couple, mappé sur un axe de plus : ce que lance un `vmap`. Le nom de l'axe est
        dérivé du nombre déjà présents, donc un `vmap` imbriqué en prend un frais, de façon
        déterministe."""
        name = f"vmap_{ len( self.batch_axes ) }"
        return name, Kernels( self.name, self.forward, self.backward, self.batch_axes + ( name, ) )
