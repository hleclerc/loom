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
    def wants_scratch( self ) -> bool:
        """Si le handler doit recevoir l'allocateur d'XLA (`Scratch`). Faux par défaut : c'est la
        SOURCE qui est hachée pour nommer un noyau, donc lier ce contexte sans qu'on le demande
        recompilerait tout le dépôt pour une capacité que personne n'utilise."""
        return False


class FfiCode( AbstractFfiCode ):
    """Un noyau : le C++ qui s'exécute PAR ITEM, plus ce que l'appel doit savoir pour le lancer.

    = Le corps, et rien que le corps

    `code` est ce qui se passe POUR UN ITEM. Il devient l'`operator()` d'un foncteur NOMMÉ, au
    niveau du namespace, dont les paramètres sont dérivés des arguments de l'appel : ajouter un
    argument à l'appel le fait apparaître dans la signature ET dans le `run_parallel`, sans que le
    corps change. C'est la bonne idée de tout ceci, et elle ne bouge pas.

    Un foncteur nommé, et pas une lambda : un compilateur de device (nvcc) est à l'aise avec un
    struct dont l'`operator()` est un template membre explicite, et bute sur une lambda qui
    traverse vers du code device.

    Trois noms sont RÉSERVÉS (injectés par l'échafaudage, ce ne sont pas des arguments d'appel) :
    `batch_index` (le multi-indice de l'item), `thread_index` (le numéro du work-item,
    `0..nb_threads-1`, stable sur tous les items que ce work-item traite -- c'est avec lui qu'on
    indexe un scratch PAR FIL) et `nb_threads`. Un corps qui ne s'en sert pas les ignore. Un kwarg
    de l'appel ne doit pas porter un de ces noms : il masquerait le paramètre injecté.

    = Un noyau ne se contient pas lui-même

    L'adjoint n'est PAS un champ de l'aller : c'est un autre noyau, et c'est l'APPEL qui en prend
    plusieurs.

        driver.call(
            FfiCode( "le corps aller", max_nb_threads = "..." ),
            FfiCode( "le corps retour", prologue = "..." ),      # optionnel
            name = "update_outputs",
            ... )

    Il tourne sur d'autres tampons (les résidus et les cotangentes), fait un autre travail, et n'a
    aucune raison de vouloir la même géométrie de lancement que l'aller -- un gather peut devenir
    une accumulation, un balayage peut devenir un tri. Tant qu'il était une paire de champs `bwd_*`,
    il héritait de force du plafond, du groupe et de la mémoire partagée de l'aller.

    Le NOM est sur l'appel, pas sur le noyau : il identifie le couple (les deux foncteurs en
    dérivent, `<name>_kernel` et `<name>_bwd_kernel`), il préfixe la cible compilée et il groupe le
    journal des compilations. Le porter sur chaque noyau obligeait à en écrire deux et à les tenir
    cohérents à la main.

    = La géométrie de lancement est du C++, pas des chaînes évaluées ailleurs

    `run_parallel` ne lit pas la géométrie sur un objet Python : il la DÉTECTE sur le foncteur
    (`requires { func.max_nb_threads( args... ); }`, voir `run_parallel.cxx`), et ces hooks
    reçoivent les arguments de l'appel. Ce sont donc de vraies fonctions C++, et c'est ainsi
    qu'on les écrit ici : `max_nb_threads`, `group_size` et `local_mem_elems` sont des CORPS DE
    MÉTHODE, émis seulement s'ils sont donnés, avec les arguments de l'appel en portée sous leurs
    propres noms -- exactement comme le corps.

      max_nb_threads = "return scratch.words.shape( 0 );"

    Ce que ça achète, par rapport aux chaînes d'expression d'avant : la portée est visible (c'est
    une signature), une faute est une erreur de compilation à l'endroit qu'on lit, et surtout le
    calcul peut s'appuyer sur les CONSTANTES DE L'ALGORITHME, qui sont en C++. Le budget de
    mémoire partagée d'un tri radix est une propriété de ce tri ; il n'a rien à faire dans une
    f-string Python qui recopie `NB_BUCKETS` « kept in sync by hand ».

    `max_nb_threads` borne le nombre de work-items (de GROUPES si `group_size` est donné)
    lancés : un scratch PAR FIL est alors dimensionné sur les travailleurs concurrents et non sur
    les items, donc un gros batch ne fait pas exploser la mémoire. Rien de tout ça n'entre dans la
    source : ce sont des valeurs lues à l'EXÉCUTION, donc un même noyau compilé sert toutes les
    machines.

    `group_size` fait passer à un SECOND niveau de parallélisme : chaque item lancé devient un
    work-GROUP de `group_size` voies coopérantes. Il EXIGE `local_mem_elems` (la taille, en
    `int32`, du scratch de mémoire locale partagé par les voies) -- `run_parallel` ne prend le
    chemin coopératif que si les deux hooks existent, donc l'un sans l'autre serait silencieusement
    ignoré : c'est refusé ici. Le jeu des noms réservés change alors : `batch_index` garde son sens
    (quel item ce GROUPE traite), et `thread_index`/`nb_threads` cèdent la place à `group_index`
    (le numéro du groupe, stable sur les items qu'il enjambe -- c'est lui qui indexe un scratch PAR
    GROUPE), `local_index` (le rang de la voie dans son groupe), `local_size`, `group` (pour
    `group_barrier`), `local_scratch` (la vue `int32` partagée, dimensionnée par `local_mem_elems`,
    dont l'accès sans course est l'affaire du corps) et `sub_group` (le niveau warp). Comme le
    `group_index` d'un groupe -- donc sa ligne de scratch -- est RÉUTILISÉ d'un item à l'autre, un
    corps DOIT finir par un `group_barrier` après sa dernière lecture de cette ligne.

    = Le reste

    `includes` : les en-têtes dont le corps a besoin, émis après ceux du runtime. `sources` : les
    unités C++ qu'il LIE (`"sdot/x.cpp"` ou `( "sdot/x.cpp", { "DEF": "1" } )`), compilées une fois
    par (source, defines, compilateur) et partagées par tous les noyaux qui les nomment.

    `prologue` : une instruction C++ émise VERBATIM avant le lancement, dans la portée du
    HANDLER -- où `queue` et les agrégats de l'appel sont déclarés. C'est la position d'une
    pré-passe unique qu'un corps par item ne peut pas exprimer. Mettre une sortie accumulée à zéro
    n'en fait plus partie : toute sortie part semée (voir `CallArg_Tensor.cpp_seed_member`).

    = L'échappatoire : un corps qui est TOUT le handler

    `FfiCode.handler( ... )` construit un noyau dont le corps n'est pas échafaudé mais recopié tel
    quel dans le handler -- à lui d'appeler `run_parallel`, ou de ne pas le faire du tout. C'est ce
    qu'il faut pour du code HÔTE qui a besoin de la `queue` et pilote lui-même son parallélisme :
    le solveur de transport d'`OtPlan` est exactement ça (cent diagrammes dans un seul appel).
    Rare, mais réel -- et jusqu'ici l'échappatoire existait sans que rien ne la nomme.
    """

    def __init__( self, code = "", prologue = "", includes = (), sources = (),
                  max_nb_threads = "", group_size = "", local_mem_elems = "",
                  scratch = False, _scaffold = True ) -> None:
        if group_size and not local_mem_elems:
            raise ValueError( "FfiCode: `group_size` without `local_mem_elems` -- `run_parallel` "
                              "only takes the cooperative path when BOTH hooks exist, so this "
                              "would be silently ignored" )
        if local_mem_elems and not group_size:
            raise ValueError( "FfiCode: `local_mem_elems` without `group_size` -- there is no "
                              "work-group to share it" )

        self.code = code
        self.prologue = prologue
        self.sources = tuple( sources )
        self.includes = tuple( includes )
        self.scratch = bool( scratch )
        self._scaffold = _scaffold

        # les corps des hooks que `run_parallel` détecte sur le foncteur -- du C++, pas des
        # expressions évaluées dans une portée que l'appelant ne voit pas. L'ordre compte :
        # `local_mem_elems` peut appeler `group_size`.
        self.hooks = { hook: body for hook, body in
                       ( ( "max_nb_threads", max_nb_threads ), ( "group_size", group_size ),
                         ( "local_mem_elems", local_mem_elems ) ) if body }

    @classmethod
    def handler( cls, code = "", **kwargs ):
        """Un noyau dont le corps EST le handler : recopié tel quel, aucun foncteur, aucun
        `run_parallel` engendré. Voir la docstring de la classe."""
        if any( kwargs.get( k ) for k in ( "max_nb_threads", "group_size", "local_mem_elems" ) ):
            raise ValueError( "FfiCode.handler: a verbatim handler launches itself, so it has no "
                              "functor for a launch hook to live on" )
        return cls( code, _scaffold = False, **kwargs )

    # ---- ce que l'appel demande ( il passe le nom du foncteur, qu'il est seul à connaître ) ----

    @property
    def cooperative( self ):
        return "group_size" in self.hooks

    @property
    def wants_scratch( self ):
        return self.scratch

    def _params( self, names ):
        """Les paramètres de l'`operator()` : les réservés, puis un par argument de l'appel --
        chacun avec son propre paramètre de template, puisque leur type côté kernel est décidé en
        C++."""
        if self.cooperative:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "int", "group_index" ), ( "int", "local_index" ),
                         ( "int", "local_size" ), ( "Group", "group" ), ( "LocalScratch", "local_scratch" ),
                         ( "SubGroup", "sub_group" ) ]
        else:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "int", "thread_index" ), ( "int", "nb_threads" ) ]
        params = reserved + [ ( f"T_{ n }", n ) for n in names ]
        tparams = [ t for t, _ in params if t != "int" ]
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
            return ""
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
            return prologue + self.code

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
                       self.sources or other.sources, scratch = self.scratch,
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
    def wants_scratch( self ):
        return self.forward.wants_scratch

    def functor_name( self ) -> str:
        """L'identifiant C++ du foncteur : le nom de l'appel, rendu identifiant."""
        base = "".join( c if c.isalnum() or c == "_" else "_" for c in self.name )
        if base[ 0 ].isdigit():
            base = "_" + base
        return f"{ base }_kernel"

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
