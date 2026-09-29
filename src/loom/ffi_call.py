"""LE point d'entree public : executer un noyau C++/CUDA, et le VOCABULAIRE de ses arguments.

`driver` est la couche BASSE -- il porte le framework ( Jax/Torch ), le device, les types resolus,
et il n'a pas a apparaitre dans le code d'un usager. `ffi_call` est le meme appel, sous le nom de
ce qu'il fait, et avec une autre facon de dire les entrees et les sorties.

CE QUI CHANGE PAR RAPPORT A `driver.call`, et pourquoi :

  * `name` est le PREMIER argument. Il est obligatoire -- il nomme les foncteurs, prefixe la
    cible compilee et groupe le journal des compilations -- donc il n'a rien a faire au milieu
    des donnees.

  * LE ROLE SE DIT SUR LA VALEUR, pas dans une liste a cote. `driver.call` porte quatre listes
    de chemins ( `output_attributes`, les deux `exceptions`, `scratch_attributes` ) qu'il faut
    tenir en accord A LA MAIN avec les noms des kwargs -- et dont une faute de frappe ne se
    voyait qu'au moment ou un attribut introuvable etait signale. Ici le role est PORTE par
    l'argument : `suivant = loom.out( suivant )` ne peut plus designer autre chose que lui-meme.

  * CE FAISANT, le namespace du noyau lui est rendu -- ENTIEREMENT. `driver.call` reservait HUIT
    noms ( `name`, `output_attributes`, `output_exceptions`, `input_exceptions`,
    `output_capacities`, `batch_alignment`, `has_dynamic_capacity`, `scratch_attributes` ) : un
    noyau qui voulait un argument appele `name` ne le pouvait pas, et `output_attribute` au
    singulier ne levait rien -- il devenait un argument du noyau. Les arguments vivant desormais
    DANS UN GROUPE, le premier niveau de `args` ne contient plus que les groupes : un argument
    appele `name`, `output_attributes` ou meme `inputs` vit a `args.inputs.<son nom>` et ne peut
    plus rien heurter.

LES GROUPES, c'est-a-dire ce que le C++ voit :

    args.inputs.<nom>         les entrees -- y compris un scalaire ou un entier nu
    args.outputs.<nom>        ce que le noyau ecrit
    args.scratch.<nom>        les tampons de travail
    args.grad_of_inputs       ( adjoint ) la cotangente a ECRIRE
    args.grad_of_outputs      ( adjoint ) la cotangente qui ENTRE
    args.machine, args.errors l'appel, pas l'usager

LE NOM D'UN GROUPE DESIGNE TOUJOURS LE ROLE A L'ALLER. C'est ce qui enleve le noeud de
`grad_inputs` / `grad_outputs`, ou « inputs » pouvait aussi se lire comme le role du RETOUR --
les deux lectures echangeant alors le sens. Le retour LIT `grad_of_outputs` et ECRIT
`grad_of_inputs`, parce qu'une cotangente de sortie est ce qu'on recoit et une cotangente
d'entree ce qu'on derive. Le miroir est le contenu ; il n'y a rien de plus a retenir.

LE VOCABULAIRE :

    loom.out( x )                        x est ECRIT par le noyau
    loom.out( cell, "nb_vertices" )      ... seulement ces membres ; le reste est observe
    loom.mutable( x )                    x est LU puis RE-ECRIT ( voir `mutable` )
    loom.scratch( x )                    un tampon de travail : alloue et ecrit a l'aller, pas
                                         rendu a l'adjoint comme residu
    loom.unbound( cell, "cut_offsets" )  ce que ce noyau n'a pas a toucher, meme si c'est rempli

Un argument sans marqueur est une ENTREE, et une valeur brute y entre telle quelle : un tableau
du framework, un tableau numpy, une liste, un flottant ( voir `Tensor.as_tensor` -- loom en lit
le type d'element, qui est un fait, et laisse la taille du scalaire au driver, qui est une
politique ).
"""

# LES ROLES. Des chaines et pas un enum : ces quatre valeurs ne sortent jamais de ce fichier.
_OUT, _MUTABLE, _SCRATCH, _UNBOUND = "out", "mutable", "scratch", "unbound"


class Arg:
    """Un argument PLUS le role qu'il tient dans l'appel.

    Bati par `loom.out` & co, jamais directement. `members` restreint le role a des sous-chemins
    de l'objet -- ce qu'un agregat demande : une `Cell` dont le noyau n'ecrit que `nb_vertices`
    et `vertex_positions`. Vide, le role vaut pour tout l'objet.
    """

    __slots__ = ( "kind", "value", "members", "capacities" )

    def __init__( self, kind, value, members = (), capacities = {} ) -> None:
        self.kind = kind
        self.value = value
        self.members = tuple( members )
        self.capacities = dict( capacities )

    def paths( self, name ):
        """Les chemins que ce role designe, vus de l'appel : l'argument, ou ses membres."""
        return [ f"{ name }.{ m }" for m in self.members ] if self.members else [ name ]

    def capacity_paths( self, name ):
        """Les capacites, re-clefees sur le chemin complet ( `nb_vertices` -> `cell.nb_vertices` )."""
        return { f"{ name }.{ p }": c for p, c in self.capacities.items() }


def out( value, *members, capacities = {} ):
    """`value` est ECRIT par le noyau : un tampon neuf, rendu a l'objet une fois l'appel fini.

    Nommer des `members` restreint l'ecriture a ces sous-chemins -- le reste de l'objet est alors
    observe comme n'importe quelle entree. `capacities` dit COMBIEN allouer, quand seul l'appelant
    le sait ( `loom.out( cell, capacities = { "nb_vertices": 8 } )` ) ; une capacite deja
    materialisee dans un tampon n'a pas a etre repetee, elle s'y lit."""
    return Arg( _OUT, value, members, capacities )


def mutable( value, *, capacities = {} ):
    """`value` est LU puis RE-ECRIT -- et `ffi_call` rend la nouvelle valeur.

    LES ENTREES ET LES SORTIES D'UN APPEL SONT DISJOINTES, comme dans XLA : un noyau n'ecrit
    jamais ce qu'il lit. Une mise a jour en place est donc DEUX tampons plus un rebinding cote
    python -- ce que ce marqueur ecrit a notre place. Le C++ voit les deux, sous des noms
    derives :

        temperature = loom.mutable( temperature )
            -> args.inputs.temperature      ce qui entre
            -> args.outputs.temperature     ce qui sort
            ( et pour l'adjoint : `args.grad_of_outputs.temperature` en entre,
              `args.grad_of_inputs.temperature` en sort )

    Le tampon de sortie est bati « comme » l'entree ( voir `_empty_like` ). Une CAPACITE ne
    decrit jamais qu'une ALLOCATION, et seule la sortie est allouee : `nb_vertices` y designe
    donc le cote sortie, sans ambiguite a lever.

    Ce qui revient est de la meme espece que ce qu'on a donne : un tableau du framework si on a
    passe un tableau, un tenseur loom si on a passe un tenseur."""
    return Arg( _MUTABLE, value, (), capacities )


def scratch( value, *members, capacities = {} ):
    """Un tampon de TRAVAIL : alloue et ecrit a l'aller comme une sortie, mais pas rendu a
    l'adjoint comme residu -- ce qu'il portait a l'aller est du transitoire par fil, que le retour
    re-alloue et re-derive lui-meme."""
    return Arg( _SCRATCH, value, members, capacities )


def unbound( value, *members ):
    """Ce que CE noyau n'a pas a toucher, meme si l'attribut est rempli : rien ne traverse la FFI,
    et -- n'etant pas un tampon -- rien ne devient une primale derivable, donc l'adjoint n'a pas
    de cotangente a lui trouver. Pour les membres qu'un agregat porte et dont ce noyau n'a que
    faire."""
    return Arg( _UNBOUND, value, members )


def ffi_call( name, *kernels, batch_alignment = None, has_dynamic_capacity = True, **args ):
    """Lance `kernels` sur les valeurs passees en kwargs.

        temperature = loom.ffi_call(
            "diffusion_pas",                 # le nom de l'appel : obligatoire, donc en premier
            avant, arriere,                  # le second est l'ADJOINT ( optionnel )
            temperature = loom.mutable( temperature ),
            coef = coef,
        )

    CE QUI EST RENDU : la valeur des arguments `mutable`, dans l'ordre ou ils ont ete donnes ( la
    valeur seule s'il n'y en a qu'un, un tuple sinon ), et `None` s'il n'y en a aucun. Une sortie
    declaree par `loom.out` n'est PAS rendue : l'objet est deja le notre, le resultat y est
    reecrit. Le framework, lui, ne voit jamais nos objets -- seulement les tenseurs dedans.
    """
    from .drivers.driver import driver

    donnees, sorties, scratchs, non_lies = {}, [], [], []
    capacites = {}
    mutables = []           # ( objet de sortie, ce qu'on nous a donne ), dans l'ordre des kwargs
    # LES GROUPES : membre C++ -> chemin. Le chemin est ce que les capacites et les sorties
    # nomment cote python ; le membre est ce que le noyau ecrit. Les deux coincident partout sauf
    # pour un `mutable`, qui donne DEUX tampons au MEME nom, dans deux groupes.
    groupes = { "inputs": {}, "outputs": {}, "scratch": {} }

    for cle, valeur in args.items():
        if not isinstance( valeur, Arg ):
            donnees[ cle ] = valeur
            groupes[ "inputs" ][ cle ] = cle
            continue

        if valeur.kind == _MUTABLE:
            entree, sortie = f"{ cle }_input", f"{ cle }_output"
            # les deux tampons portent le nom C++ `cle`, mais chacun son CHEMIN -- et un chemin est
            # ce que les capacites et les sorties designent. Deux chemins egaux les confondraient,
            # en silence : on refuse.
            for derive in ( entree, sortie ):
                if derive in args:
                    raise ValueError(
                        f"'{ cle } = loom.mutable( ... )' engendre le chemin '{ derive }', qui est "
                        f"deja un argument de cet appel. Renomme l'un des deux." )
            objet = _empty_like( valeur.value, cle )
            donnees[ entree ] = valeur.value
            donnees[ sortie ] = objet
            groupes[ "inputs" ][ cle ] = entree
            groupes[ "outputs" ][ cle ] = sortie
            sorties.append( sortie )
            capacites.update( valeur.capacity_paths( sortie ) )
            mutables.append( ( objet, valeur.value ) )
            continue

        donnees[ cle ] = valeur.value
        capacites.update( valeur.capacity_paths( cle ) )
        chemins = valeur.paths( cle )
        if valeur.kind == _UNBOUND:
            # il ne traverse pas, mais il reste UN ARGUMENT : le noyau le voit ( non lie ), donc
            # il vit du cote ou il serait lu.
            non_lies += chemins
            groupes[ "inputs" ][ cle ] = cle
        elif valeur.kind == _SCRATCH:
            # un scratch est alloue et ecrit comme une sortie ; ce qui le distingue est ce que
            # l'ADJOINT en recoit, c'est-a-dire rien -- pas l'endroit ou il vit.
            sorties += chemins
            scratchs += chemins
            groupes[ "scratch" ][ cle ] = cle
        else:
            sorties += chemins
            groupes[ "outputs" ][ cle ] = cle

    driver.call(
        *kernels,
        name = name,
        output_attributes = sorties,
        scratch_attributes = scratchs,
        input_exceptions = non_lies,
        output_capacities = capacites,
        batch_alignment = batch_alignment,
        has_dynamic_capacity = has_dynamic_capacity,
        groups = { g: m for g, m in groupes.items() if m },
        # PAS `**donnees` : à plat, les arguments retomberaient dans le namespace des options
        # ci-dessus. Sous les groupes la collision est déjà impossible ( le premier niveau ne
        # contient que les groupes ), mais le chemin plat existe encore le temps de la migration.
        call_args = donnees,
    )

    if not mutables:
        return None
    rendus = [ _same_kind( objet, donne ) for objet, donne in mutables ]
    return rendus[ 0 ] if len( rendus ) == 1 else tuple( rendus )


def _empty_like( value, name ):
    """Le jumeau VIDE de `value` -- le tampon de sortie d'un `mutable`.

    Deux protocoles, dans cet ordre : `_empty_like_me()`, qu'un objet compose definit lui-meme
    ( une `Cell` a besoin de sa dimension, de son batch et de son type de noyau, que loom ne peut
    pas deviner ) ; sinon `Tensor.like`, qui couvre un tenseur loom comme une valeur brute."""
    faire = getattr( value, "_empty_like_me", None )
    if faire is not None:
        return faire()

    from .tensor.Tensor import Tensor
    try:
        return Tensor.like( value )
    except TypeError as e:
        raise TypeError(
            f"'{ name } = loom.mutable( ... )' : loom ne sait pas batir le tampon de sortie d'un "
            f"{ type( value ).__name__ }. Un tenseur, ou une valeur ayant une forme, passe par "
            f"`Tensor.like` ; un objet compose doit definir `_empty_like_me()`." ) from e


def _same_kind( objet, donne ):
    """Ce qu'un `mutable` rend : la meme espece que ce qu'on nous a donne. Un tableau brut est
    entre, un tableau brut ressort ; un tenseur loom est entre, le tenseur ressort."""
    from .tensor.Tensor import Tensor
    if isinstance( donne, Tensor ) or not isinstance( objet, Tensor ):
        return objet
    return objet.raw
