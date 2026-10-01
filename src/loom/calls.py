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

    loom.out( x )                             x est ECRIT par le noyau
    loom.out( cell, writes = ( "nb_vertices", ) )   ... ceux-la, et eux seuls
    loom.out( cell, reads  = ( "weights", ) )       ... tout sauf ceux-la ( le meme, par l'autre
                                                    bout -- on donne la liste la plus courte )
    loom.mutable( x )                         x est LU puis RE-ECRIT ( voir `mutable` )
    loom.scratch( x )                         un tampon de travail : alloue et ecrit a l'aller,
                                              pas rendu a l'adjoint comme residu
    loom.unbound( cell, "cut_offsets" )       ce que ce noyau n'a pas a toucher, meme si c'est
                                              rempli

Un argument sans marqueur est une ENTREE, et une valeur brute y entre telle quelle : un tableau
du framework, un tableau numpy, une liste, un flottant ( voir `Tensor.as_tensor` -- loom en lit
le type d'element, qui est un fait, et laisse la taille du scalaire au driver, qui est une
politique ).

CE QUE L'APPEL REND : tout ce qu'il a ECRIT -- `out` comme `mutable` -- dans l'ordre des kwargs
( voir `returned` ). Un appel qui n'ecrit rien rend `None`.
"""

# LES ROLES. Des chaines et pas un enum : ces quatre valeurs ne sortent jamais de ce fichier.
_OUT, _MUTABLE, _SCRATCH, _UNBOUND = "out", "mutable", "scratch", "unbound"


class Arg:
    """Un argument PLUS le role qu'il tient dans l'appel.

    Bati par `loom.out` & co, jamais directement. `members` restreint le role a des sous-chemins
    de l'objet -- ce qu'un agregat demande : une `Cell` dont le noyau n'ecrit que `nb_vertices`
    et `vertex_positions`. Vide, le role vaut pour tout l'objet. `excluded` dit la MEME chose par
    l'autre bout : l'objet entier tient le role, sauf ces sous-chemins ( voir `out` ).
    """

    __slots__ = ( "kind", "value", "members", "excluded", "capacities" )

    def __init__( self, kind, value, members = (), excluded = (), capacities = {} ) -> None:
        if members and excluded:
            raise ValueError(
                "loom.out / loom.scratch : `writes` et `reads` disent la MEME chose par ses deux "
                "bouts -- ce que le noyau ecrit, ou ce qu'il laisse. En donner les deux, c'est "
                "ouvrir la porte a ce qu'ils se contredisent : n'en donne qu'un." )
        self.kind = kind
        self.value = value
        self.members = tuple( members )
        self.excluded = tuple( excluded )
        self.capacities = dict( capacities )

    def paths( self, name ):
        """Les chemins que ce role designe, vus de l'appel : l'argument, ou ses membres."""
        return [ f"{ name }.{ m }" for m in self.members ] if self.members else [ name ]

    def excluded_paths( self, name ):
        """Les chemins DECOUPES de ce role : ils gardent celui qu'ils auraient eu sans la
        declaration, c'est-a-dire ENTREE pour ce qui porte une valeur."""
        return [ f"{ name }.{ m }" for m in self.excluded ]

    def capacity_paths( self, name ):
        """Les capacites, re-clefees sur le chemin complet ( `nb_vertices` -> `cell.nb_vertices` )."""
        return { f"{ name }.{ p }": c for p, c in self.capacities.items() }


def out( value, *, writes = (), reads = (), capacities = {} ):
    """`value` est ECRIT par le noyau : un tampon neuf, rendu a l'objet une fois l'appel fini.

    ET RENDU PAR L'APPEL, donc `return loom.ffi_call( ... )` suffit -- plus besoin de batir
    l'objet, d'appeler, puis de le rendre sur une troisieme ligne ( voir `returned` ).

    UN AGREGAT DONT LE NOYAU N'ECRIT QU'UNE PARTIE se dit par l'un des deux bouts, au choix, et
    c'est le plus court qui gagne :

        loom.out( cell, writes = ( "nb_vertices", "vertex_positions" ) )   ceux-la, et eux seuls
        loom.out( cell, reads  = ( "weights", ) )                          tout sauf ceux-la

    LES DEUX LISTES SONT EXCLUSIVES, et non des indications : ce qui n'est pas ecrit est observe
    comme n'importe quelle entree. Une faute de frappe ne passe pas -- un nom qui ne designe rien,
    ou un `reads` qui n'exclut rien, est refuse.

    `reads` est en general la liste qui ne pourrit pas : elle nomme ce que le noyau ne doit PAS
    ecraser, donc elle ne bouge pas quand on ajoute une sortie a l'agregat.

    `capacities` dit COMBIEN allouer, quand seul l'appelant le sait ( `loom.out( cell, capacities
    = { "nb_vertices": 8 } )` ) ; une capacite deja materialisee dans un tampon n'a pas a etre
    repetee, elle s'y lit."""
    return Arg( _OUT, value, writes, reads, capacities )


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


def scratch( value, *, writes = (), reads = (), capacities = {} ):
    """Un tampon de TRAVAIL : alloue et ecrit a l'aller comme une sortie, mais pas rendu a
    l'adjoint comme residu -- ce qu'il portait a l'aller est du transitoire par fil, que le retour
    re-alloue et re-derive lui-meme. Il n'est pas rendu par l'appel non plus : un transitoire
    n'est pas un resultat.

    `writes` / `reads` comme pour `out`."""
    return Arg( _SCRATCH, value, writes, reads, capacities )


def unbound( value, *members ):
    """Ce que CE noyau n'a pas a toucher, meme si l'attribut est rempli : rien ne traverse la FFI,
    et -- n'etant pas un tampon -- rien ne devient une primale derivable, donc l'adjoint n'a pas
    de cotangente a lui trouver. Pour les membres qu'un agregat porte et dont ce noyau n'a que
    faire.

    La liste est POSITIONNELLE ici, et c'est coherent : elle nomme ce qui n'est pas lie, donc elle
    dit directement le role. `out`, lui, devait dire LEQUEL des deux roles sa liste designait --
    d'ou `writes` / `reads`."""
    return Arg( _UNBOUND, value, members )


def lower_args( args ):
    """LE VOCABULAIRE, traduit en ce que l'abaissement attend.

    Rend `( donnees, sorties, scratchs, non_lies, capacites, groupes, rendus, exceptions )` :

        donnees     { chemin: objet }        ce qui traverse
        groupes     { groupe: { membre: chemin } }   le premier niveau de `args` cote C++
        rendus      [ ( objet, ce qu'on nous a donne | None ) ]   ce que l'appel REND
        exceptions  [ chemin ]               les sous-chemins DECOUPES d'une sortie ( `reads` )

    Le CHEMIN et le NOM C++ se separent ici, et c'est ce qui rend `mutable` possible : deux
    tampons portent le MEME nom C++ dans deux groupes, sous deux chemins distincts -- un chemin
    etant ce que les capacites et les sorties designent.

    Vit ici, avec les marqueurs, et non dans le driver : c'est le meme sujet.
    """
    donnees, sorties, scratchs, non_lies = {}, [], [], []
    capacites = {}
    rendus, exceptions = [], []
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
            rendus.append( ( objet, valeur.value ) )
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
            exceptions += valeur.excluded_paths( cle )
            groupes[ "scratch" ][ cle ] = cle
        else:
            sorties += chemins
            exceptions += valeur.excluded_paths( cle )
            groupes[ "outputs" ][ cle ] = cle
            rendus.append( ( valeur.value, None ) )

    return ( donnees, sorties, scratchs, non_lies, capacites,
             { g: m for g, m in groupes.items() if m }, rendus, exceptions )


def returned( rendus ):
    """Ce qu'un appel REND : ses arguments ECRITS, dans l'ordre ou ils ont ete donnes ( la valeur
    seule s'il n'y en a qu'un, un tuple sinon ), et `None` s'il n'y en a aucun.

    `out` ET `mutable`, et c'est ce qui permet d'ecrire `return loom.ffi_call( ... )` au lieu de
    batir l'objet, d'appeler, puis de le rendre sur une troisieme ligne.

    Ce qui revient n'est pas le meme objet dans les deux cas, et ca ne peut pas l'etre : un `out`
    rend CELUI QU'ON A DONNE -- il etait deja le notre, le resultat y a ete reecrit -- la ou un
    `mutable` rend le NOUVEAU tampon ( les entrees et les sorties d'un appel sont disjointes ),
    de la meme espece que ce qu'on avait donne ( voir `_same_kind` ).

    `scratch` ne rend rien, et c'est sa definition : un transitoire n'est pas un resultat."""
    if not rendus:
        return None
    res = [ objet if donne is None else _same_kind( objet, donne ) for objet, donne in rendus ]
    return res[ 0 ] if len( res ) == 1 else tuple( res )


def ffi_call( name, *kernels, **kwargs ):
    """Lance `kernels` sur les valeurs passees en kwargs.

    C'EST `driver.call`, sous le nom de ce qu'il fait : il n'y a qu'une forme d'appel, et `driver`
    reste la couche basse que l'usager n'a pas a nommer.

    CE QUI EST RENDU : les arguments ECRITS -- `loom.out` comme `loom.mutable` -- dans l'ordre ou
    ils ont ete donnes ( voir `returned` ). D'ou la forme courte :

        return loom.ffi_call( "mon_appel", noyau, entree = entree,
                              sortie = loom.out( RealTensor[ n ]() ) )
    """
    from .drivers.driver import driver
    return driver.call( name, *kernels, **kwargs )


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
    entre, un tableau brut ressort ; un tenseur loom est entre, le tenseur ressort.

    `.value` ET PAS `.raw`. `raw` est le TAMPON, dimensionne a la CAPACITE -- rembourrage compris,
    parce que c'est ce dans quoi un noyau ecrit ( l'alignement de batch vaut 128 octets sur CUDA,
    donc un lot de 3 occupe 16 fentes en fp64 ). Le rendre serait rendre le rembourrage avec, en
    silence, et la valeur logique est ce qu'on veut. `value` recadre sur la forme."""
    from .tensor.Tensor import Tensor
    if isinstance( donne, Tensor ) or not isinstance( objet, Tensor ):
        return objet
    return objet.value
