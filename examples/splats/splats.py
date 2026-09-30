"""Du splatting gaussien 2D : un RAGGED dont la longueur est decouverte par le noyau.

Deuxieme usager delibererement etranger de loom ( le premier est `examples/diffusion` ). Il
n'importe que `loom`, son C++ ne connait que `<loom/support/...>`, et ce qu'il exerce est
exactement ce que `diffusion` ne touchait pas :

  * un `ShapeVar` PAR TUILE, ecrit par le noyau -- donc une forme que Python ne connait pas au
    tracage, et la boucle capacite / depassement / on-recommence ;
  * un PIPELINE a plusieurs passes qui se partagent des agregats ;
  * un adjoint en ACCUMULATION ATOMIQUE ( celui de `diffusion` etait en gather pur ).

    image( p ) = somme_i  opacite_i * exp( -q_i( p ) / 2 ) * couleur_i

Melange ADDITIF : pas d'ordre, donc pas de tri -- la composition alpha viendra apres.

= Pourquoi le ragged n'est pas un confort ici

Un pixel ne doit regarder que les splats qui l'atteignent, sinon le rendu est en O( pixels x
splats ). D'ou un index `tuile -> splats qui la touchent`, dont la longueur DEPEND DES DONNEES.

Ce que XLA peut et ne peut pas, exactement -- parce que l'argument doit etre honnete : il peut
construire un tel index, a condition qu'on BORNE le total a l'avance ( un compte, une somme
prefixe, un scatter de taille fixe ). Ce qu'il ne peut pas, c'est le decouvrir. La difference se
paie : une borne trop petite perd des splats en silence, une borne sure coute le pire cas pour tout
le monde. Ici le noyau ECRIT le compte, et si la capacite etait trop petite il le dit -- l'hote
reserve plus grand et relance. `reference_jax.py` mesure ce que la borne coute.
"""
from pathlib import Path

# `loom.Truc` et jamais `Truc` tout court : dans un exemple, on doit voir d'ou vient chaque nom.
import loom
import loom.compilation

# le C++ de CE paquet, enregistre aupres de loom comme n'importe quel usager
loom.compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )

COTE = 16          # une tuile de 16 x 16 pixels


class Splats( loom.Aggregate ):
    """Les gaussiennes : centre, inverse de covariance ( a, b, c ), couleur, opacite."""
    centres   : loom.RealTensor[ "splat", "xy" ]
    cov_inv   : loom.RealTensor[ "splat", "abc" ]
    couleurs  : loom.RealTensor[ "splat", "rvb" ]
    opacites  : loom.RealTensor[ "splat" ]

    splat     : loom.Axis[ "nb_splats" ]
    xy        : loom.Axis[ "nb_xy" ]
    abc       : loom.Axis[ "nb_abc" ]
    rvb       : loom.Axis[ "nb_rvb" ]

    nb_splats : loom.ShapeVar
    nb_xy     : loom.CtShapeVar
    nb_abc    : loom.CtShapeVar
    nb_rvb    : loom.CtShapeVar


class Index( loom.Aggregate ):
    """L'INDEX RAGGED : pour chaque tuile, les splats qui la touchent.

    `nb_par_tuile : ShapeVar[ "tuile" ]` est UN COMPTE PAR TUILE -- un `ShapeVar` qui varie le long
    d'un axe ( ses `dep_axes` ). C'est la declaration d'un ragged en loom, et c'est une capacite que
    son seul usager reel n'exerce nulle part : `grep dep_axes sdot/src` ne rend rien.

    Le compte est ECRIT PAR LE NOYAU ( passe 1 ) ; `ids` est alloue a la capacite que l'appel
    demande, et un compte qui la depasse est signale au lieu d'etre tronque en silence.
    """
    ids          : loom.IntTensor[ "tuile", "fente" ]

    tuile        : loom.Axis[ "nb_tuiles" ]
    fente        : loom.Axis[ "nb_par_tuile" ]

    nb_tuiles    : loom.CtShapeVar
    nb_par_tuile : loom.ShapeVar[ "tuile" ]


class Ecran( loom.Aggregate ):
    """La geometrie de l'image, connue a la compilation : le stencil de tuiles y gagne, au prix
    d'une compilation par resolution ( comme la grille de `examples/diffusion` )."""
    largeur : loom.CtShapeVar
    hauteur : loom.CtShapeVar
    cote    : loom.CtShapeVar


def _nb_tuiles( largeur, hauteur, cote = COTE ):
    return ( ( largeur + cote - 1 ) // cote ) * ( ( hauteur + cote - 1 ) // cote )


_INSCRIRE = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI i = flat_index;
        splats::inscrire( inputs.splats, i, SI( inputs.ecran.largeur ), SI( inputs.ecran.hauteur ), SI( inputs.ecran.cote ),
                          outputs.index.nb_par_tuile, outputs.index.ids );
    """,
)

_RENDRE = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = flat_index;
        const SI largeur = SI( inputs.ecran.largeur ), cote = SI( inputs.ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        splats::rendre_pixel( inputs.splats, inputs.index.ids, SI( inputs.index.nb_par_tuile( t ) ), t, px, py,
                              outputs.image( y = py, x = px ) );
    """,
)

_RENDRE_BWD = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = flat_index;
        const SI largeur = SI( inputs.ecran.largeur ), cote = SI( inputs.ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        if constexpr ( ! grad_of_outputs.image.surely_null )
            splats::rendre_pixel_bwd( inputs.splats, inputs.index.ids, SI( inputs.index.nb_par_tuile( t ) ), t, px, py,
                                      grad_of_outputs.image( y = py, x = px ), grad_of_inputs.splats );
    """,
)


def construire_index( splats, ecran, capacite ):
    """PASSE 1 : l'index ragged. NON differentiable ( il rend des entiers, et son lien aux centres
    est discontinu -- un splat entre ou n'entre pas dans une tuile ).

    `capacite` est une DEVINETTE : combien de splats par tuile au plus. Si elle est trop petite, le
    noyau ecrit quand meme le compte VOULU et signale le depassement, et `driver.call` reserve plus
    grand et relance tout seul -- l'appelant n'a rien a faire. La capacite finalement retenue se lit
    dans `index.ids.capacity`, ce qui est la façon d'observer que la croissance a eu lieu.
    """
    index = Index( nb_tuiles = _nb_tuiles( int( ecran.largeur.value ), int( ecran.hauteur.value ),
                                           int( ecran.cote.value ) ) )
    loom.ffi_call(
        "splats_inscrire",
        _INSCRIRE,
        splats = splats,
        ecran = ecran,
        index = loom.out( index, capacities = { "nb_par_tuile": capacite } ),
        nb_items = int( splats.nb_splats.value ),
    )
    return index


def rendre( splats, index, ecran ):
    """PASSE 2 : l'image. Differentiable par rapport a tout ce que porte `splats`."""
    largeur, hauteur = int( ecran.largeur.value ), int( ecran.hauteur.value )
    image = loom.RealTensor[ loom.Axis( loom.ShapeVar( hauteur ), name = "y" ),
                        loom.Axis( loom.ShapeVar( largeur ), name = "x" ), splats.rvb ]()
    loom.ffi_call(
        "splats_rendre",
        _RENDRE, _RENDRE_BWD,
        splats = splats,
        index = index,
        ecran = ecran,
        image = loom.out( image ),
        nb_items = largeur * hauteur,
    )
    return image.tensor


def rendu( splats, ecran, capacite ):
    """Le rendu de bout en bout, differentiable : l'index est construit sur des centres dont le
    gradient est COUPE ( l'affectation d'un splat a une tuile est discontinue, et ce n'est pas par
    la que passe la derivee -- c'est aussi ce que fait un 3DGS ), puis l'image est rendue."""
    index = construire_index( _sans_gradient( splats ), ecran, capacite )
    return rendre( splats, index, ecran )


def _sans_gradient( splats ):
    """Les memes splats, detaches : ce que la passe 1 lit."""
    autre = Splats( nb_xy = 2, nb_abc = 3, nb_rvb = 3 )
    autre.centres  = loom.stop_gradient( splats.centres )
    autre.cov_inv  = loom.stop_gradient( splats.cov_inv )
    autre.couleurs = loom.stop_gradient( splats.couleurs )
    autre.opacites = loom.stop_gradient( splats.opacites )
    return autre

# ── L'AUTRE REPRESENTATION : CSR, en deux passes ─────────────────────────────────────────────────
#
# L'index rembourre ci-dessus coute `tuiles x max_par_tuile`, alors que le contenu utile est
# `total_des_couples`. Un CSU -- offsets + liste unique -- coute exactement l'utile. La question
# honnete est donc : le ragged rembourre vaut-il son gaspillage ?
#
# Ce qu'il echange, c'est de la MEMOIRE contre une PASSE. Le rembourre inscrit en un seul balayage
# des splats ( reserver une fente et ecrire ). Le CSR en demande deux : compter, puis -- les offsets
# etant connus -- remplir. Entre les deux il faut une somme prefixe, et surtout il faut LIRE LE
# TOTAL pour allouer la liste.
#
# Et c'est la que se trouve la vraie difference avec un JIT, pas dans le rembourrage : lire ce total
# est une lecture HOTE d'un compte qu'un noyau vient d'ecrire. loom sait le faire ( `ShapeArray` est
# fait pour ca ), et alloue donc EXACTEMENT. XLA ne peut pas : sous `jit` le compte est un tracer,
# et il faut borner le total avant de tracer. Voir le tableau du README.

_COMPTER = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        splats::compter( inputs.splats, flat_index, SI( inputs.ecran.largeur ),
                         SI( inputs.ecran.hauteur ), SI( inputs.ecran.cote ), outputs.comptes );
    """,
)

_REMPLIR = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        splats::remplir( inputs.splats, flat_index, SI( inputs.ecran.largeur ),
                         SI( inputs.ecran.hauteur ), SI( inputs.ecran.cote ), inputs.offsets, outputs.curseurs, outputs.ids_plat );
    """,
)

_RENDRE_CSR = loom.FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = flat_index;
        const SI largeur = SI( inputs.ecran.largeur ), cote = SI( inputs.ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        splats::rendre_pixel_csr( inputs.splats, inputs.ids_plat, SI( inputs.offsets( t ) ), SI( inputs.comptes( t ) ),
                                  px, py, outputs.image( y = py, x = px ) );
    """,
)


def construire_index_csr( splats, ecran ):
    """L'index en CSR : `( offsets, comptes, ids_plat )`, de taille EXACTE.

    Trois etapes, dont UNE SEULE chose passe par l'hote, et ce n'est pas la somme prefixe :
    `loom.cumsum( ..., exclusive = True )` la fait sur le device, par la primitive de scan du
    backend.

    Ce qui passe par l'hote, c'est `total` -- une lecture d'un compte ECRIT PAR UN NOYAU, pour
    dimensionner la liste plate. C'est le prix de l'exactitude, c'est ce qu'un JIT ne peut pas
    payer, et c'est pour ca que cette variante n'est pas utilisable sous `jit`.
    """
    nb_tuiles = _nb_tuiles( int( ecran.largeur.value ), int( ecran.hauteur.value ), int( ecran.cote.value ) )
    tuile = loom.Axis( loom.ShapeVar( nb_tuiles ), name = "tuile_csr" )

    # 1. compter
    comptes = loom.IntTensor[ tuile ]()
    loom.ffi_call( "splats_compter", _COMPTER,
                   splats = splats, ecran = ecran, comptes = loom.out( comptes ), nb_items = int( splats.nb_splats.value ) )

    # 2. les offsets : la somme prefixe EXCLUSIVE, sur le device. Le resultat porte les MEMES
    #    objets d'axe que `comptes`, donc il est sur la meme grille sans qu'on l'ait dit.
    offsets = loom.cumsum( comptes, exclusive = True )
    #    le TOTAL, lui, doit revenir sur l'hote : c'est lui qui dimensionne la liste plate
    total = int( comptes.sum() )

    # 3. remplir
    fente = loom.Axis( loom.ShapeVar( max( total, 1 ) ), name = "fente_csr" )
    ids_plat = loom.IntTensor[ fente ]()
    curseurs = loom.IntTensor[ tuile ]()
    loom.ffi_call( "splats_remplir", _REMPLIR,
                   splats = splats, ecran = ecran, offsets = offsets,
                   curseurs = loom.out( curseurs ), ids_plat = loom.out( ids_plat ), nb_items = int( splats.nb_splats.value ) )
    return offsets, comptes, ids_plat, total


def rendre_csr( splats, offsets, comptes, ids_plat, ecran ):
    """Le meme rendu, sur l'index CSR. Doit donner la meme image, au bit pres."""
    largeur, hauteur = int( ecran.largeur.value ), int( ecran.hauteur.value )
    image = loom.RealTensor[ loom.Axis( loom.ShapeVar( hauteur ), name = "y" ),
                        loom.Axis( loom.ShapeVar( largeur ), name = "x" ), splats.rvb ]()
    loom.ffi_call(
        "splats_rendre_csr",
        _RENDRE_CSR,
        splats = splats, ecran = ecran, offsets = offsets, comptes = comptes,
        ids_plat = ids_plat, image = loom.out( image ),
        nb_items = largeur * hauteur,
    )
    return image.tensor
