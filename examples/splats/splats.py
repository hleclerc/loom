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

import loom.compilation as compilation

# le C++ de CE paquet, enregistre aupres de loom comme n'importe quel usager
compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )

from loom import Aggregate, Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, driver, stop_gradient
from loom.compilation.FfiCode import FfiCode
from loom.tensor import new_batch_axis

COTE = 16          # une tuile de 16 x 16 pixels


class Splats( Aggregate ):
    """Les gaussiennes : centre, inverse de covariance ( a, b, c ), couleur, opacite."""
    centres   : RealTensor[ "splat", "xy" ]
    cov_inv   : RealTensor[ "splat", "abc" ]
    couleurs  : RealTensor[ "splat", "rvb" ]
    opacites  : RealTensor[ "splat" ]

    splat     : Axis[ "nb_splats" ]
    xy        : Axis[ "nb_xy" ]
    abc       : Axis[ "nb_abc" ]
    rvb       : Axis[ "nb_rvb" ]

    nb_splats : ShapeVar
    nb_xy     : CtShapeVar
    nb_abc    : CtShapeVar
    nb_rvb    : CtShapeVar


class Index( Aggregate ):
    """L'INDEX RAGGED : pour chaque tuile, les splats qui la touchent.

    `nb_par_tuile : ShapeVar[ "tuile" ]` est UN COMPTE PAR TUILE -- un `ShapeVar` qui varie le long
    d'un axe ( ses `dep_axes` ). C'est la declaration d'un ragged en loom, et c'est une capacite que
    son seul usager reel n'exerce nulle part : `grep dep_axes sdot/src` ne rend rien.

    Le compte est ECRIT PAR LE NOYAU ( passe 1 ) ; `ids` est alloue a la capacite que l'appel
    demande, et un compte qui la depasse est signale au lieu d'etre tronque en silence.
    """
    ids          : IntTensor[ "tuile", "fente" ]

    tuile        : Axis[ "nb_tuiles" ]
    fente        : Axis[ "nb_par_tuile" ]

    nb_tuiles    : CtShapeVar
    nb_par_tuile : ShapeVar[ "tuile" ]


class Ecran( Aggregate ):
    """La geometrie de l'image, connue a la compilation : le stencil de tuiles y gagne, au prix
    d'une compilation par resolution ( comme la grille de `examples/diffusion` )."""
    largeur : CtShapeVar
    hauteur : CtShapeVar
    cote    : CtShapeVar


class Rangs( Aggregate ):
    """« Qui suis-je ? » : le rang plat de l'item.

    Le meme agregat-pretexte que dans `examples/diffusion`, et pour la meme raison -- l'echafaudage
    injecte `batch_index`, `thread_index` et `nb_threads`, mais pas le rang plat, et le batch d'un
    appel ne vient que des agregats. Deuxieme exemple, meme friction : elle merite que
    l'echafaudage l'injecte."""
    rang : IntTensor


def _nb_tuiles( largeur, hauteur, cote = COTE ):
    return ( ( largeur + cote - 1 ) // cote ) * ( ( hauteur + cote - 1 ) // cote )


_INSCRIRE = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI i = SI( rangs.rang( batch_index ) );
        splats::inscrire( splats, i, SI( ecran.largeur ), SI( ecran.hauteur ), SI( ecran.cote ),
                          index.nb_par_tuile, index.ids );
    """,
)

_RENDRE = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = SI( rangs.rang( batch_index ) );
        const SI largeur = SI( ecran.largeur ), cote = SI( ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        splats::rendre_pixel( splats, index.ids, SI( index.nb_par_tuile( t ) ), t, px, py,
                              image( y = py, x = px ) );
    """,
)

_RENDRE_BWD = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = SI( rangs.rang( batch_index ) );
        const SI largeur = SI( ecran.largeur ), cote = SI( ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        if constexpr ( ! CT_VALUE( grad_for_image.surely_null() ) )
            splats::rendre_pixel_bwd( splats, index.ids, SI( index.nb_par_tuile( t ) ), t, px, py,
                                      grad_for_image( y = py, x = px ), grad_for_splats );
    """,
)


def _rangs( nb, prefixe ):
    """Un agregat batche sur `nb` items, portant leur rang plat."""
    axe = new_batch_axis( nb, prefix = prefixe )
    rangs = Rangs( batch_axes = [ axe ] )
    rangs.rang = IntTensor[ axe ].iota()
    return rangs


def construire_index( splats, ecran, capacite ):
    """PASSE 1 : l'index ragged. NON differentiable ( il rend des entiers, et son lien aux centres
    est discontinu -- un splat entre ou n'entre pas dans une tuile ).

    `capacite` est une DEVINETTE : combien de splats par tuile au plus. Si elle est trop petite, le
    noyau ecrit quand meme le compte VOULU et signale le depassement, et `driver.call` reserve plus
    grand et relance tout seul -- l'appelant n'a rien a faire. La capacite finalement retenue se lit
    dans `index.ids.capacity`, ce qui est la façon d'observer que la croissance a eu lieu.
    """
    index = Index( nb_tuiles = _nb_tuiles( int( ecran.largeur.raw ), int( ecran.hauteur.raw ),
                                           int( ecran.cote.raw ) ) )
    driver.call(
        _INSCRIRE,
        name = "splats_inscrire",
        splats = splats,
        ecran = ecran,
        index = index,
        rangs = _rangs( int( splats.nb_splats.value ), "splat" ),
        output_attributes = [ "index" ],
        output_capacities = { "index.nb_par_tuile": capacite },
    )
    return index


def rendre( splats, index, ecran ):
    """PASSE 2 : l'image. Differentiable par rapport a tout ce que porte `splats`."""
    largeur, hauteur = int( ecran.largeur.raw ), int( ecran.hauteur.raw )
    image = RealTensor[ Axis( ShapeVar( hauteur ), name = "y" ),
                        Axis( ShapeVar( largeur ), name = "x" ), splats.rvb ]()
    driver.call(
        _RENDRE, _RENDRE_BWD,
        name = "splats_rendre",
        splats = splats,
        index = index,
        ecran = ecran,
        image = image,
        rangs = _rangs( largeur * hauteur, "pixel" ),
        output_attributes = [ "image" ],
    )
    return image.raw


def rendu( splats, ecran, capacite ):
    """Le rendu de bout en bout, differentiable : l'index est construit sur des centres dont le
    gradient est COUPE ( l'affectation d'un splat a une tuile est discontinue, et ce n'est pas par
    la que passe la derivee -- c'est aussi ce que fait un 3DGS ), puis l'image est rendue."""
    index = construire_index( _sans_gradient( splats ), ecran, capacite )
    return rendre( splats, index, ecran )


def _sans_gradient( splats ):
    """Les memes splats, detaches : ce que la passe 1 lit."""
    autre = Splats( nb_xy = 2, nb_abc = 3, nb_rvb = 3 )
    autre.centres  = stop_gradient( splats.centres )
    autre.cov_inv  = stop_gradient( splats.cov_inv )
    autre.couleurs = stop_gradient( splats.couleurs )
    autre.opacites = stop_gradient( splats.opacites )
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

_COMPTER = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        splats::compter( splats, SI( rangs.rang( batch_index ) ), SI( ecran.largeur ),
                         SI( ecran.hauteur ), SI( ecran.cote ), comptes );
    """,
)

_REMPLIR = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        splats::remplir( splats, SI( rangs.rang( batch_index ) ), SI( ecran.largeur ),
                         SI( ecran.hauteur ), SI( ecran.cote ), offsets, curseurs, ids_plat );
    """,
)

_RENDRE_CSR = FfiCode.per_item(
    includes = [ "splats/rendu.h" ],
    code = """
        const SI p = SI( rangs.rang( batch_index ) );
        const SI largeur = SI( ecran.largeur ), cote = SI( ecran.cote );
        const SI px = p % largeur, py = p / largeur;
        const SI t = ( py / cote ) * ( ( largeur + cote - 1 ) / cote ) + ( px / cote );

        splats::rendre_pixel_csr( splats, ids_plat, SI( offsets( t ) ), SI( comptes( t ) ),
                                  px, py, image( y = py, x = px ) );
    """,
)


def construire_index_csr( splats, ecran ):
    """L'index en CSR : `( offsets, comptes, ids_plat )`, de taille EXACTE.

    Trois etapes, dont une sur l'hote. La somme prefixe se fait en numpy parce qu'elle porte sur un
    vecteur de la taille du nombre de tuiles ( 256 ici ) : ce n'est pas la ou est le travail, et la
    faire sur le device demanderait un noyau de scan pour rien.

    NON utilisable sous `jit` : lire `total` est une lecture hote d'un compte ecrit par un noyau.
    C'est le prix de l'exactitude, et c'est exactement ce qu'un JIT ne peut pas payer.
    """
    import numpy

    nb_tuiles = _nb_tuiles( int( ecran.largeur.raw ), int( ecran.hauteur.raw ), int( ecran.cote.raw ) )
    tuile = Axis( ShapeVar( nb_tuiles ), name = "tuile_csr" )
    rangs = _rangs( int( splats.nb_splats.value ), "splat" )

    # 1. compter
    comptes = IntTensor[ tuile ]()
    driver.call( _COMPTER, name = "splats_compter",
                 splats = splats, ecran = ecran, comptes = comptes, rangs = rangs,
                 output_attributes = [ "comptes" ] )

    # 2. la somme prefixe, sur l'hote, et le TOTAL -- qui dimensionne la liste
    c = numpy.asarray( comptes.tensor ).reshape( -1 )
    total = int( c.sum() )
    offsets = IntTensor[ tuile ]( numpy.concatenate( [ [ 0 ], numpy.cumsum( c )[ :-1 ] ] ) )

    # 3. remplir
    fente = Axis( ShapeVar( max( total, 1 ) ), name = "fente_csr" )
    ids_plat = IntTensor[ fente ]()
    curseurs = IntTensor[ tuile ]()
    driver.call( _REMPLIR, name = "splats_remplir",
                 splats = splats, ecran = ecran, offsets = offsets, curseurs = curseurs,
                 ids_plat = ids_plat, rangs = rangs,
                 output_attributes = [ "ids_plat", "curseurs" ] )
    return offsets, comptes, ids_plat, total


def rendre_csr( splats, offsets, comptes, ids_plat, ecran ):
    """Le meme rendu, sur l'index CSR. Doit donner la meme image, au bit pres."""
    largeur, hauteur = int( ecran.largeur.raw ), int( ecran.hauteur.raw )
    image = RealTensor[ Axis( ShapeVar( hauteur ), name = "y" ),
                        Axis( ShapeVar( largeur ), name = "x" ), splats.rvb ]()
    driver.call(
        _RENDRE_CSR,
        name = "splats_rendre_csr",
        splats = splats, ecran = ecran, offsets = offsets, comptes = comptes,
        ids_plat = ids_plat, image = image,
        rangs = _rangs( largeur * hauteur, "pixel" ),
        output_attributes = [ "image" ],
    )
    return image.raw
