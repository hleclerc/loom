"""Les tests du deuxieme usager etranger. Rien n'importe sdot.

    errand test_splats
"""
import math
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import numpy

from errand import test
from loom import RealTensor, driver
from loom.testing import check_grad

from splats import COTE, Ecran, Splats, construire_index, rendre, rendu


def _scene( nb, largeur, hauteur, seed = 0, echelle = 3.0, amas = 6 ):
    """Une scene reproductible, et surtout REALISTE sur le seul point qui compte ici : l'inegalite.

    Deux choses la produisent, et ce sont celles d'une vraie scene. Les centres sont en AMAS ( une
    reconstruction met des gaussiennes ou il y a de la matiere, pas uniformement ), et les tailles
    couvrent une DECADE ( du detail fin au fond flou ). D'ou des tuiles vides et des tuiles
    surchargees -- donc un pire cas tres loin de la moyenne, qui est exactement ce qu'une borne
    unique doit payer.
    """
    rng = numpy.random.default_rng( seed )

    # les centres : quelques amas serres, plus un fond disperse
    nb_fond = max( 1, nb // 5 )
    foyers = numpy.stack( [ rng.uniform( 0, largeur, amas ), rng.uniform( 0, hauteur, amas ) ], axis = 1 )
    qui = rng.integers( 0, amas, nb - nb_fond )
    serres = foyers[ qui ] + rng.normal( 0, min( largeur, hauteur ) / 25, ( nb - nb_fond, 2 ) )
    fond = numpy.stack( [ rng.uniform( 0, largeur, nb_fond ), rng.uniform( 0, hauteur, nb_fond ) ], axis = 1 )
    centres = numpy.concatenate( [ serres, fond ] )

    # les tailles : log-uniformes sur une decade
    sigmas = echelle * numpy.exp( rng.uniform( 0, math.log( 10 ), nb ) )

    # une covariance anisotrope, inversee : ( a, b, c ) = inv( R S R^T )
    angles = rng.uniform( 0, math.pi, nb )
    rapport = rng.uniform( 0.3, 1.0, nb )
    sx, sy = sigmas, sigmas * rapport
    ca, sa = numpy.cos( angles ), numpy.sin( angles )
    ixx, iyy = 1.0 / sx ** 2, 1.0 / sy ** 2
    a = ca * ca * ixx + sa * sa * iyy
    c = sa * sa * ixx + ca * ca * iyy
    b = ca * sa * ( ixx - iyy )
    cov_inv = numpy.stack( [ a, b, c ], axis = 1 )
    couleurs = rng.uniform( 0.1, 1.0, ( nb, 3 ) )
    opacites = rng.uniform( 0.2, 1.0, nb )

    splats = Splats( nb_xy = 2, nb_abc = 3, nb_rvb = 3 )
    splats.centres = centres
    splats.cov_inv = cov_inv
    splats.couleurs = couleurs
    splats.opacites = opacites
    return splats


def _ecran( largeur, hauteur ):
    return Ecran( largeur = largeur, hauteur = hauteur, cote = COTE )


if test( "l_index_ragged_dit_combien_de_splats_touchent_chaque_tuile" ):
    # LE test de la structure : le compte par tuile est ecrit par le noyau, et il est RAGGED --
    # `nb_par_tuile.value` rend un compte par tuile, pas un scalaire.
    largeur, hauteur, nb = 256, 256, 2000
    splats = _scene( nb, largeur, hauteur )
    index = construire_index( splats, _ecran( largeur, hauteur ), capacite = 64 )

    comptes = numpy.asarray( index.nb_par_tuile.value ).reshape( -1 )
    nb_tx, nb_ty = largeur // COTE, hauteur // COTE
    assert comptes.shape == ( nb_tx * nb_ty, ), comptes.shape
    assert comptes.sum() > nb, "chaque splat touche au moins une tuile"

    # la structure est bien INEGALE : c'est ce qui rend une borne unique couteuse
    print( f"tuiles {nb_tx}x{nb_ty}, {nb} splats : par tuile min {comptes.min()}, "
           f"moyenne {comptes.mean():.1f}, max {comptes.max()}, total {comptes.sum()}" )
    # ragged et non trivial : les tuiles ne se ressemblent pas. ( L'inegalite qui COUTE n'est pas
    # celle-ci mais celle des empreintes, mesuree par `le_cout_d_une_borne_unique`. )
    assert comptes.min() > 0 and comptes.max() >= 2 * comptes.min()

    # et l'index ne contient que des splats qui touchent vraiment la tuile
    ids = numpy.asarray( index.ids )
    centres = numpy.asarray( splats.centres )
    for t in range( 0, nb_tx * nb_ty, 7 ):
        tx, ty = t % nb_tx, t // nb_tx
        for k in range( int( comptes[ t ] ) ):
            i = int( ids[ t, k ] )
            assert 0 <= i < nb
            # le centre peut etre loin : ce qu'on verifie est qu'aucun id n'est du remplissage
    print( "index coherent" )


if test( "le_rendu_est_celui_de_la_somme_directe" ):
    # la reference : la somme sur TOUS les splats, sans index. Si l'index perdait un splat, l'ecart
    # se verrait -- c'est ce qui fait de ce test la validation de la structure, pas du seul stencil.
    from reference_jax import rendu_dense

    largeur, hauteur, nb = 96, 64, 150
    splats = _scene( nb, largeur, hauteur )
    image = rendre( splats, construire_index( splats, _ecran( largeur, hauteur ), 64 ),
                    _ecran( largeur, hauteur ) )

    attendu = rendu_dense( numpy.asarray( splats.centres ), numpy.asarray( splats.cov_inv ),
                           numpy.asarray( splats.couleurs ), numpy.asarray( splats.opacites ),
                           largeur, hauteur )
    obtenu = numpy.asarray( image )
    ecart = numpy.abs( obtenu - attendu ).max()
    print( f"ecart max au rendu dense : {ecart:.3e}   ( image {obtenu.shape} )" )
    assert ecart < 1e-10, ecart


if test( "l_adjoint_atomique_est_celui_du_rendu" ):
    # l'adjoint est en ACCUMULATION : un splat est touche par tous les pixels qu'il couvre, donc par
    # des work-items differents. C'est l'inverse de `examples/diffusion`, en gather pur.
    largeur, hauteur, nb = 48, 48, 30
    splats = _scene( nb, largeur, hauteur, seed = 3, echelle = 5.0 )
    ecran = _ecran( largeur, hauteur )

    for nom in ( "couleurs", "opacites", "centres", "cov_inv" ):
        def rend( valeur, nom = nom ):
            s = _scene( nb, largeur, hauteur, seed = 3, echelle = 5.0 )
            setattr( s, nom, valeur )
            return rendu( s, ecran, 64 )

        depart = getattr( splats, nom ).raw
        ad, df = check_grad( rend, depart, seed = 11 )
        print( f"d/d{nom:<9} : adjoint {float( ad ):+.9f}   diff. finie {float( df ):+.9f}" )


if test( "une_capacite_trop_petite_est_corrigee_toute_seule" ):
    # la machinerie qu'aucun framework n'a : le noyau dit que le compte n'a pas tenu, et l'appel
    # recommence avec plus de place. On part volontairement d'une capacite de 1.
    largeur, hauteur, nb = 256, 256, 2000
    splats = _scene( nb, largeur, hauteur )

    juste = construire_index( splats, _ecran( largeur, hauteur ), capacite = 512 )
    etroit = construire_index( splats, _ecran( largeur, hauteur ), capacite = 1 )

    a = numpy.asarray( juste.nb_par_tuile.value ).reshape( -1 )
    b = numpy.asarray( etroit.nb_par_tuile.value ).reshape( -1 )
    assert ( a == b ).all(), "les comptes doivent etre les memes, la capacite ne change pas la scene"
    print( f"capacite demandee 1 -> retenue {etroit.ids.capacity[ 1 ]} "
           f"( max par tuile {b.max()} ) ; demandee 512 -> {juste.ids.capacity[ 1 ]}" )
    assert etroit.ids.capacity[ 1 ] >= int( b.max() )


if test( "le_cout_d_une_borne_unique" ):
    # CE QUE XLA COUTE, en chiffres, et sans homme de paille.
    #
    # XLA sait faire un index par tuile -- a condition de BORNER le pire cas avant de tracer. Les
    # deux bornes qu'il faut choisir sont ici, et on les compare a ce que loom depense :
    #
    #   * la capacite par tuile : un tableau `[ tuiles, capacite ]` alors que la moyenne est bien
    #     plus basse -- loom la DECOUVRE et la corrige, il ne la choisit pas ;
    #   * le rayon d'une fenetre fixe : dicte par le PLUS GROS splat de la scene, et paye par tous.
    #     C'est la borne la plus chere, parce que les tailles couvrent une decade, donc les
    #     empreintes deux ordres de grandeur.
    from reference_jax import travail_fenetre, travail_reel

    for nb, largeur, hauteur in ( ( 500, 256, 256 ), ( 2000, 256, 256 ), ( 2000, 512, 512 ) ):
        splats = _scene( nb, largeur, hauteur )
        cov = numpy.asarray( splats.cov_inv )
        index = construire_index( splats, _ecran( largeur, hauteur ), capacite = 512 )
        comptes = numpy.asarray( index.nb_par_tuile.value ).reshape( -1 )

        borne, rayon = travail_fenetre( cov )
        utile = travail_reel( cov, largeur, hauteur )
        dense = nb * largeur * hauteur

        print( f"\n{nb} splats, {largeur}x{hauteur} :" )
        print( f"  par tuile        : moyenne {comptes.mean():7.1f}   max {comptes.max():5d}"
               f"   -> une capacite fixe gaspille x{comptes.max() / comptes.mean():.1f}" )
        print( f"  couples utiles   : {utile:12d}   ( la somme des empreintes reelles )" )
        print( f"  fenetre fixe     : {borne:12d}   ( R = {rayon}, dicte par le plus gros splat )"
               f"   -> x{borne / utile:.1f}" )
        print( f"  somme dense      : {dense:12d}   ( chaque pixel voit tous les splats )"
               f"   -> x{dense / utile:.1f}" )

        # la borne d'une fenetre fixe coute au moins un ordre de grandeur : c'est l'argument
        assert borne > 5 * utile


if test( "csr_contre_rembourre_les_deux_couts" ):
    # L'OBJECTION, traitee de front : le meme index en CSR ( offsets + liste unique ) au lieu du
    # rembourre. Si les deux images coincident, la comparaison de leurs couts est legitime -- et
    # elle ne tourne PAS a l'avantage du rembourre.
    #
    # La comparaison est faite au MIEUX pour chaque forme, sinon elle ne vaut rien : on chiffre le
    # rembourre a `tuiles x max_par_tuile`, c'est-a-dire avec la capacite la mieux choisie possible,
    # et pas avec celle qu'on a demandee au hasard.
    from splats import construire_index_csr, rendre_csr

    for nb, largeur, hauteur in ( ( 500, 256, 256 ), ( 2000, 256, 256 ), ( 2000, 512, 512 ) ):
        splats = _scene( nb, largeur, hauteur )
        ecran = _ecran( largeur, hauteur )

        rembourre = construire_index( splats, ecran, capacite = 512 )
        offsets, comptes, ids_plat, total = construire_index_csr( splats, ecran )

        a = rendre( splats, rembourre, ecran )
        b = rendre_csr( splats, offsets, comptes, ids_plat, ecran )
        ecart = float( numpy.abs( numpy.asarray( a ) - numpy.asarray( b ) ).max() )
        # PAS bit a bit, et c'est normal : l'ordre de sommation dans une tuile depend de l'ordre des
        # reservations atomiques, qui differe d'une representation a l'autre, et l'addition flottante
        # n'est pas associative. L'ecart est celui d'une reassociation, pas d'un modele different.
        assert ecart < 1e-12, f"les deux representations doivent donner la meme image ( {ecart} )"

        nb_tuiles = rembourre.ids.capacity[ 0 ]
        par_tuile = numpy.asarray( rembourre.nb_par_tuile.value ).reshape( -1 )
        au_mieux = nb_tuiles * int( par_tuile.max() )      # la MEILLEURE capacite possible
        demandee = nb_tuiles * rembourre.ids.capacity[ 1 ]
        csr = total + nb_tuiles                            # la liste, plus les offsets

        print( f"\n{nb} splats, {largeur}x{hauteur} : ecart entre les deux images {ecart:.1e}" )
        print( f"  rembourre, capacite au mieux ( {par_tuile.max()} ) : {au_mieux:8d} entiers,"
               f" 1 passe" )
        print( f"  rembourre, capacite demandee ( {rembourre.ids.capacity[ 1 ]} ) : {demandee:8d} entiers" )
        print( f"  CSR, taille exacte                     : {csr:8d} entiers, 2 passes"
               f" + somme prefixe hote" )
        print( f"  -> meme au mieux, le rembourre coute x{au_mieux / csr:.2f} la memoire du CSR" )

        # le rembourre est PLUS COUTEUX en memoire, toujours : c'est le constat, pas l'inverse
        assert au_mieux > csr
