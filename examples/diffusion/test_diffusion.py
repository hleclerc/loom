"""Les tests de l'usager etranger. Rien n'importe sdot ; tout passe par `loom`.

    errand test_diffusion
"""
import math
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import loom
from errand import test
from loom.testing import check_grad

from diffusion import evolution, pas


def _grille( n, f ):
    """Un champ `n x n` donne cellule par cellule. La CLASSE porte le type ( des reels ), donc
    il n'y a rien a declarer de plus -- et `driver` n'a pas a apparaitre."""
    return loom.RealTensor( [ [ float( f( j, i ) ) for i in range( n ) ] for j in range( n ) ] ).raw


def _mode_propre( n ):
    """`sin( pi x ) sin( pi y )` : un vecteur propre EXACT du stencil a cinq points, nul sur le
    bord. Son facteur de decroissance par pas se calcule a la main."""
    return _grille( n, lambda j, i: math.sin( math.pi * i / ( n - 1 ) ) * math.sin( math.pi * j / ( n - 1 ) ) )


def _bosse( n, x0 = 0.35, y0 = 0.4, s = 0.15 ):
    def f( j, i ):
        x, y = i / ( n - 1 ), j / ( n - 1 )
        if i in ( 0, n - 1 ) or j in ( 0, n - 1 ):
            return 0.0
        return math.exp( - ( ( x - x0 ) ** 2 + ( y - y0 ) ** 2 ) / ( 2 * s * s ) )
    return _grille( n, f )


if test( "le_mode_propre_decroit_du_facteur_exact" ):
    # `sin( pi x ) sin( pi y )` est vecteur propre du stencil : un pas doit le multiplier par
    #   1 + coef * ( 4 cos( pi h ) - 4 ),   h = 1 / ( n - 1 )
    # a la precision de la machine. C'est le stencil lui-meme qui est teste, pas une tendance.
    n = 17
    coef = 0.2                                     # dt / h^2, sous la limite de stabilite ( 0.25 )
    u = _mode_propre( n )

    attendu = 1 + coef * ( 4 * math.cos( math.pi / ( n - 1 ) ) - 4 )
    v = pas( u, coef )

    for j in range( 1, n - 1 ):
        for i in range( 1, n - 1 ):
            a, b = float( v[ j ][ i ] ), attendu * float( u[ j ][ i ] )
            assert abs( a - b ) <= 1e-12 + 1e-10 * abs( b ), ( j, i, a, b )

    print( f"mode propre : facteur {attendu:.6f} retrouve sur {(n-2)**2} cellules" )


if test( "le_bord_reste_impose" ):
    # les cellules du bord portent une temperature imposee : un pas ne doit pas y toucher, quoi
    # que fasse l'interieur.
    n = 12
    u = loom.RealTensor[ n, n ].random( seed = 3 ).raw
    v = pas( u, 0.15 )

    for j in range( n ):
        for i in range( n ):
            if j in ( 0, n - 1 ) or i in ( 0, n - 1 ):
                assert float( v[ j ][ i ] ) == float( u[ j ][ i ] ), ( j, i )
    assert any( float( v[ j ][ i ] ) != float( u[ j ][ i ] ) for j in range( 1, n - 1 ) for i in range( 1, n - 1 ) )


if test( "l_adjoint_est_celui_du_solveur" ):
    # LE test qui compte : l'adjoint, confronte a la difference finie, a travers UNE CHAINE de pas
    # ( il doit la remonter ).
    n = 9
    coef = 0.18
    u0 = _bosse( n )

    ad, df = check_grad( lambda u: evolution( u, coef, 3 ), u0, seed = 11 )
    print( f"d/du  : adjoint {float( ad ):+.9f}   diff. finie {float( df ):+.9f}" )


if test( "on_remonte_le_temps" ):
    # CE POUR QUOI on a rendu le solveur derivable : une inversion. On observe la temperature apres
    # `nb_pas` pas de diffusion, et on retrouve l'etat INITIAL par descente de gradient a travers
    # toute la chaine -- le tout compile une fois ( `loom.driver.jit` ).
    n, nb_pas, coef = 14, 6, 0.2

    vrai = _bosse( n )
    observee = evolution( vrai, coef, nb_pas )

    def perte( u ):
        ecart = evolution( u, coef, nb_pas ) - observee
        return ( ecart * ecart ).sum()

    u = loom.RealTensor[ n, n ].zeros().raw
    perte_jit = loom.driver.jit( perte )
    gradient = loom.driver.jit( loom.driver.grad( perte ) )

    # le pas : `evolution` CONTRACTE ( la diffusion ne fait que lisser ), donc les valeurs
    # singulieres de sa jacobienne sont <= 1 et la hessienne de la perte a ses valeurs propres
    # <= 2. Un pas au-dela de ~0.5 diverge -- c'est ce qui rend l'inversion lente, pas un
    # reglage a tatonner.
    depart = float( perte_jit( u ) )
    for _ in range( 400 ):
        u = u - 0.4 * gradient( u )
    arrivee = float( perte_jit( u ) )

    # c'est la PERTE qu'on asserte, pas `u` : remonter le temps est mal pose ( la diffusion efface
    # les hautes frequences, que rien ne peut restituer ), donc sans regularisation on retrouve un
    # etat qui explique les donnees, pas l'etat vrai. Ce qui est teste ici, c'est que le gradient
    # traverse bien les six appels.
    ecart_u = float( ( ( u - vrai ) ** 2 ).sum() ) ** 0.5
    print( f"perte {depart:.3e} -> {arrivee:.3e}   ( x{depart / max( arrivee, 1e-30 ):.0f} ),"
           f"   || u - u_vrai || = {ecart_u:.3f}" )
    assert arrivee < depart / 20


if test( "le_meme_corps_se_batche_sans_le_savoir" ):
    # CE QUE LES AXES ACHETENT. Le stencil est ecrit une fois, sans compter de dimensions :
    # `coords.axes - batch_axes` lui donne ses axes PROPRES, `for_each` les deroule, et
    # `coords + axis` ne decale que l'axe nomme. Un `vmap` ajoute donc un axe SANS que le corps
    # change -- et sans qu'il sache qu'il existe.
    import jax
    import numpy

    n, nb = 8, 3
    rng = numpy.random.default_rng( 0 )
    u = rng.normal( size = ( nb, n, n ) )

    # la reference : un appel par lot, a la main
    ref = numpy.stack( [ numpy.asarray( pas( loom.driver.array( u[ b ] ), 0.1 ) ) for b in range( nb ) ] )

    # le meme, vmape sur le premier axe de `u`
    batche = jax.vmap( lambda uu: pas( uu, 0.1 ), in_axes = 0 )
    got = numpy.asarray( batche( loom.driver.array( u ) ) )

    assert got.shape == ref.shape, ( got.shape, ref.shape )
    ecart = float( numpy.abs( ref - got ).max() )
    assert ecart == 0.0, ecart
    print( f"vmap : { nb } lots, ecart exactement { ecart }" )
