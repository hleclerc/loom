"""Les tests de l'usager etranger. Rien n'importe sdot ; tout passe par `loom`.

    errand test_diffusion
"""
import math
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

from loom import RealTensor, driver
from errand import test
from loom.testing import check_grad

from diffusion import axes, evolution, pas


def _grille( n, f ):
    """Un champ `n x n` donne cellule par cellule. La CLASSE porte le type ( des reels ), donc
    il n'y a rien a declarer de plus -- et `driver` n'a pas a apparaitre."""
    return RealTensor( [ [ float( f( j, i ) ) for i in range( n ) ] for j in range( n ) ] ).raw


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
    y, x = axes( n )
    u = _mode_propre( n )
    k = RealTensor[ y, x ].ones().raw

    attendu = 1 + coef * ( 4 * math.cos( math.pi / ( n - 1 ) ) - 4 )
    v = pas( u, k, coef )

    for j in range( 1, n - 1 ):
        for i in range( 1, n - 1 ):
            a, b = float( v[ j ][ i ] ), attendu * float( u[ j ][ i ] )
            assert abs( a - b ) <= 1e-12 + 1e-10 * abs( b ), ( j, i, a, b )

    print( f"mode propre : facteur {attendu:.6f} retrouve sur {(n-2)**2} cellules" )


if test( "le_bord_reste_impose" ):
    # les cellules du bord portent une temperature imposee : un pas ne doit pas y toucher, quoi
    # que fasse l'interieur.
    n = 12
    y, x = axes( n )
    u = RealTensor[ y, x ].random( seed = 3 ).raw
    # une diffusivite qui croit vers le coin bas-droit : la somme des deux coordonnees
    k = ( 0.5 + 0.25 * ( RealTensor[ y, x ].linspace( 0, 1, x ) + RealTensor[ y, x ].linspace( 0, 1, y ) ) ).raw
    v = pas( u, k, 0.15 )

    for j in range( n ):
        for i in range( n ):
            if j in ( 0, n - 1 ) or i in ( 0, n - 1 ):
                assert float( v[ j ][ i ] ) == float( u[ j ][ i ] ), ( j, i )
    assert any( float( v[ j ][ i ] ) != float( u[ j ][ i ] ) for j in range( 1, n - 1 ) for i in range( 1, n - 1 ) )


if test( "l_adjoint_est_celui_du_solveur" ):
    # LE test qui compte : les deux adjoints ecrits a la main dans `pas.h`, confrontes a la
    # difference finie, a travers UNE CHAINE de pas ( l'adjoint doit la remonter ).
    n = 9
    coef = 0.18
    u0 = _bosse( n )
    k0 = _grille( n, lambda j, i: 1.0 + 0.3 * math.sin( 2 * i ) * math.cos( 3 * j ) )

    ad, df = check_grad( lambda u: evolution( u, k0, coef, 3 ), u0, seed = 11 )
    print( f"d/du  : adjoint {float( ad ):+.9f}   diff. finie {float( df ):+.9f}" )

    ad, df = check_grad( lambda k: evolution( u0, k, coef, 3 ), k0, seed = 12 )
    print( f"d/dk  : adjoint {float( ad ):+.9f}   diff. finie {float( df ):+.9f}" )


if test( "on_retrouve_la_diffusivite" ):
    # CE POUR QUOI on a rendu le solveur derivable : une inversion. On observe la temperature
    # apres `nb_pas` pas avec une diffusivite inconnue, et on la retrouve par descente de
    # gradient a travers toute la chaine -- le tout compile une fois ( `driver.jit` ).
    n, nb_pas, coef = 14, 10, 0.2
    u0 = _bosse( n )

    vraie = _grille( n, lambda j, i: 1.5 if ( 3 <= i < 8 and 4 <= j < 10 ) else 0.6 )
    observee = evolution( u0, vraie, coef, nb_pas )

    def perte( k ):
        ecart = evolution( u0, k, coef, nb_pas ) - observee
        return ( ecart * ecart ).sum()

    y, x = axes( n )
    k = RealTensor[ y, x ].ones().raw
    perte_jit = driver.jit( perte )
    gradient = driver.jit( driver.grad( perte ) )

    depart = float( perte_jit( k ) )
    for _ in range( 120 ):
        k = k - 3.0 * gradient( k )
    arrivee = float( perte_jit( k ) )

    # c'est la PERTE qu'on asserte, pas `k` : l'inversion est mal posee ( la ou la temperature ne
    # varie pas, `k` n'a aucun effet observable ), donc sans regularisation on retrouve un champ
    # qui explique les donnees, pas le champ vrai. Ce qui est teste ici, c'est que le gradient
    # traverse bien les dix appels.
    ecart_k = float( ( ( k - vraie ) ** 2 ).sum() ) ** 0.5
    print( f"perte {depart:.3e} -> {arrivee:.3e}   ( x{depart / max( arrivee, 1e-30 ):.0f} ),"
           f"   || k - k_vraie || = {ecart_k:.3f}" )
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
    k = numpy.full( ( n, n ), 0.2 )

    # la reference : un appel par lot, a la main
    ref = numpy.stack( [ numpy.asarray( pas( driver.array( u[ b ] ), driver.array( k ), 0.1 ) )
                         for b in range( nb ) ] )

    # le meme, vmape sur le premier axe de `u` ( `k` n'est PAS mappe : il traverse tel quel )
    batche = jax.vmap( lambda uu: pas( uu, driver.array( k ), 0.1 ), in_axes = 0 )
    got = numpy.asarray( batche( driver.array( u ) ) )

    assert got.shape == ref.shape, ( got.shape, ref.shape )
    ecart = float( numpy.abs( ref - got ).max() )
    assert ecart == 0.0, ecart
    print( f"vmap : { nb } lots, ecart exactement { ecart }" )
