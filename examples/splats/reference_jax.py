"""Les references JAX : ce qu'un usager competent ecrirait SANS loom, et ce que ca coute.

L'argument de l'exemple serait malhonnete avec un homme de paille. XLA n'est pas incapable de faire
ceci -- il est incapable de le faire SANS BORNER LE PIRE CAS D'AVANCE. Les deux references ci-dessous
sont donc les deux formes correctes qu'on ecrit vraiment, et ce qu'on mesure est le prix de la borne,
pas une maladresse.

  `rendu_dense`   : la somme sur TOUS les splats pour chaque pixel. Simple, exacte, et c'est la
                    premiere chose qu'on ecrit. Coute `pixels x splats` -- elle sert de VERITE dans
                    les tests, et de repere d'echelle.

  `rendu_fenetre` : chaque splat n'ecrit que dans une fenetre de `R x R` pixels autour de son
                    centre, par `scatter-add`. C'est la bonne implementation XLA, et elle est
                    rapide. Mais `R` est un ENTIER DE LA FORME, donc il doit couvrir le PLUS GROS
                    splat de la scene : tout le monde paie l'empreinte du pire. C'est exactement la
                    borne dont loom se passe.

Ce que loom fait a la place : le noyau ecrit le compte par tuile, et la capacite qui n'a pas tenu
est signalee et reprise. Le cout est la somme reelle des empreintes, pas `n x pire^2`.
"""
import numpy

RAYON_SIGMA = 3.0          # le meme seuil que `include/splats/rendu.h`


def _poids( centres, cov_inv, opacites, xs, ys ):
    """Les poids `opacite * exp( -q/2 )`, tronques au-dela de `RAYON_SIGMA`, pour une grille de
    points donnee. `xs`, `ys` de forme `( ..., )` diffusables contre `( n, 1... )`."""
    dx = xs - centres[ :, 0 ].reshape( -1, *( [ 1 ] * ( xs.ndim ) ) )
    dy = ys - centres[ :, 1 ].reshape( -1, *( [ 1 ] * ( ys.ndim ) ) )
    a = cov_inv[ :, 0 ].reshape( -1, *( [ 1 ] * xs.ndim ) )
    b = cov_inv[ :, 1 ].reshape( -1, *( [ 1 ] * xs.ndim ) )
    c = cov_inv[ :, 2 ].reshape( -1, *( [ 1 ] * xs.ndim ) )
    q = a * dx * dx + 2 * b * dx * dy + c * dy * dy
    # la MEME troncature continue que `include/splats/rendu.h` : on retranche la valeur au seuil,
    # sinon ce test comparerait deux modeles differents
    seuil = numpy.exp( -0.5 * RAYON_SIGMA ** 2 )
    w = opacites.reshape( -1, *( [ 1 ] * xs.ndim ) ) * ( numpy.exp( -0.5 * q ) - seuil )
    return numpy.where( q > RAYON_SIGMA ** 2, 0.0, w )


def rendu_dense( centres, cov_inv, couleurs, opacites, largeur, hauteur ):
    """La verite : chaque pixel somme sur tous les splats. `pixels x splats` operations, et un
    tableau intermediaire `[ n, hauteur, largeur ]` -- 500 Mo pour 2000 splats en 256x256."""
    ys, xs = numpy.mgrid[ 0:hauteur, 0:largeur ].astype( float )
    w = _poids( centres, cov_inv, opacites, xs, ys )                 # [ n, h, l ]
    return numpy.einsum( "nhl,nc->hlc", w, couleurs )


def rayon_requis( cov_inv ):
    """Le demi-cote `R` qu'une fenetre fixe doit avoir pour contenir le PLUS GROS splat.

    C'est le nombre que la version XLA doit choisir avant de tracer, et il est dicte par un seul
    splat -- d'ou son coût : `n x ( 2R+1 )^2` pour tout le monde."""
    a, b, c = cov_inv[ :, 0 ], cov_inv[ :, 1 ], cov_inv[ :, 2 ]
    det = a * c - b * b
    demi_x = RAYON_SIGMA * numpy.sqrt( c / det )
    demi_y = RAYON_SIGMA * numpy.sqrt( a / det )
    return int( numpy.ceil( max( demi_x.max(), demi_y.max() ) ) )


def travail_fenetre( cov_inv ):
    """Le nombre de couples ( splat, pixel ) que la version a fenetre fixe traite."""
    n = cov_inv.shape[ 0 ]
    r = rayon_requis( cov_inv )
    return n * ( 2 * r + 1 ) ** 2, r


def travail_reel( cov_inv, largeur, hauteur ):
    """Le nombre de couples ( splat, pixel ) reellement utiles : la somme des empreintes."""
    a, b, c = cov_inv[ :, 0 ], cov_inv[ :, 1 ], cov_inv[ :, 2 ]
    det = a * c - b * b
    demi_x = RAYON_SIGMA * numpy.sqrt( c / det )
    demi_y = RAYON_SIGMA * numpy.sqrt( a / det )
    return int( numpy.sum( ( 2 * demi_x + 1 ) * ( 2 * demi_y + 1 ) ) )
