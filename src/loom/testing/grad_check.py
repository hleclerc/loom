"""Vérificateur de dérivées agnostique (Jax aujourd'hui, Torch demain).

`check_grad` compare la dérivée d'une fonction, obtenue par le mode adjoint du
driver (`driver.vjp`), à son estimation par différence finie centrée. Rien n'est
spécifique à un framework : tout passe par `driver.vjp` / `driver.random` et par
l'arithmétique des tenseurs (`+`, `*`, `.sum()`), commune à Jax et Torch -- voir
la note d'archi sur les tests agnostiques.

Principe : on ne matérialise pas la jacobienne complète, on la teste sur des
projections aléatoires. Avec une tangente `v` en entrée et une cotangente `w` en
sortie, l'adjoint donne exactement `< vjp(w), v > = < w, J v >`, et le membre de
droite est estimé par `( f(x+εv) - f(x-εv) ) / 2ε`. Un désaccord signale une
dérivée fausse (un adjoint nul le fait ressortir immédiatement).

`f` et ses arguments s'expriment en `Tensor` : un `Tensor` en entrée est dérivé
par rapport à son buffer, un `Tensor` en sortie est comparé sur sa vue dense
(`.tensor`) -- le padding de capacité est retiré pour nous, sans écrire de
`.raw[ :n ]`.
"""
from loom.tensor import Tensor
from loom import driver


def _raw( x ):
    """Le buffer différentiable derrière `x` : celui d'un `Tensor`, ou `x` tel quel."""
    return x.raw if isinstance( x, Tensor ) else x


def check_grad( f, *args, eps = 1e-4, rtol = 2e-3, atol = 1e-4 ):
    """Vérifie la dérivée de `f` par différence finie.

    `f` prend un ou plusieurs `Tensor` (ou tenseurs bruts du driver) et renvoie un `Tensor`
    (ou un tenseur brut). Lève une `AssertionError` si l'adjoint et la différence finie
    s'écartent de plus de `atol + rtol * |num|`. Renvoie le couple ( adjoint, diff. finie ).
    """
    # La différence finie centrée amplifie le bruit d'arrondi (~machine_eps / eps) : en FP32
    # (machine_eps ~1.2e-7) avec eps=1e-4 ça reste marginal, mais suffisant pour noyer un vrai
    # écart de la taille de `tol`. On force FP64 pour la durée du check, puis on restaure la
    # valeur précédente pour ne pas la faire fuiter sur les tests suivants (driver est un
    # singleton global partagé par tout le process de test).
    previous_ftype = driver.ftype
    driver.ftype = "FP64"
    try:
        primals = [ _raw( a ) for a in args ]

        # `f` renvoie en général un `Tensor` : c'est sa vue DENSE qu'on compare (le padding de
        # capacité n'est pas une vraie sortie). L'étendue d'un axe écrit par le kernel est une valeur
        # DEVICE sous une trace ; on capture donc la forme dense maintenant, à l'exécution eager, en
        # entiers Python -- le rognage par trace devient alors statique, donc compatible avec la trace.
        probe = f( *primals )
        if isinstance( probe, Tensor ):
            dense_shape = tuple( probe.shape )
            crop  = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
            out_f = lambda *r: crop( f( *r ) )
        else:
            out_f = f

        out, pullback = driver.vjp( out_f, *primals )

        # cotangente aléatoire en sortie, tangentes aléatoires en entrée
        w  = driver.random( out.shape )
        vs = [ driver.random( p.shape ) for p in primals ]

        # adjoint : < vjp(w), v >, sommé sur les entrées
        grads = pullback( w )
        ana = sum( float( ( g * v ).sum() ) for g, v in zip( grads, vs ) )

        # différence finie centrée : < w, ( f(x+εv) - f(x-εv) ) / 2ε >
        plus  = out_f( *[ p + eps * v for p, v in zip( primals, vs ) ] )
        minus = out_f( *[ p - eps * v for p, v in zip( primals, vs ) ] )
        num = float( ( ( plus - minus ) * w ).sum() ) / ( 2 * eps )
    finally:
        driver.ftype = previous_ftype

    err = abs( ana - num )
    tol = atol + rtol * abs( num )
    assert err <= tol, (
        f"dérivée incorrecte : adjoint = { ana }, diff. finie = { num }, "
        f"|Δ| = { err } > { tol }"
    )
    return ana, num
