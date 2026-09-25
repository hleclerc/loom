"""COMBIEN de noyaux ce process a compilés, et POURQUOI chacun était neuf.

Un corps de noyau ne donne pas un binaire : il en donne un PAR SOURCE ENGENDRÉE, et la source
dépend de l'appel -- les extents figés dans le type, les axes de batch, et surtout, au backward,
quels arguments sont perturbés (un `NoneTensor` plutôt qu'un tampon) et quelles cotangentes sont
des zéros symboliques (un `ZeroTensor`). C'est voulu : le `if constexpr` du corps fait alors
tomber des termes entiers, et un tampon qui n'existe pas n'est pas alloué.

Mais ça se paie, en secondes de compilation, et RIEN NE LE DISAIT. Le seul moyen de s'apercevoir
qu'un test de deux dérivées avait fabriqué dix noyaux était de compter des lignes de `ninja` dans
un log. D'où ce journal : l'instrument vient avant le bouton -- on ne règle pas une spécialisation
qu'on ne mesure pas, et le bon réglage n'est sûrement pas le même pour de la géométrie et pour une
boucle d'entraînement.

    LOOM_JOURNAL=1        imprime le rapport à la fin du process

    from loom.compilation.journal import report, stats
    print( report() )     à la main, quand on veut

Le rapport groupe par NOM de code, et pour un nom qui a plusieurs variantes, il dit ce qui change
d'une variante à la première -- pas « 10 noyaux », mais « `grad_for_cell.data` : none -> out ».
"""
import atexit
import os


# une entrée par noyau DISTINCT (par cible), dans l'ordre où ils sont apparus
_kernels = []
# combien de fois une cible déjà connue a resservi -- le dénominateur qui dit si le cache marche
_reuses = 0


def record( target, code_name, signature, outcome, seconds = 0.0 ):
    """Un noyau distinct de plus. `outcome` : `"compiled"`, `"catalogue"`.

    `signature` est ce qui décrit l'APPEL (pas la source) : un dict lisible, dont la différence
    avec celle d'une autre variante est la réponse à « pourquoi celle-ci est-elle neuve ? »."""
    _kernels.append( dict( target = target, code_name = code_name or "?",
                           signature = dict( signature or {} ), outcome = outcome,
                           seconds = float( seconds ) ) )


def record_reuse():
    """Une cible déjà chargée, resservie telle quelle."""
    global _reuses
    _reuses += 1


def stats():
    """`( noyaux distincts, compilés, pris au catalogue, réutilisations, secondes )`."""
    return dict(
        kernels    = len( _kernels ),
        compiled   = sum( 1 for k in _kernels if k[ "outcome" ] == "compiled" ),
        catalogue  = sum( 1 for k in _kernels if k[ "outcome" ] == "catalogue" ),
        reuses     = _reuses,
        seconds    = sum( k[ "seconds" ] for k in _kernels ),
    )


def _differences( reference, other ):
    """Les clés où deux signatures diffèrent, en `clé : avant -> après`."""
    res = []
    for key in sorted( set( reference ) | set( other ) ):
        before, after = reference.get( key, "-" ), other.get( key, "-" )
        if before != after:
            res.append( f"{ key } : { before } -> { after }" )
    return res


def report():
    """Le rapport, en texte. Vide s'il n'y a rien eu à compiler."""
    if not _kernels:
        return "loom : aucun noyau compilé."

    st = stats()
    lines = [ f"loom : { st[ 'kernels' ] } noyau(x) distinct(s) "
              f"({ st[ 'compiled' ] } compilé(s) en { st[ 'seconds' ]:.1f} s, "
              f"{ st[ 'catalogue' ] } au catalogue), { st[ 'reuses' ] } réutilisation(s)." ]

    by_name = {}
    for k in _kernels:
        by_name.setdefault( k[ "code_name" ], [] ).append( k )

    for name, variants in sorted( by_name.items(), key = lambda kv: -len( kv[ 1 ] ) ):
        total = sum( v[ "seconds" ] for v in variants )
        lines.append( f"\n  { name } : { len( variants ) } variante(s), { total:.1f} s" )
        if len( variants ) == 1:
            continue
        # ce qui sépare chaque variante de la PREMIÈRE : la réponse à « pourquoi neuve ? »
        reference = variants[ 0 ][ "signature" ]
        for index, v in enumerate( variants[ 1 : ], start = 1 ):
            diffs = _differences( reference, v[ "signature" ] )
            lines.append( f"    [{ index }] " + ( "; ".join( diffs ) if diffs
                                                  else "rien de visible ici (une source qui diffère "
                                                       "autrement : corps, includes, compilateur)" ) )
    return "\n".join( lines )


def _print_at_exit():
    value = os.environ.get( "LOOM_JOURNAL", "" ).strip().lower()
    if value in ( "", "0", "false", "no", "off" ):
        return
    if _kernels:
        print( report() )


atexit.register( _print_at_exit )
