"""Batching an `@aggregate` over extra leading axes.

Batching is a construction-time option of EVERY aggregate, not a distinct type: `Cell` stays
`Cell`. Passing `batch_axes = [ ax, ... ]` to a constructor (e.g. `Cell.make_hypercube`) adds those
axes on the LEFT of every tensor the aggregate declares. The annotations are the "scalar" schema
and stay untouched; `__base_init__` builds each per-instance `Tensor` with the batch axes prepended
(see `util/aggregate.py`).

On the C++ side a batch is nothing but a leading, NAMED tensor dimension, which the existing kernel
machinery already threads transparently (`global_batch_indices`, `cell( batch_index )`, each member
a template parameter carrying its axis-name types). So batching lives entirely on the Python side.

A batch axis is an ordinary `Axis` over its own `ShapeVar`, whose size is PRESCRIBED (a batch extent
is known in Python, so the outputs can be allocated and displayed without a kernel writing a count).
`new_batch_axis( size )` mints a fresh, unshared one; passing the SAME axis object to several
aggregates is how they get JOINED (co-iterated) instead -- which is opt-in, never the default.

= Le NOM d'un axe de batch est une RESSOURCE, pas un compteur

Le nom traverse jusqu'à la source C++ (`DEFINE_AXIS( thread_0 )`, le type `_thread_0` que porte
chaque tenseur batché), et la clé du cache de compilation est le HASH DE CETTE SOURCE. Un nom
frais à chaque appel signifiait donc : deux appels identiques, deux sources différentes, deux
compilations de ~8 s -- et un cache disque qui grossit sans jamais resservir. Ce n'était pas une
inefficacité de détail : ça rendait tout chronométrage d'une boucle d'appels impossible à lire.

Ce dont on a réellement besoin d'un nom, c'est qu'il soit DISTINCT DES AUTRES NOMS VIVANTS -- deux
axes de batch simultanés ne doivent pas se confondre dans `DEFINE_AXIS` ni dans
`global_batch_indices` (`CallArgsAnalysis` les collecte PAR NOM). Rien n'exige qu'il soit distinct
de ceux du passé. Un nom est donc EMPRUNTÉ : pris au plus petit indice libre de son préfixe, rendu
quand l'axe meurt. Une suite d'appels qui se répète retrouve les mêmes noms, donc la même source,
donc le cache.

Le PRÉFIXE sépare les familles (`thread_0` pour les work-items, `cell_0` pour les cellules) : il
rend la source engendrée lisible, et surtout il isole les numérotations -- un axe retenu quelque
part par erreur ne décale plus que sa propre famille. Un axe retenu ne casse d'ailleurs rien : il
garde son nom, le suivant en prend un autre, et on paie une compilation -- jamais un résultat faux.

= Le `gc.collect()` est un FILET, et il ne devrait plus jamais servir

Emprunter suppose de rendre, et rendre suppose que l'axe MEURE. Il ne mourait pas : deux anneaux
de références le retenaient, et un anneau n'est défait que par le ramasse-miettes cyclique.

  * `CallArgsAnalysis <-> CallArg_*` -- l'arbre d'abaissement, dont chaque noeud tensoriel tenait
    son analyse en retour (`CallArg_Tensor._caa`). C'était LE coupable : il retenait aussi les
    agrégats de l'appel, donc leurs tampons device, bien après la fin de l'appel. La référence
    remontante est maintenant faible.
  * `Axis -> coeffs -> ShapeVar -> usages -> résolveur -> Axis` -- les résolveurs de
    `AbstractAxis._register_dense` capturaient leur axe et leur compte par argument par défaut.
    Faibles aussi désormais (la référence vers le TENSEUR l'était déjà).

Vérifié en désactivant le ramasse-miettes cyclique : les axes sont rendus par le simple comptage
de références. Le `gc.collect()` ci-dessous ne se déclenche que si la réserve est vide -- donc au
moment précis où l'on paierait une compilation -- et ne devrait donc plus jamais rien avoir à
rendre. Il reste comme filet : un anneau réintroduit ailleurs coûterait 9 ms au lieu de 9 s, et
`in_use( prefix )` le dirait.

Le remède définitif serait de nommer les axes AU MOMENT DE L'ABAISSEMENT (`CallArgsAnalysis`
connaît l'ensemble exact des axes de CET appel et peut les numéroter dans un ordre déterministe) :
plus de durée de vie à suivre du tout. Ce qui l'en empêche est qu'un axe de batch peut atteindre
un appel par un tenseur NU et pas seulement par les `batch_axes` d'un agrégat, donc il faudrait
une pré-passe qui les trouve tous avant que le premier `CallArg` ne soit construit.
"""
import gc
import threading
import weakref

from .ShapeVar import ShapeVar
from .Axis import Axis


class _NamePool:
    """Les indices EMPRUNTABLES d'un préfixe : le plus petit libre est repris avant d'en créer un.

    Reprendre le plus petit, et non le dernier rendu, est ce qui rend la suite REPRODUCTIBLE :
    deux exécutions du même programme empruntent dans le même ordre, quelle que soit la façon dont
    les emprunts se sont chevauchés.
    """

    # ramasser avant de créer un nom neuf. Coupable : un appelant qui ferait BEAUCOUP d'appels
    # très courts et préférerait payer les compilations. Voir la docstring du module.
    collect_when_empty = True

    def __init__( self ):
        self._lock = threading.Lock()
        self._free = {}                 # prefix -> set des indices rendus
        self._next = {}                 # prefix -> le prochain indice jamais distribué

    def take( self, prefix ):
        index = self._take( prefix )
        if index is not None:
            return index

        # la réserve est vide : c'est ici, et seulement ici, qu'on s'apprête à payer une
        # compilation. Un ramassage cyclique rend les axes des appels précédents (le graphe de
        # shapes est cyclique, ils ne partent pas d'eux-mêmes) -- et s'il n'en rend aucun, c'est
        # qu'ils sont vraiment vivants, et on crée le nom.
        if self.collect_when_empty:
            gc.collect()
            index = self._take( prefix )
            if index is not None:
                return index

        with self._lock:
            index = self._next.get( prefix, 0 )
            self._next[ prefix ] = index + 1
            return index

    def _take( self, prefix ):
        """Le plus petit indice libre, ou `None` s'il n'y en a aucun."""
        with self._lock:
            free = self._free.get( prefix )
            if not free:
                return None
            index = min( free )
            free.discard( index )
            return index

    def give_back( self, prefix, index ):
        with self._lock:
            self._free.setdefault( prefix, set() ).add( index )

    def in_use( self, prefix ):
        """Combien d'indices de ce préfixe sont dehors -- pour les tests, et pour diagnostiquer une
        fuite (un axe retenu quelque part) sans avoir à deviner."""
        with self._lock:
            return self._next.get( prefix, 0 ) - len( self._free.get( prefix, () ) )


_pool = _NamePool()


def new_batch_axis( size, prefix = "batch" ):
    """A fresh batch `Axis` of extent `size`: a private `ShapeVar` prescribed to `size`, wrapped in
    an `Axis` whose name is BORROWED from `prefix`'s pool (see the module docstring: the name is a
    resource, and reusing it is what lets two identical calls hit the compilation cache). Pass a
    list of these as `batch_axes = [ ... ]` to any aggregate constructor; reuse one object across
    aggregates to co-iterate them.

    `prefix` should say what the axis IS (`"thread"`, `"cell"`, `"angle"`), which is what the
    generated C++ will then read like. It also isolates the numbering of one family from another.
    """
    index = _pool.take( prefix )

    axis = Axis( ShapeVar( size ) )
    axis.name = f"{ prefix }_{ index }"
    axis.is_batch = True    # sorts first in the logical layout of an elementwise result

    # rendu quand l'axe meurt -- un `finalize` et non un `__del__` : il ne retient pas l'axe (il ne
    # capture que le préfixe et l'indice), donc il ne repousse pas sa mort d'un cycle de GC.
    weakref.finalize( axis, _pool.give_back, prefix, index )

    return axis
