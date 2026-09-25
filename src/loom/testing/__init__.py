"""Ce que loom prête à qui l'éprouve -- et plus le harnais, qui est `errand`.

Un fichier de travail déclare ses entrées avec `errand` :

    from errand import test, bench, experiment, Param
    from loom.testing import check_grad          # si besoin

    if test( "my test" ):
        assert 0 == 0

    if p := bench( "my bench", nb_diracs = Param( 1000, help = "nb diracs" ) ):
        p.results[ "cost" ] = run_bench( p.nb_diracs )

Ce module n'héberge plus que ce qui est PROPRE À LOOM : vérifier un gradient. Tout le reste
-- l'enregistrement en deux phases, les paramètres, `p.out_dir`, `result.yaml`, les
environnements, les matrices -- vivait ici par accident d'histoire et vit maintenant dans
`errand`, qui ne sait rien de loom et qu'un autre projet peut donc utiliser.

Ce qui a disparu, et par quoi :

* `test`/`bench`/`experiment`/`Param`/`Args` -> `errand`
* `driver_is( "torch" )`                     -> `errand.has_tag( "driver=torch" )`
* `out_dir()`                                -> `errand.out_dir()`, ou `p.out_dir`
* `info`/`infox`/`new_batch_axis` posés dans `builtins` -> importés comme tout le monde
  ( `from loom.util import info` ). Un nom qui apparaît sans avoir été importé est une
  dette qu'on paie en cherchant d'où il vient.
"""
from .grad_check import check_grad

__all__ = [ "check_grad" ]
