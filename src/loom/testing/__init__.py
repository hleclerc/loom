"""What loom lends to whoever puts it to the test -- and no longer the harness, which is `errand`.

A working file declares its entries with `errand`:

    from errand import test, bench, experiment, Param
    from loom.testing import check_grad          # if needed

    if test( "my test" ):
        assert 0 == 0

    if p := bench( "my bench", nb_diracs = Param( 1000, help = "nb diracs" ) ):
        p.results[ "cost" ] = run_bench( p.nb_diracs )

This module now hosts only what is SPECIFIC TO LOOM: checking a gradient. Everything else
-- the two-phase registration, the parameters, `p.out_dir`, `result.yaml`, the
environments, the matrices -- lived here by historical accident and now lives in
`errand`, which knows nothing about loom and which another project can therefore use.

What disappeared, and what replaces it:

* `test`/`bench`/`experiment`/`Param`/`Args` -> `errand`
* `driver_is( "torch" )`                     -> `errand.has_tag( "driver=torch" )`
* `out_dir()`                                -> `errand.out_dir()`, or `p.out_dir`
* `info`/`infox`/`new_batch_axis` injected into `builtins` -> imported like everything else
  ( `from loom.util import info` ). A name that appears without having been imported is a
  debt paid by searching for where it comes from.
"""
from .grad_check import check_grad
from .need import need, need_autodiff
from .host import host

__all__ = [ "check_grad", "need", "need_autodiff", "host" ]
