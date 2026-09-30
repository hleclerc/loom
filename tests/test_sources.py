"""Les noyaux MULTI-FICHIERS : `FfiCode.per_item( sources = [ ( "x.cpp", { "DEF": ... } ) ] )`.

Une source de domaine est compilée une fois par (source, defines, compilateur) en un `.o` que
tous les noyaux qui la nomment lient -- le code qu'on ne veut pas réinstancier dans chaque
unité générée (une densité, une dimension : des macros choisissent). Ce test vérifie que les
defines font bien des unités DISTINCTES, et que le graphe (ninja) les réutilise.

ELLE EST COMPILÉE EN CODE HÔTE, et c'est ce que ce test doit exercer. Une unité séparée n'est pas
du code device : l'appeler depuis un corps `HD` demanderait la compilation device séparée de CUDA
(`-rdc=true` des deux côtés, puis `-dlink`), que loom ne fait pas -- et que personne ne lui
demande. Le seul usager réel de cette facilité, `sdot/otplan/Lineaire.cpp`, l'appelle depuis le
HANDLER, qui est du code hôte.

Ce test le faisait depuis un corps `per_item`, donc depuis le device : il compilait sur CPU et
échouait sur CUDA (« calling a __host__ function from a __host__ __device__ function »), ce qui
n'était pas un défaut de loom mais du test. Il est écrit sur l'usage réel : `scaled` est appelée
dans `kernel`, et le foncteur emporte le résultat par CAPTURE.
"""
from pathlib import Path
import numpy

import loom
from loom import driver
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, IntTensor, ShapeVar
from errand import test

HERE = Path( __file__ ).resolve().parent / "cpp_sources"


_CODE = """
    namespace {
        /// ce que l'HOTE a calcule, emporte par valeur jusqu'au device
        struct Poser {
            SI v[ 3 ];
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.res( coords ) = v[ coords[ num ] ];
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            // `scaled` vit dans une unite compilee A PART ( `sources = ...` ), en code HOTE.
            // `kernel` EST du code hote : c'est ici qu'on l'appelle, exactement comme
            // `OtPlan` appelle `otplan::resoudre`.
            queue.run_parallel( Poser{ { sdot::scaled( 1 ), sdot::scaled( 2 ), sdot::scaled( 3 ) } },
                                args.outputs.res.domain(), args, batch_axes );
        }
    }
"""


def _scaled( scale ):
    num = Axis( ShapeVar( 3 ), name = "num" )
    res = IntTensor[ num ]()
    loom.ffi_call(
        f"test_sources_scaled_{ scale }",
        FfiCode( code = _CODE,
            includes = [ str( HERE / "scaled.h" ) ],
            sources = [ ( str( HERE / "scaled.cpp" ), { "SCALE": str( scale ) } ) ] ),
        res = loom.out( res ),
    )
    return numpy.asarray( res.value ).reshape( -1 ).tolist()


if test( "a_source_compiled_with_a_define_is_linked_in" ):
    assert _scaled( 2 ) == [ 2, 4, 6 ]


if test( "another_define_is_another_unit" ):
    assert _scaled( 3 ) == [ 3, 6, 9 ]
    assert _scaled( 2 ) == [ 2, 4, 6 ]
