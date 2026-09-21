"""Les noyaux MULTI-FICHIERS : `FfiCode( sources = [ ( "x.cpp", { "DEF": ... } ) ] )`.

Une source de domaine est compilée une fois par (source, defines, compilateur) en un `.o` que
tous les noyaux qui la nomment lient -- le code qu'on ne veut pas réinstancier dans chaque
unité générée (une densité, une dimension : des macros choisissent). Ce test vérifie que les
defines font bien des unités DISTINCTES, et que le graphe (ninja) les réutilise.
"""
from pathlib import Path
import numpy

from loom import driver
from loom.compilation.FfiCode import FfiCodeParallel
from loom.tensor import Axis, IntTensor, ShapeVar
from loom.testing import test

HERE = Path( __file__ ).resolve().parent / "cpp_sources"


def _scaled( scale ):
    num = Axis( ShapeVar( 3 ), name = "num" )
    res = IntTensor[ num ]()
    driver.call(
        FfiCodeParallel( name = f"test_sources_scaled_{ scale }",
            fwd_code = "for ( int i = 0; i < 3; ++i ) res( i ) = scaled( i + 1 );",
            includes = [ str( HERE / "scaled.h" ) ],
            sources = [ ( str( HERE / "scaled.cpp" ), { "SCALE": str( scale ) } ) ] ),
        output_attributes = [ "res" ],
        res = res,
    )
    return numpy.asarray( res.tensor ).reshape( -1 ).tolist()


if test( "a_source_compiled_with_a_define_is_linked_in" ):
    assert _scaled( 2 ) == [ 2, 4, 6 ]


if test( "another_define_is_another_unit" ):
    assert _scaled( 3 ) == [ 3, 6, 9 ]
    assert _scaled( 2 ) == [ 2, 4, 6 ]
