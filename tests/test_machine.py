"""`Machine` : ce qu'un noyau peut savoir de la machine, sous des noms qui ne parlent d'aucune
machine.

Ça manquait pour écrire un noyau portable : la géométrie de lancement était soit codée en dur
( `const int block = 128` dans `CudaQueue.h` ), soit devinée en Python. Un noyau qui veut choisir sa
taille de groupe ou son budget de mémoire partagée doit pouvoir les DEMANDER.

Quatre champs, et chacun a un sens des deux côtés -- c'est ce qui permet d'écrire le noyau une fois.
Le test vérifie qu'ils traversent jusque dans un kernel et qu'ils sont plausibles, puis les
contraintes propres à chaque device.
"""
from pathlib import Path

from loom import Axis, ShapeVar, IntTensor, driver, compilation
from loom.compilation.FfiCode import FfiCode
from errand import test

compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )

_CHAMPS = ( "nb_workers", "sub_group_width", "local_mem_bytes", "suggested_group" )


def _machine():
    """Les quatre champs, tels que le NOYAU les voit ( et non tels que Python les devinerait )."""
    champs = IntTensor[ Axis( ShapeVar( len( _CHAMPS ) ), name = "num_champ" ) ]()
    driver.call(
        FfiCode( preamble = '#include "loom_tests/machine.h"',
                 code = "loom_tests::poser_machine( args );" ),
        name = "test_machine", champs = champs, output_attributes = [ "champs" ] )
    return dict( zip( _CHAMPS, ( int( v ) for v in champs.raw.tolist() ) ) )


if test( "les_quatre_champs_traversent" ):
    m = _machine()
    print( f"{ driver.device } : " + "  ".join( f"{ k }={ v }" for k, v in m.items() ) )

    # rien ne doit être nul : un noyau portable divise par `sub_group_width` ou dimensionne sur
    # `local_mem_bytes`, et un zéro ferait exploser du code par ailleurs correct.
    for nom, valeur in m.items():
        assert valeur >= 1, ( nom, valeur )

    # un groupe ne peut pas dépasser ce qui peut être en vol
    assert m[ "suggested_group" ] <= m[ "nb_workers" ], m


if test( "ce_que_chaque_device_promet" ):
    m = _machine()

    if getattr( driver.device, "is_cuda_gpu", False ):
        # la largeur de warp est 32 sur tout ce qui existe ; le test le fige pour qu'un changement
        # se remarque plutôt que de passer en silence.
        assert m[ "sub_group_width" ] == 32, m
        assert m[ "suggested_group" ] == m[ "sub_group_width" ], m
        # 48 kio par bloc est le plancher depuis Fermi
        assert m[ "local_mem_bytes" ] >= 48 * 1024, m
        # SMs x fils par SM : au moins quelques milliers sur une carte qui existe
        assert m[ "nb_workers" ] >= 1024, m
    else:
        # pas de voies sur un CPU, donc pas de sub-group : tout est à 1, et le budget de mémoire
        # partagée est NOTIONNEL ( `CpuQueue` prend un `std::vector` sur le tas ).
        assert m[ "sub_group_width" ] == 1, m
        assert m[ "suggested_group" ] == 1, m
        assert m[ "local_mem_bytes" ] == 64 * 1024, m
        # le pool de fils : au moins un, et pas plus que ce que la machine a de coeurs logiques
        import os
        assert 1 <= m[ "nb_workers" ] <= ( os.cpu_count() or 1 ), m
    print( f"{ driver.device } : les promesses du device tiennent" )
