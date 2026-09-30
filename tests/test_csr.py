"""Le tenseur RAGGED en CSR ( `loom.CsrTensor` ) : des lignes de longueurs differentes, rangees
bout a bout.

Ce qu'il faut verifier tient en une phrase : DANS `csr( i, j )`, LE `i` VA CHERCHER DANS LES
OFFSETS. Le reste -- la somme prefixe, le total, l'allocation exacte -- en decoule.
"""
import numpy

import loom
from loom.compilation.FfiCode import FfiCode
from errand import test


_REMPLIR = FfiCode.per_item( code = """
    const SI i = flat_index;
    for ( SI j = 0; j < outputs.csr.row_size( i ); ++j )
        outputs.csr( i, j ) = 10 * i + j;
""" )


if test( "le_i_va_chercher_dans_les_offsets" ):
    # trois lignes de longueurs 2, 0 et 3 -- dont une VIDE, qui est le cas ou un rembourrage
    # gaspille le plus et ou une arithmetique d'offsets se trompe le plus facilement.
    comptes = loom.IntTensor[ 3 ]( [ 2, 0, 3 ] )
    csr = loom.CsrTensor.from_counts( comptes )

    assert numpy.asarray( csr.offsets.value ).tolist() == [ 0, 2, 2, 5 ]
    assert csr.total == 5          # la derniere borne EST le total
    assert csr.nb_rows_value == 3

    loom.ffi_call( "csr_remplir", _REMPLIR, csr = loom.out( csr, "values" ), nb_items = 3 )

    # ligne 0 -> 0, 1 ; ligne 1 -> rien ; ligne 2 -> 20, 21, 22
    assert numpy.asarray( csr.values.value ).reshape( -1 ).tolist() == [ 0, 1, 20, 21, 22 ]
    print( f"3 lignes ( 2, 0, 3 ) -> { csr.total } fentes, exactement le total" )


if test( "une_ligne_vide_a_une_taille_nulle" ):
    # `row_size` est une SOUSTRACTION de deux bornes voisines : une ligne vide donne 0, et il n'y
    # a pas de tableau de comptes a tenir en accord avec les offsets.
    tailles = loom.IntTensor[ 4 ]()
    comptes = loom.IntTensor[ 4 ]( [ 0, 3, 0, 1 ] )
    csr = loom.CsrTensor.from_counts( comptes )

    loom.ffi_call(
        "csr_tailles",
        FfiCode.per_item( code = "outputs.tailles( flat_index ) = inputs.csr.row_size( flat_index );" ),
        csr = csr,
        tailles = loom.out( tailles ),
        nb_items = 4,
    )
    assert numpy.asarray( tailles.value ).reshape( -1 ).tolist() == [ 0, 3, 0, 1 ]


if test( "le_csr_ne_rembourre_rien" ):
    # LE POINT : `values` est alloue au TOTAL, pas a `lignes x plus_longue_ligne`. Une ligne longue
    # au milieu de lignes courtes est exactement ce qui fait diverger les deux.
    comptes = loom.IntTensor[ 5 ]( [ 1, 1, 60, 1, 1 ] )
    csr = loom.CsrTensor.from_counts( comptes )

    rembourre = 5 * 60
    assert csr.total == 64
    assert tuple( csr.values.shape ) == ( 64, )
    print( f"rembourre { rembourre } fentes, csr { csr.total } -- x{ rembourre / csr.total:.1f}" )
