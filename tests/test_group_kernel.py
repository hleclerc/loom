"""La facilité « kernel de GROUPE » (`FfiCodeParallel( group_size = ... )`), pour elle-même.

Elle n'était exercée que par `OtPlan1d`, à travers un tri radix coopératif et un balayage de
transport optimal : quand elle casse, le symptôme est un coût de transport faux de 3 %, ce qui
n'oriente vers rien. Ces tests-ci vérifient le CONTRAT, en trois assertions séparables :

    1. chaque voie du groupe s'exécute (`local_index` couvre bien `0..local_size-1`) ;
    2. `local_scratch` est PARTAGÉ par le groupe -- ce qu'une voie y écrit, une autre le lit ;
    3. `sycl::group_barrier` ordonne ces écritures avant ces lectures.

Un backend qui rend `local_scratch` privé par voie, ou qui n'exécute qu'une voie sur deux, tombe
ici, à un endroit qui le nomme.
"""
import numpy

from loom import driver
from loom.compilation.FfiCode import FfiCodeParallel
from loom.tensor import Axis, IntTensor, ShapeVar
from loom.testing import test


def _sum_over_lanes( group_size ):
    """Chaque voie dépose son rang dans `local_scratch`, puis la voie 0 en somme le contenu.

    La somme vaut `0+1+...+(local_size-1)` SI et seulement si les trois garanties tiennent : une
    voie manquante retire son terme, un scratch privé n'en laisse qu'un, une barrière absente en
    perd au hasard. Le résultat attendu est donc une valeur unique, pas un intervalle.
    """
    num_group = Axis( ShapeVar( 2 ), name = "num_group" )
    res = IntTensor[ num_group ]()

    driver.call(
        FfiCodeParallel( name = f"test_group_kernel_{ group_size }",
            fwd_code = """
                local_scratch[ local_index ] = local_index;
                sycl::group_barrier( group );
                if ( local_index == 0 ) {
                    int s = 0;
                    for ( int k = 0; k < local_size; ++k )
                        s += local_scratch[ k ];
                    res( group_index ) = s;
                }
            """,
            thread_cap = "res.shape( 0 )",
            group_size = str( group_size ),
            local_mem_elems = str( group_size ) ),
        output_attributes = [ "res" ],
        res = res,
    )
    return int( numpy.asarray( res.tensor ).reshape( -1 )[ 0 ] )


if test( "a_group_of_one_degenerates_to_the_plain_kernel" ):
    # `group_size == 1` est le cas de PRODUCTION sur CPU (`Cpu.group_size`) : le chemin coopératif
    # doit s'y réduire exactement au kernel ordinaire. Une voie, son rang vaut 0, somme nulle.
    assert _sum_over_lanes( 1 ) == 0


if test( "every_lane_of_a_group_runs_and_shares_its_scratch" ):
    # le vrai test : au-delà d'une voie, les trois garanties deviennent observables.
    for gs in ( 2, 4, 8 ):
        got = _sum_over_lanes( gs )
        assert got == gs * ( gs - 1 ) // 2, f"group_size={ gs }: somme={ got }, attendu { gs * ( gs - 1 ) // 2 }"


def _runtime_subgroup_width( group_size ):
    """La largeur de sub-group que le BACKEND rapporte à l'exécution, pour ce `group_size`."""
    num_group = Axis( ShapeVar( 2 ), name = "num_group" )
    res = IntTensor[ num_group ]()

    driver.call(
        FfiCodeParallel( name = f"test_group_sgw_{ group_size }",
            fwd_code = """
                if ( local_index == 0 )
                    res( group_index ) = SI( sub_group.get_local_range()[ 0 ] );
            """,
            thread_cap = "res.shape( 0 )",
            group_size = str( group_size ),
            local_mem_elems = str( group_size ) ),
        output_attributes = [ "res" ],
        res = res,
    )
    return int( numpy.asarray( res.tensor ).reshape( -1 )[ 0 ] )


if test( "the_runtime_subgroup_width_matches_what_the_device_claims" ):
    # `Device.subgroup_size` est GRAVÉ côté Python dans la mémoire locale qu'`OtPlan1d` réserve
    # (`local_mem_elems`, « kept in sync by hand »). Si le backend en rapporte une autre à
    # l'exécution, le nombre de rangs réservés et le nombre de rangs utilisés divergent -- et le
    # tri coopératif écrit hors de sa réservation, sans que rien ne le dise. D'où ce test : c'est
    # une invariante tenue À LA MAIN, donc exactement le genre qui pourrit en silence.
    # `Device.subgroup_size` est la largeur MATÉRIELLE (32 sur CUDA), donc un MAJORANT : un
    # work-group de 4 items n'a pas un sous-groupe de 32. L'attendu est `min( group_size, claimed )`
    # -- c'est bien ce que le dimensionnement d'`OtPlan1d` suppose, `num_sg = ceil( gs / sgs )`.
    claimed = driver.device.subgroup_size
    for gs in ( 1, 2, 4, 8 ):
        got = _runtime_subgroup_width( gs )
        expected = min( gs, claimed )
        assert got == expected, ( f"group_size={ gs }: le backend rapporte un sub-group de { got }, "
                                  f"attendu min( { gs }, { claimed } ) = { expected }" )


def _probe( group_size, expr, tag ):
    num_lane = Axis( ShapeVar( group_size ), name = "num_lane" )
    res = IntTensor[ num_lane ]()
    driver.call(
        FfiCodeParallel( name = f"probe_sg_{ group_size }_{ tag }",
            fwd_code = f"res( local_index ) = SI( { expr } );",
            thread_cap = "1",
            group_size = str( group_size ),
            local_mem_elems = str( group_size ) ),
        output_attributes = [ "res" ],
        res = res,
    )
    return numpy.asarray( res.tensor ).reshape( -1 ).tolist()


if test( "the_lane_to_subgroup_mapping_is_linear" ):
    # Ce dont le tri radix coopératif d'`OtPlan1d` a besoin : que chaque voie sache dans QUEL
    # sous-groupe elle se trouve, pour recevoir un morceau distinct de l'entrée.
    #
    # Le demander au backend ne marche pas. `omp.library-only` renvoie `get_group_linear_id() == 0`
    # pour TOUTE voie, tout en donnant des `local_index` distincts -- le découpage s'effondrait
    # alors sur le même morceau pour toutes, qui triaient la même tranche vers les mêmes
    # destinations. Le symptôme était un coût de transport faux de 3 %, sans aucun accès hors
    # bornes pour le signaler : rien ne pointait vers les sous-groupes.
    #
    # L'identifiant est donc DÉRIVÉ de `local_index` (voir `OtPlan1d.cxx::sort_diracs`), et c'est
    # cette dérivation que ce test verrouille : distincte par sous-groupe, couvrant 0..num_sg-1.
    for gs in ( 1, 2, 4, 8 ):
        widths = _probe( gs, "sub_group.get_local_linear_range()", "rng" )
        assert len( set( widths ) ) == 1, f"largeur de sub-group non uniforme dans le groupe: { widths }"
        w = widths[ 0 ]
        derived = sorted( set( li // w for li in _probe( gs, "local_index", "li" ) ) )
        num_sg = ( gs + w - 1 ) // w
        assert derived == list( range( num_sg ) ), \
            f"group_size={ gs }, sub-group={ w } : sous-groupes dérivés { derived }, attendu 0..{ num_sg - 1 }"
