"""SPIKE, deuxième moitié -- une taille décidée par la DONNÉE, sur la carte, ET SOUS `jit`.

CE QU'ON CROYAIT. `examples/splats/README.md` documente une frontière : l'hôte peut lire un compte
qu'un noyau vient d'écrire, mais « eager-only » -- `ShapeArray` refuse un tracer et
`capacity_overflows()` rend `None` sous `jit`. La conclusion était qu'une taille dépendante des
données ne passe pas un `jit`.

CE QUE CE TEST ÉTABLIT. Cette frontière n'est PAS une propriété d'XLA. C'est une propriété de faire
la relecture EN PYTHON. Déplacée dans le C++ du handler, elle disparaît : le handler tourne à
l'EXÉCUTION, donc il peut lire un compte device, allouer dessus, et rien de tout cela n'a à exister
au traçage.

La chaîne, entièrement dans un seul appel :

  1. un noyau compte, sur la carte ;
  2. l'hôte -- le handler, qui est du code hôte -- lit ce compte ;
  3. il alloue EXACTEMENT ça dans le pool d'XLA ( `XLA_FFI_DeviceMemory_Allocate` ) ;
  4. un noyau compacte dedans, un autre le somme.

Aucune borne n'est prescrite, ni par Python ni par XLA. Mesuré le 2026-09-26 sur une sm_75 : exact
sur trois tailles en eager, et sur quatre tirages sous `jit` ( 1989 / 2023 / 2040 / 2074 éléments
alloués pour la même fonction compilée UNE fois ).

Ce test demande un GPU ( sur CPU, `scratch` passe par `aligned_alloc` et la relecture device n'a
pas de sens -- voir `test_scratch.py` ), et se saute tout seul sinon. L'environnement `nsdot`
d'errand est tague cuda mais vide a ce jour ; en attendant, directement :

    job -- env ERRAND_IN_ENV=1 LOOM_DEVICE=cuda PYTHONPATH=<errand>:<loom>/src \
        /data/venvs/sdot/bin/python -m errand test_scratch_gpu
"""
from pathlib import Path

from loom import Axis, ShapeVar, RealTensor, driver, compilation
from loom.compilation.FfiCode import FfiCode
from errand import test

import numpy

compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )


_CORPS = """
    SI n = valeurs.shape( 0 );

    // 1. un noyau compte, sur la carte
    auto cpt = scratch.view<int>( 1 );
    cpt.fill_with( queue, 0 );
    run_parallel( queue, range( n ), loom_tests::Compter{}, OutList(), cpt, InpList(), valeurs );

    // 2. l'HOTE lit ce que le noyau vient d'ecrire. `Ptr::value()` ferait ca, mais il est marque
    //    `HD` alors que son chemin de transfert est HOTE seul -> le compilateur CUDA le refuse
    //    sous `--Werror cross-execution-space-call`. D'ou le `copy` direct.
    int m_hote = 0;
    copy( Ptr<int,CpuHostMemorySpace>( &m_hote ), cpt.data(), 1 );
    SI m = SI( m_hote );

    // 3. ... et on alloue EXACTEMENT ca
    auto compact = scratch.view<double>( m );
    cpt.fill_with( queue, 0 );
    run_parallel( queue, range( n ), loom_tests::Compacter{}, OutList(), compact, OutList(), cpt, InpList(), valeurs );

    // 4. de quoi verifier : la somme des positifs, et la taille qui a ete allouee
    run_parallel( queue, range( m ), loom_tests::Sommer{}, OutList(), somme, InpList(), compact );
    run_parallel( queue, range( 1 ), loom_tests::Poser{}, OutList(), somme, InpList(), double( m ) );
"""


def _code():
    return FfiCode.handler( code = _CORPS, scratch = True,
                            includes = [ "loom_tests/scratch_gpu.h" ] )


def _calcul( x ):
    v = RealTensor[ Axis( ShapeVar( len( x ) ), name = "num_point" ) ]( x )
    somme = RealTensor[ Axis( ShapeVar( 2 ), name = "num_sortie" ) ]()
    driver.call( _code(), name = "test_scratch_gpu", valeurs = v, somme = somme,
                 output_attributes = [ "somme" ] )
    return somme


def _attendu( x ):
    return float( x[ x > 0 ].sum() ), int( ( x > 0 ).sum() )


def _sur_gpu():
    """Un GPU est-il la ? La question porte sur le DEVICE de loom, pas sur la presence d'une carte :
    sans jaxlib CUDA, loom retombe sur le CPU et ce test n'a plus d'objet."""
    try:
        return bool( getattr( driver.device, "is_cuda_gpu", False ) )
    except Exception:
        return False


_GPU = _sur_gpu()
_HORS_GPU = "pas de GPU ici -- sauté ( voir l'en-tête du fichier pour la commande )"


if test( "taille_decidee_par_la_donnee" ):
    if not _GPU:
        print( _HORS_GPU )
    else:
        # trois tailles utiles différentes, pour une source unique et sans borne
        for graine, n in ( ( 0, 1000 ), ( 1, 5000 ), ( 2, 37 ) ):
            x = numpy.random.default_rng( graine ).normal( size = n )
            s, m = _calcul( x ).raw.tolist()
            att_s, att_m = _attendu( x )
            assert int( m ) == att_m, ( n, int( m ), att_m )
            assert abs( s - att_s ) < 1e-9, ( n, s, att_s )
        print( f"scratch : taille exacte lue sur la carte, sur { driver.device }" )


if test( "sous_jit" ):
    if not _GPU:
        print( _HORS_GPU )
    else:
        # LE point. La fonction est compilée UNE fois ; la taille allouée change à chaque appel.
        n = 4096

        def calcul( x ):
            return _calcul( x ).tensor

        compile = driver.jit( calcul )

        tailles = []
        for graine in range( 4 ):
            x = numpy.random.default_rng( graine ).normal( size = n )
            r = numpy.asarray( compile( driver.array( x ) ) )
            att_s, att_m = _attendu( x )
            assert int( r[ 1 ] ) == att_m, ( graine, int( r[ 1 ] ), att_m )
            assert abs( float( r[ 0 ] ) - att_s ) < 1e-8, ( graine, float( r[ 0 ] ), att_s )
            tailles.append( att_m )

        # et elles DIFFÈRENT : sinon le test passerait avec une capacité prescrite
        assert len( set( tailles ) ) > 1, tailles
        print( f"scratch SOUS JIT : une seule compilation, tailles allouées { tailles }" )
