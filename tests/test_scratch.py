"""SPIKE -- allouer PENDANT l'appel, à une taille que seul le noyau connaît.

LA PRÉMISSE QU'ON TESTE. On avait supposé la chose impossible : Jax préalloue le gros de la RAM du
GPU, donc il ne resterait rien à prendre. Cette prémisse est fausse en droit -- XLA expose son
propre pool au handler FFI : `XLA_FFI_DeviceMemory_Allocate` est un CHAMP de la struct
`XLA_FFI_Api` ( `xla/ffi/api/c_api.h` ), donc de l'ABI C stable, et non un utilitaire d'en-tête.

CE QUE ÇA CHANGERAIT. La forme d'une SORTIE doit être connue au traçage, et ça ne bougera pas. Une
taille INTERNE, non -- XLA ne l'a jamais exigée. Un index construit et consommé dans le même appel
pourrait donc être dimensionné EXACTEMENT, sans borne devinée, et sous `jit`. C'est la différence
avec `scratch_attributes`, dont la capacité descend de Python.

CE QUE CE TEST ÉTABLIT, ET CE QU'IL N'ÉTABLIT PAS. Sur CPU, il l'établit pour de bon : trois
tailles internes différentes pour une seule source, exactes, sans borne -- et la même chose SOUS
`jit`. Là, la mémoire vient d'un `aligned_alloc` du handler, pas d'XLA : le backend CPU répond
« No device memory allocator available on this platform », et il a raison, il n'y a pas de device
( c'est ce qu'a montré la première version de ce spike, qui passait par le pool d'XLA partout ).

Sur GPU il faudra bien le pool d'XLA, et ÇA N'EST PAS TESTÉ ICI : nvcc 13.4 segfaute sur la source
CUDA engendrée par loom, à -O0 et après un préprocessing réussi, indépendamment de ce spike
( reproduit sans lui ). Le test accepte donc aussi un refus, du moment qu'il est LISIBLE -- parce
qu'un refus propre est un comportement correct, alors qu'un segfault, un silence ou un résultat
faux n'en sont pas.
"""
from loom import Axis, ShapeVar, RealTensor, driver
from loom.compilation.FfiCode import FfiCode
from errand import test

import numpy


# Le corps EST le handler ( `FfiCode.handler` ) : c'est du code hôte, il a besoin de `scratch`, et
# il n'y a pas d'item à échafauder autour. `scratch = True` est ce qui fait lier l'allocateur
# d'XLA -- un opt-in, parce que le nom d'un noyau est le hachage de sa source et qu'une clause
# `Bind()` ajoutée sans condition recompilerait tout le dépôt.
#
# NB la boucle de compactage est bornée par `compact.shape( 0 )` ET PAS par l'entrée. C'est LE
# geste qui rend un refus inoffensif : `Scratch::view` rend une vue VIDE quand le pool dit non, et
# un corps qui boucle sur sa propre taille ne fait alors rien. La première version de ce test
# bornait sur l'entrée et écrivait à travers un pointeur nul -- c'est ce segfault qui a révélé le
# refus du CPU.
_CODE = """
    void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
        // ce que seul le noyau sait : combien d'entrees sont positives. Aucune borne n'a ete
        // donnee, ni par Python ni par XLA.
        SI n = 0;
        for ( SI i = 0; i < args.valeurs.shape( 0 ); ++i )
            if ( args.valeurs( i ) > 0 )
                ++n;

        // ... et on alloue EXACTEMENT ca. `scratch` est dans `args` : allouer est une operation
        // hote, donc il n'a rien a faire dans la forme kernel.
        auto compact = args.scratch.template view<double>( n );

        // la boucle est bornee par LA VUE et pas par l'entree : c'est ce qui rend un refus
        // inoffensif ( voir Scratch.h ).
        SI k = 0;
        for ( SI i = 0; i < args.valeurs.shape( 0 ) && k < compact.shape( 0 ); ++i )
            if ( args.valeurs( i ) > 0 )
                compact( k++ ) = args.valeurs( i );

        double s = 0;
        for ( SI i = 0; i < compact.shape( 0 ); ++i )
            s += compact( i );

        args.somme( 0 ) = s;
        args.somme( 1 ) = double( compact.shape( 0 ) );
    }
"""


def _appel( valeurs ):
    """`( somme des positifs, combien )`, passées par un tampon interne de taille exacte --
    ou `None` si la plateforme refuse d'allouer."""
    v = RealTensor[ Axis( ShapeVar( len( valeurs ) ), name = "num_point" ) ]( valeurs )
    somme = RealTensor[ Axis( ShapeVar( 2 ), name = "num_sortie" ) ]()
    try:
        driver.call(
            FfiCode( code = _CODE, scratch = True ),
            name = "test_scratch_positifs",
            valeurs = v,
            somme = somme,
            output_attributes = [ "somme" ],
        )
    except Exception as e:
        # un refus, et il doit être LISIBLE : c'est la moitié du contrat.
        assert "RESOURCE_EXHAUSTED" in str( e ), e
        assert "scratch" in str( e ), e
        return None
    brut = somme.raw.tolist()
    return brut[ 0 ], int( brut[ 1 ] )


if test( "alloue_a_la_taille_exacte" ):
    # trois tailles utiles DIFFÉRENTES pour une même source : si la capacité devait être prescrite,
    # il faudrait ici une borne -- donc du gâchis, ou une troncature.
    cas = ( ( [ 1.0, -2.0, 3.0 ], 4.0, 2 ),
            ( [ -1.0, -2.0, -3.0 ], 0.0, 0 ),
            ( [ 5.0 ] * 17, 85.0, 17 ) )

    servi = None
    for donnees, attendu, combien in cas:
        res = _appel( donnees )
        if res is None:
            # la plateforme n'alloue pas. Le seul cas qui doit tout de même passer est celui qui
            # ne demande rien : zéro élément ne s'alloue pas, donc rien ne peut être refusé.
            assert combien > 0, ( "une allocation de 0 élément a été refusée -- `Scratch::view` "
                                  "doit court-circuiter avant d'appeler le pool", donnees )
            servi = False
            continue
        assert abs( res[ 0 ] - attendu ) < 1e-12, ( donnees, res )
        assert res[ 1 ] == combien, ( donnees, res )
        if combien > 0:
            servi = True

    print( "scratch : servi par la plateforme" if servi else
           "scratch : plomberie ok, mais la plateforme REFUSE d'allouer ( CPU : « No device "
           "memory allocator available on this platform » )" )


if test( "sous_jit" ):
    # LE point du spike. Une SORTIE de forme dépendante des données est impossible sous `jit` et le
    # restera. Une taille INTERNE passerait : la forme du tampon n'est pas dans le programme XLA,
    # elle n'existe qu'à l'exécution du handler. Ce test le vérifie là où c'est servi, et vérifie
    # sinon que le refus traverse le `jit` proprement plutôt que de corrompre un résultat.
    n = 64

    def perte( x ):
        v = RealTensor[ Axis( ShapeVar( n ), name = "num_point" ) ]( x )
        somme = RealTensor[ Axis( ShapeVar( 2 ), name = "num_sortie" ) ]()
        driver.call(
            FfiCode( code = _CODE, scratch = True ),
            name = "test_scratch_positifs",
            valeurs = v,
            somme = somme,
            output_attributes = [ "somme" ],
        )
        return somme.tensor[ 0 ]

    compile = driver.jit( perte )
    rng = numpy.random.default_rng( 0 )

    refuse = False
    for _ in range( 3 ):
        x = rng.normal( size = n )
        attendu = float( x[ x > 0 ].sum() )
        try:
            obtenu = float( compile( driver.array( x ) ) )
        except Exception as e:
            assert "RESOURCE_EXHAUSTED" in str( e ), e
            refuse = True
            break
        assert abs( obtenu - attendu ) < 1e-10, ( obtenu, attendu )

    print( "scratch sous jit : refus propre, pas de résultat faux" if refuse else
           f"scratch sous jit : { n } entrées, taille interne variable -- ok" )
