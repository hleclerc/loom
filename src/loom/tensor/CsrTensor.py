"""Le tenseur RAGGED en CSR : des lignes de longueurs differentes, rangees bout a bout."""

from ..util.Aggregate import Aggregate
from .CtShapeVar import CtShapeVar
from .IntTensor import IntTensor
from .ShapeVar import ShapeVar
from .Tensor import Tensor
from .Axis import Axis


class CsrTensor( Aggregate ):
    """Des lignes de longueurs differentes, rangees BOUT A BOUT -- pas de rembourrage.

        offsets  [ 0, 2, 2, 5 ]        quatre bornes pour trois lignes
        values   [ a, b, c, d, e ]
        => ligne 0 = { a, b }, ligne 1 = {}, ligne 2 = { c, d, e }

    Cote C++ ( `loom/include/sdot/CsrTensor.h` ), le seul geste a connaitre est :

        csr( i, j )        le `i` va chercher dans les OFFSETS, le `j` indexe dans la ligne
        csr.row_size( i )  la longueur de la ligne `i`

    CE QU'IL ECHANGE CONTRE LE RAGGED REMBOURRE de loom ( `ShapeVar[ "axe" ]`, un compte par
    ligne, un tampon `lignes x plus_longue_ligne` ) : de la MEMOIRE contre une PASSE. Le rembourre
    s'inscrit en un seul balayage -- reserver une fente et ecrire -- la ou le CSR demande de
    COMPTER d'abord, puis de remplir une fois les offsets connus. Mesure sur `examples/splats` :
    le rembourre y coute de x2.38 a x5.08 la memoire du CSR.

    ET UNE LECTURE HOTE, qui est le vrai prix : le TOTAL dimensionne `values`, et c'est un compte
    qu'un noyau vient d'ecrire. Une forme batie par `from_counts` n'est donc pas utilisable sous
    `jit` -- exactement ce qu'un JIT ne peut pas payer, et ce que `driver.call` sait faire.

    `offsets` porte `nb_rows + 1` BORNES et non `nb_rows` comptes : la taille d'une ligne est alors
    une soustraction de deux voisins, sans tableau de plus, et la derniere borne EST le total.
    """

    offsets   : IntTensor[ "num_bound" ]
    values    : Tensor[ "num_slot" ]

    num_bound : Axis[ "nb_rows + 1" ]
    num_slot  : Axis[ "nb_slots" ]

    nb_rows   : ShapeVar
    nb_slots  : ShapeVar

    @classmethod
    def from_counts( cls, counts, **template_kwargs ):
        """Le CSR qu'un tenseur de COMPTES decrit : `offsets` est leur somme prefixe exclusive,
        plus le total en derniere borne, et `values` est alloue a ce total EXACTEMENT.

            comptes  [ 2, 0, 3 ]   ->   offsets [ 0, 2, 2, 5 ]   et  values de 5 fentes

        La somme prefixe se fait SUR LE DEVICE ( `Tensor.cumsum` ). Le total, lui, revient sur
        l'hote : c'est lui qui dimensionne l'allocation, et une taille ne peut pas etre une valeur
        device. C'est la seule chose ici qu'un `jit` ne peut pas traverser, et c'est ce qu'on
        achete en echange de l'exactitude.

        LES COMPTES PEUVENT AVOIR N'IMPORTE QUEL RANG, et sont lus dans l'ordre du tenseur : une
        GRILLE de comptes ( une grille de tuiles ) donne autant de lignes, prises ligne par ligne.
        C'est le seul endroit ou une structure a plusieurs axes s'aplatit, et c'est ce qu'est un
        CSR -- des bornes cumulees le long d'UN ordre, donc une suite de lignes et pas une grille.

        `template_kwargs` va a `values` ( `dtype`, `size`, `device` ) : ce que les lignes PORTENT
        n'est pas decide par les comptes.
        """
        from ..drivers.driver import driver
        from .functions import cumsum

        brut = counts.value if isinstance( counts, Tensor ) else counts
        plat = Tensor.wrap( brut.reshape( -1 ) )

        total = int( plat.sum() )
        nb = int( plat.shape[ 0 ] )

        res = cls( nb_rows = nb, nb_slots = max( total, 1 ),
                   values = dict( template_kwargs ) if template_kwargs else {} )
        # les bornes : la somme prefixe exclusive, PUIS le total -- `nb + 1` entrees. Le
        # `concatenate` passe par le driver, donc il reste la ou la donnee vit.
        debuts = cumsum( plat, exclusive = True )
        res.offsets = driver.concatenate( [ debuts.value, driver.array( [ total ], dtype = debuts.dtype ) ] )
        return res

    @property
    def nb_rows_value( self ) -> int:
        """Combien de lignes -- la derniere borne n'en est pas une."""
        return int( self.nb_rows.value )

    @property
    def total( self ) -> int:
        """Combien d'elements en tout : la derniere borne."""
        return int( self.nb_slots.value )
