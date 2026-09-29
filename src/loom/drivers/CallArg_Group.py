from .CallArg_Aggregate import CallArg_Aggregate
from .IoCategory import IoCategory
from .CallArg import CallArg


class CallArg_Group( CallArg_Aggregate ):
    """UN GROUPE D'ARGUMENTS : `inputs`, `outputs`, `scratch`, `grad_of_inputs`, `grad_of_outputs`.

    C'est un AGREGAT dont les membres sont DONNES au lieu d'etre trouves sur un objet python. Tout
    ce qu'un agregat sait deja faire vaut tel quel pour lui -- la struct de vues, la politique d'io
    membre par membre ( `<Type>_io` ), `kernel_form`, les en-tetes a inclure -- et c'est pourquoi
    le regroupement ne demande pas une seconde machinerie a cote de la premiere.

    CE QU'IL ACHETE. Les arguments de l'appel occupaient le PREMIER niveau de `args`, qu'ils
    partageaient avec les reglages de l'appel ( `name`, `output_attributes`, ... ) : un noyau ne
    pouvait pas avoir un argument appele `name`. Sous les groupes, le premier niveau ne contient
    plus que les groupes eux-memes -- un argument appele `name`, ou meme `inputs`, vit a
    `args.inputs.name` et ne peut plus rien heurter.

    LE NOM ET LE CHEMIN SE SEPARENT ICI, et c'est ce qui rend `loom.mutable` possible : un argument
    mutable est DEUX tampons qui portent le MEME nom C++ ( `inputs.temperature` et
    `outputs.temperature` ) sous DEUX chemins distincts ( `temperature_input`, `temperature_output`
    -- ce que les capacites et les sorties nomment cote python ). `make_CallArg` prenait deja les
    deux separement ; il ne restait qu'a s'en servir.

    Un groupe n'est pas un tampon : il n'a pas de categorie d'io propre, ses membres portent la
    leur ( ce que la politique engendree exprime membre par membre ). D'ou `IoCategory.INPUT`
    ci-dessous, qui n'est lu par personne pour un noeud sans tampon.
    """

    def __init__( self, call_args_analysis, name, type_name, children ) -> None:
        # PAS `super().__init__` : celui de `CallArg_Aggregate` va chercher ses membres sur un
        # objet python ( `attributes_of( inst )` ), et un groupe n'en a pas -- ses membres sont
        # deja des `CallArg`, batis par l'appelant. On pose donc directement ce que la classe de
        # base emet ensuite : le nom, le type, les membres.
        CallArg.__init__( self, IoCategory.INPUT, name )
        self.type_name = type_name
        # un groupe ne vient d'aucun objet de l'appelant : rien a re-deriver pour l'adjoint depuis
        # lui ( c'est `_call_backward` qui rebatit les groupes du retour, membre par membre ).
        self.inst = None
        self.attributes = dict( children )
        # les axes DECLARES le sont par les membres, chacun repond pour lui-meme.
        self.declared_axes = []
