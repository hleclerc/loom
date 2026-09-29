"""LE point d'entree : executer un noyau C++/CUDA sur des tenseurs.

`driver` est la couche BASSE -- il porte le framework (Jax/Torch), le device, les types resolus, et
il n'a pas a apparaitre dans le code d'un usager. `ffi_call` est le meme appel, sous le nom de ce
qu'il fait.
"""


def ffi_call( *kernels, **kwargs ):
    """Lance `kernels` sur les valeurs passees en kwargs.

        loom.ffi_call(
            avant, arriere,                  # le second est l'ADJOINT ( optionnel )
            name = "diffusion_pas",          # identifie le couple, prefixe la cible compilee
            temperature = temperature,       # les arguments, sous les noms que le C++ verra
            coef = 0.2,
            suivant = suivant,
            output_attributes = [ "suivant" ],
        )

    LES ENTREES SONT PRISES TELLES QU'ELLES SONT. Un tableau du framework, un tableau numpy, une
    liste, un flottant : loom en lit le type d'element ( un fait ) et en deduit les axes de la
    forme, la TAILLE du scalaire restant la politique du driver ( `loom.driver.ftype` ). Il n'y a
    donc rien a emballer -- `loom.RealTensor( u )` reste ecrivable, mais n'est plus une formalite
    a traverser. Voir `Tensor.as_tensor`.

    LES SORTIES, ELLES, SE DECLARENT, et pour une raison : rien n'est renvoye ici. Les resultats
    sont reecrits sur les objets qu'on a passes, donc une sortie doit etre un objet a NOUS --
    `loom.RealTensor.like( temperature )` en dit ce qu'il faut ( meme forme, meme type ), et
    `like` accepte lui aussi une valeur brute. Une valeur brute nommee dans `output_attributes`
    est refusee : le resultat n'aurait nulle part ou revenir.

    ( Le framework, lui, ne voit jamais nos objets -- seulement les tenseurs a l'interieur. )
    """
    from .drivers.driver import driver
    return driver.call( *kernels, **kwargs )
