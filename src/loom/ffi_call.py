"""LE point d'entree : executer un noyau C++/CUDA sur des tenseurs.

`driver` est la couche BASSE -- il porte le framework (Jax/Torch), le device, les types resolus, et
il n'a pas a apparaitre dans le code d'un usager. `ffi_call` est le meme appel, sous le nom de ce
qu'il fait.
"""


def ffi_call( *kernels, **kwargs ):
    """Lance `kernels` sur les tenseurs passes en kwargs.

        loom.ffi_call(
            avant, arriere,                  # le second est l'ADJOINT ( optionnel )
            name = "diffusion_pas",          # identifie le couple, prefixe la cible compilee
            temperature = temperature,       # les arguments, sous les noms que le C++ verra
            suivant = suivant,
            output_attributes = [ "suivant" ],
        )

    Rien n'est renvoye : les sorties sont reecrites sur les objets qu'on a passes ( ce sont nos
    objets Python -- le framework ne voit que les tenseurs a l'interieur ).
    """
    from .drivers.driver import driver
    return driver.call( *kernels, **kwargs )
