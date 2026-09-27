#pragma once

// LE LANCEMENT, et rien d'autre.
//
// Trois couches, et c'est tout le propos de cet exemple :
//
//   pas.h      la PHYSIQUE. Des fonctions libres sur des vues indexees positionnellement. Elle ne
//              sait pas qu'elle sera parallele, ni derivable, ni appelee depuis Python.
//   noyaux.h   CE FICHIER : comment on parcourt la grille. C'est ici, et nulle part ailleurs, que
//              le parallelisme se decide.
//   diffusion.py  ce que loom ecrit : l'enrobage FFI, la liaison des tampons, l'adjoint cote Jax.
//              Il ne sait pas comment on parallelise, et n'a pas a le savoir.
//
// D'ou vient l'interet : ce fichier est celui qu'on remplace si on prefere Kokkos, SYCL, OpenMP ou
// une simple boucle sequentielle. `run_parallel` ci-dessous est l'outil de loom, pas une obligation
// de loom.
//
// CE QUE LOOM NOUS REMET est un seul objet, `args`, dont les membres portent les noms des kwargs de
// l'appel -- plus `queue`, `machine` ( voir loom/support/kernels/Machine.h ) et `errors`, et une
// politique d'io par argument ( `<nom>_io` ). Nos fonctions sont donc des templates sur ce type :
// l'ordre des kwargs ne compte pas, et ajouter un argument a l'appel ne touche pas ce fichier.

#include "pas.h"

#include <loom/support/kernels/run_parallel.h>
#include <loom/support/algorithms/CartesianIndices.h>
#include <loom/support/kernels/IoCategory.h>

namespace diffusion {

using sdot::operator""_c; // `0_c` : un indice connu a la compilation ( voir loom/support/Ct.h )

/// ce qui se passe pour UNE cellule.
///
/// Un foncteur NOMME au niveau du namespace, et pas une lambda : C++ interdit les methodes template
/// dans une classe locale, et un compilateur device est plus heureux avec une struct plate.
struct UnPas {
    HD void operator()( auto item, auto grille, auto coef, auto suivant ) const {
        const sdot::SI j = item[ 0_c ], i = item[ 1_c ];
        suivant( j, i ) = pas_explicite( grille.temperature, grille.diffusivite, j, i,
                                         sdot::SI( grille.ny ), sdot::SI( grille.nx ), coef );
    }
};

/// UN PAS, sur toute la grille. Le domaine est celui de la DONNEE -- deux coordonnees, pas un rang
/// plat a redecouper.
///
/// Les politiques d'io viennent de `args` ( `a.grille_io` ) et non d'un tag ecrit ici : loom les a
/// deduites de ce que l'appel declare en sortie, attribut par attribut. Un tag nu marcherait aussi
/// aujourd'hui, mais ce serait redire -- et moins juste le jour ou un device transferera.
void pas( auto &a ) {
    sdot::run_parallel( a.queue, sdot::indices_over( a.grille.ny, a.grille.nx ), UnPas{},
                        a.grille_io,   a.grille,
                        a.coef_io,     a.coef,
                        a.suivant_io,  a.suivant );
}

/// l'adjoint pour UNE cellule : les deux gradients, en gather pur ( voir `pas.h` ).
struct UnPasAdjoint {
    HD void operator()( auto item, auto grille, auto coef, auto grad_for_suivant,
                        auto grad_for_grille, auto grad_for_coef ) const {
        const sdot::SI n = sdot::SI( grille.nx ), m = sdot::SI( grille.ny );
        const sdot::SI j = item[ 0_c ], i = item[ 1_c ];

        // `coef` est une constante du probleme, jamais perturbee : son gradient demanderait une
        // reduction globale ( une somme atomique sur toute la grille ), et il n'est pas ecrit.
        static_assert( DECAYED_TYPE_OF( grad_for_coef.is_valid() )::value == 0,
            "diffusion : le gradient par rapport au coefficient dt/h^2 n'est pas implemente" );

        // un tampon de sortie n'est PAS garanti a zero : quand la cotangente est un zero
        // symbolique il faut quand meme ecrire le gradient nul.
        constexpr bool nulle = DECAYED_TYPE_OF( grad_for_suivant.surely_null() )::value;

        if constexpr ( DECAYED_TYPE_OF( grad_for_grille.temperature.is_valid() )::value ) {
            if constexpr ( nulle )
                grad_for_grille.temperature( j, i ) = 0;
            else
                grad_for_grille.temperature( j, i ) = adjoint_temperature(
                    grille.temperature, grille.diffusivite, grad_for_suivant, j, i, m, n, coef );
        }

        if constexpr ( DECAYED_TYPE_OF( grad_for_grille.diffusivite.is_valid() )::value ) {
            if constexpr ( nulle )
                grad_for_grille.diffusivite( j, i ) = 0;
            else
                grad_for_grille.diffusivite( j, i ) = adjoint_diffusivite(
                    grille.temperature, grille.diffusivite, grad_for_suivant, j, i, m, n, coef );
        }
    }
};

/// l'adjoint du pas, sur toute la grille. Meme domaine que l'aller -- c'est le meme parcours.
void pas_adjoint( auto &a ) {
    sdot::run_parallel( a.queue, sdot::indices_over( a.grille.ny, a.grille.nx ), UnPasAdjoint{},
                        a.grille_io,           a.grille,
                        a.coef_io,             a.coef,
                        a.grad_for_suivant_io, a.grad_for_suivant,
                        a.grad_for_grille_io,  a.grad_for_grille,
                        a.grad_for_coef_io,    a.grad_for_coef );
}

} // namespace diffusion
