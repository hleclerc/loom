#pragma once

// les membres + les methodes engendrees ( `operator()`, `kernel_form`, ... ) sous forme de macros
// que cette struct depose, ecrites dans l'arbre d'inclusion par `CallArg_Aggregate`.
#include <sdot/generated/aggregates/CsrTensor.h>
#include <loom/support/common_macros.h>

namespace sdot {

/// UN TENSEUR RAGGED EN CSR : `offsets` dit ou chaque ligne commence dans `values`, et `values`
/// est la liste plate de tout le contenu.
///
///     offsets  [ 0, 2, 2, 5 ]        quatre bornes pour trois lignes
///     values   [ a, b, c, d, e ]
///     => ligne 0 = { a, b }, ligne 1 = {}, ligne 2 = { c, d, e }
///
/// CE QU'IL ECHANGE CONTRE LE REMBOURRE. Un ragged rembourre coute `lignes x plus_longue_ligne` ;
/// un CSR coute exactement l'utile. Mesure sur `examples/splats` : le rembourre y coute de x2.38 a
/// x5.08 la memoire du CSR. Le prix est une PASSE de plus a la construction -- compter, puis
/// remplir, les offsets etant la somme prefixe des comptes ( `loom.cumsum( ..., exclusive = True )` )
/// -- et une lecture HOTE du total, qui dimensionne `values`.
///
/// `offsets` porte `nb_rows + 1` bornes et non `nb_rows` comptes : la taille d'une ligne est alors
/// une SOUSTRACTION de deux voisins, sans tableau de plus, et la derniere borne est le total.
SDOT_TEMPLATE_DECL_FOR_CsrTensor
struct CsrTensor {
    SDOT_ATTRIBUTES_OF_CsrTensor

    using TF = DECAYED_TYPE_OF( values )::TF;

    /// ou la ligne `row` commence dans `values`
    HD SI row_begin( SI row ) const { return SI( offsets( num_bound = row ) ); }

    /// combien d'elements la ligne `row` porte
    HD SI row_size( SI row ) const { return SI( offsets( num_bound = row + 1 ) ) - row_begin( row ); }

    /// combien de lignes -- la derniere borne n'en est pas une
    HD SI nb_rows_of() const { return offsets.size( num_bound ) - 1; }

    /// `a( i, j )` : LE `i` VA CHERCHER DANS LES OFFSETS, le `j` indexe dans la ligne. C'est la
    /// seule chose qu'un usager ait a savoir du CSR, et elle se lit comme un tableau a deux
    /// indices.
    ///
    /// Non template et exactement deux `SI` : l'`operator()` variadique engendre ( celui qui
    /// indexe TOUT l'agregat, `csr( batch_index )` ) est un modele, donc cette surcharge-ci gagne
    /// pour deux entiers et lui laisse le reste.
    HD decltype( auto ) operator()( SI row, SI slot ) const { return values( num_slot = row_begin( row ) + slot ); }

    /// le meme, sous un nom, pour un appelant qui prefere le dire
    HD decltype( auto ) at( SI row, SI slot ) const { return operator()( row, slot ); }
};

}
