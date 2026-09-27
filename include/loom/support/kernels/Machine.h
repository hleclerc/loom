#pragma once

#include "../common_macros.h" // HD
#include "../common_types.h" // SI

namespace sdot {

/// CE QUE LE NOYAU PEUT SAVOIR DE LA MACHINE, sous une forme qui ne parle d'aucune machine.
///
/// C'est ce qui manquait pour ecrire un noyau portable : la geometrie de lancement etait soit codee
/// en dur ( `const int block = 128` dans `CudaQueue.h` ), soit devinee en Python, soit recopiee de
/// tutoriel en tutoriel. Un noyau qui veut choisir sa taille de groupe ou son budget de memoire
/// partagee doit pouvoir la DEMANDER.
///
/// Les quatre champs sont volontairement peu nombreux, et chacun a un sens sur CPU comme sur GPU :
/// un noyau ecrit une fois lit les memes noms partout. Ce sont des CONSEILS et des budgets, pas des
/// lois -- `run_parallel` n'en lit aucun tout seul, c'est le noyau qui decide.
///
/// Obtenu par `queue.machine()`. Les interrogations du driver sont faites UNE fois et gardees.
struct Machine {
    /// combien de work-items peuvent progresser en meme temps.
    /// CPU : la taille du pool de fils. CUDA : SMs x fils par SM.
    /// A quoi ca sert : dimensionner un scratch PAR FIL sur les travailleurs concurrents et non sur
    /// les items, donc un gros batch ne fait pas exploser la memoire.
    SI  nb_workers;

    /// combien de voies avancent en verrou ( un warp ).
    /// CPU : 1, il n'y a pas de voies. CUDA : `warpSize`, 32 en pratique.
    /// A quoi ca sert : un algorithme coopératif ( scan, histogramme ) se decoupe dessus.
    SI  sub_group_width;

    /// budget de memoire partagee par groupe, en octets -- ce que `local_mem_elems` doit respecter.
    /// CUDA : le maximum par bloc, lu sur le driver. CPU : une valeur NOTIONNELLE ( il n'y a pas de
    /// memoire partagee materielle, `CpuQueue` prend un `std::vector` sur le tas ), choisie pour
    /// qu'un noyau portable dimensionne quelque chose de sense plutot que de diviser par zero.
    SI  local_mem_bytes;

    /// une taille de groupe qui marche, quand le noyau n'a pas de raison d'en preferer une autre.
    /// CPU : 1. CUDA : la largeur de warp.
    SI  suggested_group;

    /// un POD trivialement copiable : sa forme noyau est lui-meme, donc il traverse jusque dans un
    /// kernel ( voir `make_avaiable.h` ). Utile pour un corps qui adapte son decoupage sur place.
    HD Machine kernel_form( auto &&, auto ) const { return *this; }
};

/// la valeur notionnelle du budget de memoire partagee sur une machine qui n'en a pas.
static constexpr SI cpu_notional_local_mem_bytes = 64 * 1024;

} // namespace sdot
