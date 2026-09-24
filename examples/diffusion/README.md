# `diffusion` — un usager délibérément étranger de loom

Un solveur de diffusion dérivable, écrit comme si son auteur n'avait jamais entendu parler de
transport optimal. Il n'importe que `loom`, son C++ ne connaît que `<loom/support/...>`, et rien
de ce qu'il fait ne ressemble à une cellule de Laguerre : grille cartésienne, stencil à cinq
points, aucun ragged, aucune géométrie, aucun scratch.

Ce n'est pas une démonstration : c'est un **test de généralité**. La question posée est « qu'est-ce
que, dans loom, est général, et qu'est-ce qui n'est que du sdot déguisé ? ». Le bilan est en bas.

```
include/diffusion/pas.h   le C++ qu'on avait DÉJÀ : la physique et ses deux adjoints
diffusion.py              l'enrobage loom : deux agrégats, un noyau, ~40 lignes utiles
test_diffusion.py         les tests
```

```bash
./run test test_diffusion --device cpu
./run test test_diffusion --driver torch --device cpu     # les mêmes chiffres
```

## Ce que ça fait

`du/dt = div( k grad u )`, pas de temps explicite, température imposée au bord :

    u'( a ) = u( a ) + dt/h² · Σ_{b voisine} K( a, b ) ( u( b ) − u( a ) ),   K = ( k_a + k_b ) / 2

L'histoire est celle d'un étranger : « j'ai déjà ce solveur en C++, je veux le mettre dans une
boucle d'optimisation ». Donc on retrouve le champ de diffusivité `k` à partir de la température
observée après N pas, par descente de gradient à travers toute la chaîne.

Les deux adjoints sont écrits à la main dans `pas.h` — la dérivée d'un solveur fait partie du
solveur. Ils s'écrivent en **gather pur** (chaque cellule lit ses quatre voisines et écrit sa seule
valeur), donc sans accumulation atomique : `k( a )` n'entre que dans les faces qui touchent `a`, et
`u( a )` que dans la sortie de `a` et de ses voisines. `check_grad` les confronte à la différence
finie — accord à 9 chiffres sur les deux.

Le seul gradient non écrit est celui de `dt/h²`, qui lui demanderait une réduction globale ; un
`static_assert` sur `grad_for_coef.is_valid()` le dit à la compilation plutôt que de rendre un
zéro silencieux.

## Ce qui a marché sans rien demander à personne

- **L'agrégat → struct C++.** `Grille` déclare deux champs et deux axes ; le noyau reçoit
  `grille.temperature`, `grille.diffusivite`, avec les mêmes noms des deux côtés. Rien à écrire.
- **Le C++ existant n'a rien appris.** `TensorView` s'indexe positionnellement (`u( j, i )`), donc
  `pas.h` ne contient aucun vocabulaire loom : `HD`, `SI`, et c'est tout. Un en-tête qu'on avait
  déjà reste un en-tête qu'on avait déjà.
- **Le VJP.** `bwd_code` est branché par `driver.grad` sans qu'on déclare quoi que ce soit, et les
  catégories `NoneTensor` / `ZeroTensor` font vraiment tomber les termes à la compilation.
- **`register_include_root`** : un paquet tiers enregistre son propre `-I`, exactement comme sdot.
  loom ne connaît pas ses usagers par leur nom, et ça se vérifie.
- **Jax et Torch.** Le même fichier, les mêmes appels, `driver.jit` et `driver.grad` compris :
  `+18.336847518` des deux côtés, au dernier chiffre. C'est la promesse centrale de loom, et elle
  tient pour quelqu'un qui n'a jamais ouvert sdot.

## Les frictions, dans l'ordre où on les rencontre

1. **Tout s'appelle `sdot`.** Le C++ de loom vit dans `namespace sdot` ; les variables
   d'environnement sont `SDOT_BUILD_DIR`, `SDOT_CACHE_DIR`, `SDOT_KERNELS`, `SDOT_EXTERNALS` ; le
   cache est `~/.cache/sdot` ; les messages d'erreur disent « sdot: » ; le functor par défaut d'un
   `FfiCodeParallel` sans nom s'appelle `sdot_fwd_kernel`. On installe loom et on reçoit sdot. Un
   simple renommage — mais c'est la première chose que l'étranger voit.

2. **`driver.array( [ 0, 1, 2 ] )` rend des flottants.** `dtype or self.ftype` : une liste
   d'entiers devient FP64, et l'erreur ne tombe qu'à l'affectation (`cannot bind a FP64 value to a
   TI tensor`). Le type naturel de la donnée devrait gagner.

3. **Le batch d'un appel ne vient que des agrégats.** `CallArgsAnalysis` collecte `batch_axes` sur
   les arguments qui en ont ; un **tenseur nu** portant le même axe est ignoré en silence, le
   noyau reçoit un `batch_index` vide, et ça échoue en `static_assert` au fond de
   `TensorView.cxx`, pas en Python. D'où l'agrégat `Cellules`, qui n'existe que pour porter l'axe.
   (La docstring de `tensor/batch.py` connaît déjà le cas — « un axe de batch peut atteindre un
   appel par un tenseur NU » — mais l'analyse, elle, ne le gère pas.)

4. **« Qui suis-je ? » n'a pas de réponse.** Le scaffold injecte `batch_index`, `thread_index`,
   `nb_threads` — pas le rang plat de l'item. Il faut donc matérialiser un iota et le transférer à
   chaque appel. sdot fait exactement pareil (`_ranks_of_items` → `np.arange`). Un `IotaTensor`
   existe pourtant côté C++, il n'est simplement pas atteignable depuis l'appelant.

5. **Pas d'`arange`, pas de `linspace`, pas de `ones`, pas de `concatenate`** sur le driver (ils
   sont en commentaire dans `JaxDriver`). À quelqu'un à qui on interdit numpy (« ça force un
   transfert vers l'hôte »), il ne reste que des compréhensions de listes pour construire un
   indice ou un champ de coordonnées.

6. **Une compilation par taille de grille** (`CtShapeVar` grave l'extent dans la source) — ici
   c'est voulu, le stencil y gagne, mais rien ne le dit — **et une compilation par motif de
   dérivation**. Le seul test des adjoints a produit **dix noyaux distincts** : quels arguments
   sont perturbés × si la cotangente est un zéro symbolique, en chaque position de la chaîne. Le
   `if constexpr` est puissant, mais ~8 s par variante au premier passage, et le compte n'est
   visible nulle part.

7. **Un tampon de sortie n'est pas garanti à zéro.** On ne l'apprend que dans un commentaire de
   `loom/tests/test_call.py`. Un étranger qui écrit un backward gardé par
   `if constexpr ( surely_null )` et qui omet la branche `else` rend des ordures, silencieusement.

8. **Importer l'exemple demande une chirurgie de `sys.path`** : `./run test` importe le fichier de
   test par son chemin, et le module voisin n'est pas trouvable. Sans conséquence ici, mais c'est
   la première ligne du fichier.

## Ce que l'exercice dit de loom

Le cœur — `tensor` + `drivers` + `compilation` — a encaissé un usage franchement étranger **sans
qu'une ligne de loom change**, et sous les deux frameworks. C'est le meilleur argument dont on
dispose pour dire que loom n'est pas un sous-produit de sdot.

Toutes les frictions sont au bord : un renommage (1), deux constructeurs côté appelant (2, 4, 5),
une hypothèse structurelle (3), et de la documentation (6, 7, 8). Aucune ne touche au modèle.
