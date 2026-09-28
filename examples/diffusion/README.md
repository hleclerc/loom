# `diffusion` — un usager délibérément étranger de loom

Un solveur de diffusion dérivable, écrit comme si son auteur n'avait jamais entendu parler de
transport optimal. Il n'importe que `loom`, son C++ ne connaît que `<loom/support/...>`, et rien
de ce qu'il fait ne ressemble à une cellule de Laguerre : grille cartésienne, stencil à cinq
points, aucun ragged, aucune géométrie, aucun scratch.

Ce n'est pas une démonstration : c'est un **test de généralité**. La question posée est « qu'est-ce
que, dans loom, est général, et qu'est-ce qui n'est que du sdot déguisé ? ». Le bilan est en bas.

**Un seul fichier**, `diffusion.py` : le C++ du noyau est dedans, en clair. On lit le tutoriel sans
naviguer.

    du/dt = k Δu, pas de temps explicite, température imposée au bord, k CONSTANTE

    u'( a ) = u( a ) + c · Σ_{b voisine} ( u( b ) − u( a ) ),   c = dt·k / h²

## Ce que les axes achètent

Le corps ne compte **jamais** de dimensions :

```cpp
HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
    const auto main_axes = coords.axes - batch_axes;          // mes axes, pas ceux du vmap

    const TF uc = args.temperature( coords );
    if ( on_bord( coords, main_axes, args ) ) { args.suivant( coords ) = uc; return; }

    TF somme = 0;
    for_each( main_axes, [&]( auto axis ) {                   // déroulé à la compilation
        somme += args.temperature( coords + axis )
               + args.temperature( coords - axis ) - 2 * uc;
    } );
    args.suivant( coords ) = uc + args.coef * somme;
}
```

Trois primitives, et tout en découle :

| | |
|---|---|
| `coords[ axis ]` | la coordonnée **par nom**, pas par position. |
| `coords ± axis` | le voisin le long de **cet** axe ; les autres coordonnées ne bougent pas — y compris celles du batch. C'est ce qui rend le stencil écrivable une fois. |
| `coords.axes - batch_axes` | mes axes propres. Une **soustraction d'ensembles**, faite à la compilation. |

Conséquence, et elle est testée (`le_meme_corps_se_batche_sans_le_savoir`) : **un `vmap` ajoute un axe
sans que le corps change, et sans qu'il sache qu'il existe** — écart exactement `0.0` contre la
boucle faite à la main. Le même corps vaut en 2-D, en 3-D, batché ou non.

Noter ce qui n'est **pas** écrit : aucune conversion `TF(...)`. Un tenseur de rang 0 se convertit
tout seul en scalaire, donc `args.temperature( coords + axis ) - uc` se lit comme la formule.

Loom appelle `void kernel( auto &&queue, auto &&batch_axes, auto &&args )` et n'écrit que ce qui fait
mal — l'enrobage FFI, la liaison des tampons, l'adjoint côté Jax :

| | |
|---|---|
| `queue` | le contexte d'exécution. `queue.run_parallel` est **son** outil, pas une obligation : un usager Kokkos, SYCL ou OpenMP l'ignore et prend `queue.stream` plus les pointeurs et les formes de `args`. |
| `batch_axes` | les axes que l'appel a ajoutés. Une **valeur** : on la soustrait, ou on la compose (`+` fait l'**union**, pas la concaténation — un tenseur batché porte déjà l'axe). |
| `args` | nos arguments sous leurs noms Python, plus `machine` ([`Machine.h`](../../include/loom/support/kernels/Machine.h)), `errors`, et `scratch` si on l'a demandé. `TF` est le scalaire réel de l'appel. |

Le `namespace { }` est écrit par l'usager : ses `#include` vont donc où il veut, et il donne à tout
ce qu'il contient une liaison interne — nécessaire, puisque plusieurs noyaux finissent liés dans une
même bibliothèque. Et `include_roots` n'est pas dit : par défaut c'est le répertoire du `.py`.

```bash
errand test_diffusion
```

## Ce que ça fait

`du/dt = div( k grad u )`, pas de temps explicite, température imposée au bord :

    u'( a ) = u( a ) + dt/h² · Σ_{b voisine} K( a, b ) ( u( b ) − u( a ) ),   K = ( k_a + k_b ) / 2

L'histoire est celle d'un étranger qui veut mettre son solveur dans une boucle d'optimisation. Ici :
**remonter le temps** — retrouver l'état initial à partir de la température observée après N pas, par
descente de gradient à travers toute la chaîne. La perte tombe d'un facteur 1,8·10⁶.

L'adjoint est écrit à la main — la dérivée d'un solveur fait partie du solveur — et il est lui aussi
générique en dimension. Il s'écrit en **gather pur** (chaque cellule lit ses voisines et écrit sa
seule valeur), donc sans accumulation atomique : `u( a )` n'entre que dans la sortie de `a` et de ses
voisines. `check_grad` le confronte à la différence finie — accord à 9 chiffres.

Le pas de descente n'est pas un réglage à tâtonner : `evolution` **contracte** (la diffusion ne fait
que lisser), donc la hessienne de la perte a ses valeurs propres ≤ 2 et tout pas au-delà de ~0,5
diverge. C'est aussi pourquoi l'inversion est lente, et pourquoi c'est la **perte** qui est assertée
et pas `u` : remonter le temps est mal posé, la diffusion efface les hautes fréquences et rien ne
peut les restituer.

Le seul gradient non écrit est celui de `dt/h²`, qui lui demanderait une réduction globale ; un
`static_assert` sur `grad_for_coef.is_valid()` le dit à la compilation plutôt que de rendre un
zéro silencieux.

## Ce qui a marché sans rien demander à personne

- **Les noms traversent.** Un argument de l'appel arrive sous son nom Python (`args.temperature`),
  et un axe déclaré `"y"` s'écrit `y` dans le noyau. Rien à déclarer deux fois.
- **Le VJP.** L'adjoint est branché par `driver.grad` sans qu'on déclare quoi que ce soit, et les
  catégories `NoneTensor` / `ZeroTensor` font vraiment tomber les termes à la compilation.
- **La racine des en-têtes** est le répertoire du `.py` par défaut : un paquet tiers n'a rien à
  déclarer, et loom ne connaît pas ses usagers par leur nom.
- **Les fabriques de tenseurs.** `IntTensor[ cellule ].iota()`, `RealTensor[ y, x ].ones()`,
  `.linspace( 0, 1, x )`, `.random( seed = 3 )` : la forme vient des axes, le type vient de la
  classe, et `driver` n'apparaît plus nulle part sauf `grad` / `jit` / `call`.

## Les frictions, dans l'ordre où on les rencontre

0. ~~**`compilation.register_include_root( ... )`**~~ **Réglé** : c'était une incantation de
   module, prononcée avant tout autre import et sans rapport visible avec le noyau qui en a besoin.
   Un noyau sait où sont ses en-têtes, et le dit dans l'appel : `FfiCode( include_roots = [ ... ] )`.

1. **Tout s'appelle `sdot`.** Le C++ de loom vit dans `namespace sdot` ; les variables
   d'environnement sont `SDOT_BUILD_DIR`, `SDOT_CACHE_DIR`, `SDOT_KERNELS`, `SDOT_EXTERNALS` ; le
   cache est `~/.cache/sdot` ; les messages d'erreur disent « sdot: » ; le functor par défaut d'un
   `FfiCode` sans nom s'appelait `sdot_fwd_kernel` (le nom est obligatoire depuis). On installe loom et on reçoit sdot. Un
   simple renommage — mais c'est la première chose que l'étranger voit.

2. ~~**`driver.array( [ 0, 1, 2 ] )` rend des flottants.**~~ **Réglé** : le chemin documenté est la
   classe, qui EST la déclaration de type (`IntTensor[ x ]( [ 0, 1, 2 ] )` ne peut pas se tromper),
   et `driver` est redevenu la couche basse. La friction était de passer par lui.

3. ~~**Le batch d'un appel ne vient que des agrégats.**~~ et ~~**« Qui suis-je ? » n'a pas de
   réponse.**~~ **Dissoutes**, deux fois — et c'est instructif, parce que ni l'une ni l'autre n'était le
   problème. Les deux étaient des symptômes d'une seule cause : **le lancement était implicite.**

   Le scaffold écrivait toujours `run_parallel( queue, global_batch_indices, ... )`, et
   `global_batch_indices` ne se remplit que des axes de `vmap`. Un noyau dont le parallélisme
   n'est pas un axe de `vmap` — une grille cartésienne, parcourue en (j, i) — n'avait donc aucun
   moyen de le dire. D'où la cascade : fabriquer un axe de batch plat de `ny · nx`, **matérialiser
   un `iota` int64 de n² éléments** pour porter le rang, l'envelopper dans un agrégat-prétexte
   (`Cellules`, dont la docstring admettait qu'il ne servait à rien), et redécouper `j = p / n,
   i = p % n` dans les **deux** noyaux — pour retrouver deux coordonnées que la donnée avait déjà.
   Sur une grille 128², cela faisait 131 ko alloués et traversés par appel, 40 appels par gradient,
   pour calculer une division euclidienne.

   Le remède est que **l'utilisateur lance lui-même**, sur le domaine de la donnée :

   ```cpp
   queue.run_parallel( UnPas(), args.suivant.axes(), args, batch_axes );
   ```

   `Cellules`, `new_batch_axis`, l'`iota`, le rang, le décodage : tout a disparu. Et « qui
   suis-je ? » ne se pose plus, parce que l'item EST le multi-indice qu'on a demandé — nommé, donc
   lisible par `coords[ y ]` au lieu d'une position à compter.

   **Puis une seconde fois, plus loin.** Le remède ci-dessus faisait encore écrire le foncteur et
   le lancement *en Python*, sous forme de chaînes. La bonne réponse était plus simple : le C++ de ce
   noyau vit dans **ses propres en-têtes**, foncteurs compris, et Python n'en dit qu'une ligne. Un
   foncteur y déclare ses propres paramètres au lieu de les hériter de l'ordre des kwargs, et le
   fichier se compile et se teste **sans loom**. Les deux mécanismes que loom avait gagnés pour
   l'étape précédente (`functors = { ... }` et un `launch` injecté) sont devenus inutiles le jour
   même, et ont été retirés.

   Ce que ça coûte, et il faut le dire : un corps qui lance lui-même ignore
   `global_batch_indices`, donc **il ne participe plus à `vmap`** tout seul. Pour cet exemple c'est
   gratuit (il ne se `vmap` pas) ; en général il faudrait composer les deux domaines.

5. ~~**Pas d'`arange`, pas de `linspace`, pas de `ones`.**~~ **Réglé**, et sur les tenseurs plutôt
   que sur le driver : `zeros`, `ones`, `full`, `iota`, `linspace`, `random`, qui lisent leur forme
   dans les AXES — il n'y a donc pas de forme à répéter.

6. **Une compilation par taille de grille** (`CtShapeVar` grave l'extent dans la source) — ici
   c'est voulu, le stencil y gagne, mais rien ne le dit — **et une compilation par motif de
   dérivation**. **Mesuré, puis réglé** : `LOOM_JOURNAL=1` dit maintenant combien de noyaux ont
   été fabriqués et *pourquoi chacun était neuf*. Il a immédiatement montré que la chaîne de dix
   pas en compilait **trente**, dont vingt-huit ne différaient que par le nom de l'axe de batch
   (`cellule_0` … `cellule_19`) — les axes d'une chaîne sont vivants en même temps, donc chacun
   empruntait un indice différent, et ça croissait linéairement avec la longueur de la chaîne.
   Les axes de batch sont désormais nommés **à l'abaissement** (`batch_0`, `batch_1`, …), donc
   deux appels identiques rendent la même source : **30 noyaux → 3**, 135 s → 17 s. Les deux
   variantes du backward qui restent sont de vraies variantes (`temperature : out -> in`, selon
   que le pas lit une entrée perturbée ou une constante).

7. **Un tampon de sortie n'est pas garanti à zéro.** On ne l'apprend que dans un commentaire de
   `loom/tests/test_call.py`. Un étranger qui écrit un backward gardé par
   `if constexpr ( surely_null )` et qui omet la branche `else` rend des ordures, silencieusement.

8. **Les fabriques avalaient leurs mots-clés.** `Parametrized.__getattr__` versait *tout* kwarg
   dans les `template_kwargs` : `RealTensor[ x ].random( seed = 7 )` tirait donc une valeur
   différente à chaque appel, en silence — et `full( v )` ne marchait que parce que son argument
   est positionnel. Corrigé : la signature de la fabrique départage.

9. **Le driver Torch n'existe pas.** C'est la trouvaille la plus lourde, et elle a d'abord été
   *ratée ici* : `--driver torch` sélectionne un ENVIRONNEMENT, pas un driver, et cet
   environnement a jax installé — les deux premières exécutions, « jax » et « torch », étaient
   donc toutes deux du `JaxDriver`. Forcé par `SDOT_FRAMEWORK=torch`, rien ne démarre :
   `TorchDriver` n'a ni `available_gpus`, ni `array`, ni `grad`, ni `vjp`, ni `vmap`, ni `jit` —
   **ni `call`**, qui est pourtant LE point d'entrée. Ce qu'il porte à la place
   (`optimize_using_lbfgs`, `to_nanobind_compatible_objects`, `linalg_solve`) date d'un autre
   design. Aujourd'hui loom est une bibliothèque **Jax**.

10. **Importer l'exemple demande une chirurgie de `sys.path`** : `./run test` importe le fichier de
   test par son chemin, et le module voisin n'est pas trouvable. Sans conséquence ici, mais c'est
   la première ligne du fichier.

## Ce que l'exercice dit de loom

Le cœur — `tensor` + `drivers` + `compilation` — a encaissé un usage franchement étranger **sans
qu'une ligne de loom change** pour le faire marcher : le modèle tient, et c'est le meilleur
argument dont on dispose pour dire que loom n'est pas un sous-produit de sdot.

Mais la moitié de la promesse n'est pas tenue (9) : il n'y a qu'un driver. Tant que le chemin
Torch n'existe pas, « interface homogène pour qui a du C++ à brancher sur Jax **ou** Torch » ne
peut pas être la phrase d'accroche — et c'est justement la phrase qui justifierait une vie
indépendante. Les autres frictions sont au bord : un renommage (1), des constructeurs côté
appelant (2, 4, 5, 8 — réglés), une hypothèse structurelle (3), de la documentation (6, 7, 10).
