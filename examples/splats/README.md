# `splats` — le ragged, sur un cas où le JIT doit deviner

Du splatting gaussien 2D en mélange additif : un deuxième **usager délibérément étranger** de loom,
après [`diffusion`](../diffusion/). Il n'importe que `loom`, son C++ ne connaît que
`<loom/support/...>`, et il est là pour exercer ce que `diffusion` ne touchait pas.

```
include/splats/rendu.h   le C++ qu'on avait DÉJÀ : le rendu et son adjoint
splats.py                deux passes, trois agrégats
reference_jax.py          ce qu'un usager compétent écrirait sans loom, et ce que ça coûte
test_splats.py           les tests
```

```bash
errand test_splats
```

## Ce que ça fait

    image( p ) = Σ_i  opacité_i · max( exp( −q_i(p)/2 ) − exp( −k²/2 ), 0 ) · couleur_i
    q_i(p)     = a_i dx² + 2 b_i dx dy + c_i dy²

Des gaussiennes **anisotropes** (l'inverse de covariance a trois paramètres), mélange **additif** :
pas d'ordre, donc pas de tri — la composition alpha viendra après.

Un pixel ne doit regarder que les splats qui l'atteignent. D'où un index `tuile → splats`, dont la
longueur **dépend des données**. C'est lui l'objet de l'exemple.

## Ce que le JIT peut, et ce qu'il ne peut pas

L'argument serait malhonnête avec un homme de paille, alors disons-le exactement : **XLA sait
construire cet index** — un compte, une somme préfixe, un scatter de taille fixe. Ce qu'il ne sait
pas, c'est le **découvrir** : toute borne doit être choisie avant de tracer. Une borne trop petite
perd des splats en silence ; une borne sûre fait payer le pire cas à tout le monde.

`reference_jax.py` chiffre les deux bornes qu'il faut choisir, sur des scènes en amas dont les
tailles couvrent une décade (ce qui est le régime réel : du détail fin et du fond flou) :

| 2000 splats, 512×512 | couples (splat, pixel) | rapport |
|---|---|---|
| ce qui est utile — la somme des empreintes | 9 749 623 | — |
| une **fenêtre fixe**, R = 90 dicté par le plus gros splat | 65 522 000 | **×6,7** |
| la somme **dense**, chaque pixel voit tous les splats | 524 288 000 | **×53,8** |

Et sur la capacité par tuile : moyenne 41, max 214 — un tableau `[tuiles, capacité]` gaspille
**×5,2**. loom ne choisit pas cette capacité, il la découvre :

```
capacité demandée 1 -> retenue 384   ( max par tuile 381 )
```

Le noyau écrit le compte **voulu**, signale que la capacité n'a pas tenu, et l'appel recommence avec
plus de place — jusqu'à 384. C'est la machinerie qu'aucun framework n'a, et c'est la première fois
qu'un usager l'exerce.

## Ce qui a marché

- **Le ragged est déclaratif et il est bien fait.** `nb_par_tuile : ShapeVar[ "tuile" ]` — un compte
  par cellule, qui porte *par cellule* sa valeur, sa capacité (`max`) et le tampon d'erreurs de
  l'appel. `.value` rend un `ShapeArray` de rang 1. Rien à assembler à la main.
- **Le semis automatique était exactement ce qu'il fallait.** Les `grad_for_splats.*` sont des
  sorties partagées qu'un adjoint atomique accumule : elles partent à zéro d'office. Sans ça,
  l'adjoint aurait été faux sans que rien ne le dise.
- **Deux passes dans une fonction**, qui se partagent les agrégats, sans cérémonie.

## Les frictions

1. **`ShapeVarView` n'a pas de `reserve()`.** Réserver une fente atomiquement dans une liste ragged
   — plusieurs work-items écrivant dans la *même* liste — est un besoin général, et il faut
   aujourd'hui l'écrire à la main en exposant quatre détails internes :

   ```cpp
   auto cellule = compte( t );
   auto &compteur = cellule.view.ref();
   const SI fente = atomic_fetch_add( compteur, TC( 1 ) );
   if ( fente < cellule.max ) ids( t, fente ) = i;
   else cellule.errors.record( ErrorKind::capacity_overflow, cellule.id, fente + 1 );
   ```

   `set()` a son pendant capacité-vérifié ; `reserve()` manque, et c'est le geste du scatter ragged.

2. **« Qui suis-je ? » revient.** Deuxième exemple, même agrégat-prétexte (`Rangs`) pour porter le
   rang plat de l'item, parce que l'échafaudage injecte `batch_index` / `thread_index` /
   `nb_threads` mais pas le rang, et que le batch d'un appel ne vient que des agrégats. Deux
   exemples sur deux : ça mérite d'être réglé.

3. **Un piège latent, que l'exemple a frôlé.** La règle « sortie accumulée » de
   `CallArg_Tensor.cpp_seed_member` est conditionnée à `dtype.floating_point`. Un tenseur
   d'**entiers** accumulé à travers le batch serait donc *empoisonné* sous
   `LOOM_ZERO_OUTPUTS=poison` au lieu d'être mis à zéro. Ici le compteur est un `ShapeVar` (zéroté
   inconditionnellement), donc on passe à côté — mais le critère devrait être « partagée », pas
   « flottante ».

## Une leçon qui n'est pas de loom mais du modèle

Le premier adjoint était faux, et la *structure* des écarts a donné le diagnostic : exact à onze
chiffres sur `couleurs` et `opacités`, faux de 5,4 % sur `centres` et de 0,8 % sur `cov_inv`. Les
deux exacts sont ceux qui n'entrent pas dans l'exponentielle ; les deux faux sont ceux qui
**déplacent la frontière** de troncature, où le poids sautait de `opacité · 1,1·10⁻²` à zéro.

Le remède n'est pas d'élargir la tolérance mais de supprimer le saut : on retranche la valeur au
seuil, la troncature devient continue, et les quatre redeviennent exacts. Ce n'était pas qu'un
artefact de test — une frontière qui saute met du bruit dans chaque pas de descente de gradient.

Conséquence à ne pas manquer dans l'adjoint : la **valeur** passe par `(e − E)`, la **dérivée en
géométrie** par `e` seul, puisque la constante retranchée ne dépend pas de `q`.

## La suite

La **composition alpha** : c'est l'algorithme vrai, et elle rend la liste *ordonnée* nécessaire
(donc un tri par profondeur, et un adjoint qui remonte la liste à l'envers en suivant la
transmittance). Elle exercerait en plus le lancement **coopératif** — un work-group par tuile, la
liste en mémoire locale — qui n'a aujourd'hui qu'un seul usager dans tout le dépôt.
