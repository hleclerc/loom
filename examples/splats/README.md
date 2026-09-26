# `splats` — deux représentations d'un index découvert, et laquelle gagne

Du splatting gaussien 2D en mélange additif : un deuxième **usager délibérément étranger** de loom,
après [`diffusion`](../diffusion/). Il n'importe que `loom`, son C++ ne connaît que
`<loom/support/...>`, et il est là pour exercer ce que `diffusion` ne touchait pas — le ragged, un
compte écrit par le noyau, un pipeline à plusieurs passes, un adjoint en accumulation atomique.

```
include/splats/rendu.h   le C++ qu'on avait DÉJÀ : le rendu, son adjoint, les deux index
splats.py                les deux représentations de l'index, et le rendu sur chacune
reference_jax.py         ce qu'un usager écrirait sans loom, et ce que ça coûte
test_splats.py           les tests, dont la comparaison chiffrée
```

```bash
errand test_splats
```

## Ce que ça fait

    image( p ) = Σ_i  opacité_i · max( exp( −q_i(p)/2 ) − exp( −k²/2 ), 0 ) · couleur_i
    q_i(p)     = a_i dx² + 2 b_i dx dy + c_i dy²

Des gaussiennes **anisotropes**, mélange **additif** : pas d'ordre, donc pas de tri — la composition
alpha viendra après. Un pixel ne doit regarder que les splats qui l'atteignent, d'où un index
`tuile → splats` dont la longueur **dépend des données**.

## Ce que je croyais démontrer, et qui était faux

La première version de cet exemple n'avait qu'une représentation — un **ragged rembourré**,
`ids[ tuiles, capacité ]` — et citait comme argument le gaspillage d'une « capacité fixe » : ×2,6,
×5,2. C'était se tirer dans le pied : ce gaspillage est celui de **notre** représentation, pas d'une
limitation de XLA. Un **CSR** (offsets + liste unique) ne gaspille rien sur cet axe.

Alors on l'a écrit, et mesuré. Les deux donnent la même image (à 1e-14 près : les sommes d'une tuile
ne sont pas dans le même ordre, et l'addition flottante n'est pas associative), et le coût se compare
**au mieux pour chacune** — le rembourré chiffré avec la meilleure capacité possible, pas avec une
capacité choisie au hasard :

| 2000 splats, 512×512 | entiers d'index | passes sur les splats |
|---|---|---|
| rembourré, capacité idéale (214) | 219 136 | 1 |
| **CSR, taille exacte** | **43 147** | 2 + une somme préfixe |

et sur les trois scènes mesurées : **×2,38, ×2,59, ×5,08**. Le rembourré est plus coûteux en
mémoire, toujours. Ce qu'il achète en échange est une **passe de moins** sur les splats et aucun
aller-retour vers l'hôte.

**Donc pour ce problème, CSR est la bonne représentation**, et l'exemple le dit au lieu de le cacher.

## Où est la vraie différence avec un JIT — et ce n'est pas le rembourrage

Soyons exacts, parce qu'un homme de paille ne démontre rien. **XLA sait construire un CSR** : un
compte, une somme préfixe, un scatter. Ce qu'il ne sait pas, c'est **connaître le total**. Sous
`jit`, le compte qu'un noyau vient d'écrire est un tracer : rien ne peut le lire sur l'hôte, donc la
liste doit être dimensionnée par une **borne choisie avant de tracer**.

loom peut le lire — `ShapeArray` existe pour ça, et refuse explicitement un tracer avec le message
qui explique pourquoi. D'où :

| | mémoire d'index | borne à choisir | sous `jit` |
|---|---|---|---|
| XLA, CSR borné | `total_max` (à deviner) | oui | oui |
| XLA, fenêtre fixe par splat | `n · (2R+1)²`, R du plus gros | oui | oui |
| loom, **CSR exact** | `total` **mesuré** | **aucune** | non (lecture hôte) |
| loom, rembourré | `tuiles × max` | une devinette, **corrigée** | capacité à prescrire |

Les deux lignes loom sont les deux moitiés de la même capacité : *soit* on lit le compte et on
alloue juste (CSR), *soit* on devine et on se fait corriger. La correction, en action :

```
capacité demandée 1 -> retenue 384   ( max par tuile 381 )
```

Le noyau écrit le compte **voulu**, signale que la capacité n'a pas tenu, et l'appel recommence —
jusqu'à 384. Aucun framework ne fait ni l'un ni l'autre.

Pour situer l'échelle, la borne que XLA doit choisir sur l'autre axe (une fenêtre fixe par splat,
dont le rayon est dicté par le plus gros de la scène, les tailles couvrant une décade) :

| 2000 splats, 512×512 | couples (splat, pixel) | |
|---|---|---|
| ce qui est utile | 9 749 623 | — |
| fenêtre fixe, R = 90 | 65 522 000 | ×6,7 |
| somme dense | 524 288 000 | ×53,8 |

## Ce qui a marché

- **Le ragged est déclaratif.** `nb_par_tuile : ShapeVar[ "tuile" ]` — un compte par cellule, qui
  porte *par cellule* sa valeur, sa capacité (`max`) et le tampon d'erreurs de l'appel.
- **Le semis automatique des sorties était exactement ce qu'il fallait.** Les `grad_for_splats.*`
  sont des sorties partagées qu'un adjoint atomique accumule : elles partent à zéro d'office. Sans
  ça l'adjoint aurait été faux sans que rien ne le dise.
- **Deux, puis quatre passes** dans une fonction, qui se partagent les agrégats, sans cérémonie.

Et une remarque : `grep dep_axes sdot/src` ne rend rien. Le ragged est une capacité de loom que son
seul usager réel n'exerce nulle part.

## Les frictions

1. **`ShapeVarView` n'a pas de `reserve()`.** Réserver une fente atomiquement dans une liste ragged
   est le geste même du scatter ragged, et il faut l'écrire à la main en exposant quatre détails
   internes (`.view.ref()`, `.max`, `.errors`, `.id`). `set()` a son pendant capacité-vérifié.

2. **« Qui suis-je ? » revient.** Deuxième exemple, même agrégat-prétexte pour porter le rang plat.
   Deux sur deux : ça mérite que l'échafaudage l'injecte.

3. **Un piège latent.** La règle « sortie accumulée » de `CallArg_Tensor.cpp_seed_member` est
   conditionnée à `dtype.floating_point` : un tenseur d'**entiers** accumulé à travers le batch
   serait *empoisonné* sous `LOOM_ZERO_OUTPUTS=poison`. Ici les compteurs sont des `ShapeVar`
   (zérotés inconditionnellement), donc on passe à côté — mais le critère devrait être « partagée ».

4. **La somme préfixe est sur l'hôte**, en numpy, parce qu'elle porte sur un vecteur de la taille du
   nombre de tuiles (256 à 1024). Ce n'est pas là qu'est le travail, mais loom n'offre pas de scan,
   donc un cas où le vecteur serait grand demanderait d'écrire le noyau soi-même.

## Une leçon qui n'est pas de loom mais du modèle

Le premier adjoint était faux, et la **structure** des écarts a donné le diagnostic sans débogage :
exact à onze chiffres sur `couleurs` et `opacités`, faux de 5,4 % sur `centres` et 0,8 % sur
`cov_inv`. Les exacts sont ceux qui n'entrent pas dans l'exponentielle ; les faux sont ceux qui
**déplacent la frontière** de troncature, où le poids sautait de `opacité · 1,1·10⁻²` à zéro — ce
qu'une différence finie mesure comme un saut divisé par 2ε.

Le remède n'est pas d'élargir la tolérance mais de supprimer le saut : on retranche la valeur au
seuil, la troncature devient continue, et les quatre redeviennent exacts. Ce n'était pas un artefact
de test — une frontière qui saute met du bruit dans chaque pas de descente de gradient.

À ne pas manquer dans l'adjoint : la **valeur** passe par `(e − E)`, la **dérivée en géométrie** par
`e` seul, la constante retranchée ne dépendant pas de `q`.

## La suite

La **composition alpha** : c'est l'algorithme vrai, elle rend la liste *ordonnée* nécessaire (tri par
profondeur, adjoint qui remonte la liste en suivant la transmittance) et elle exercerait le
lancement **coopératif** — un work-group par tuile, la liste en mémoire locale — qui n'a aujourd'hui
qu'un seul usager dans tout le dépôt.
