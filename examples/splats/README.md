# `splats` — un index découvert à l'exécution, et ce que loom achète exactement

Du splatting gaussien 2D en mélange additif : deuxième **usager délibérément étranger** de loom,
après [`diffusion`](../diffusion/). Il n'importe que `loom`, son C++ ne connaît que
`<loom/support/...>`, et il a été écrit pour éprouver le ragged.

Il l'a éprouvé, et la conclusion n'est pas celle qu'on attendait. Ce document dit ce que loom
apporte *ici*, ce qu'il n'apporte pas, et les chiffres des deux.

```
include/splats/rendu.h   le C++ qu'on avait DÉJÀ : le rendu, son adjoint, les deux index
splats.py                les deux représentations de l'index, et le rendu sur chacune
reference_jax.py         ce qu'on écrirait sans loom, et ce que la borne coûte
test_splats.py           six tests et un bench
```

```bash
errand test_splats          # les tests
errand -k bench test_splats # les temps ( prend la machine pour lui seul )
```

## Le cas

    image( p ) = Σ_i  opacité_i · max( exp( −q_i(p)/2 ) − exp( −k²/2 ), 0 ) · couleur_i
    q_i(p)     = a_i dx² + 2 b_i dx dy + c_i dy²

Des gaussiennes **anisotropes** (l'inverse de covariance a trois paramètres), sommées sur une image.
Un pixel ne doit regarder que les splats qui l'atteignent, sinon le rendu est en O(pixels × splats).
D'où un index `tuile → splats`, dont la longueur **dépend des données** : elle n'est connue qu'après
avoir regardé où sont les splats et quelle taille ils ont.

C'est là toute la question. **Qui décide de la taille de cet index, et quand ?**

## Ce que loom achète, et la frontière qu'il ne franchit pas

Une seule chose, mais elle n'existe nulle part ailleurs : **l'hôte peut lire un compte qu'un noyau
vient d'écrire, et dimensionner l'allocation suivante avec**. C'est ce qui rend le chemin CSR
*exact* — aucune borne à deviner :

```python
comptes = IntTensor[ tuile ]()
driver.call( _COMPTER, ..., comptes = comptes, output_attributes = [ "comptes" ] )

c = numpy.asarray( comptes.tensor )      # <- le compte, sur l'hôte
total = int( c.sum() )                   #    la taille EXACTE de la liste
ids = IntTensor[ Axis( ShapeVar( total ) ) ]()
```

Et sa variante, quand on préfère deviner : le noyau écrit le compte **voulu**, signale que la
capacité n'a pas tenu, et l'appel recommence avec plus de place.

```
capacité demandée 1 -> retenue 384   ( max par tuile 381 )
```

XLA ne sait faire ni l'un ni l'autre. Il sait *construire* un CSR (un compte, une somme préfixe, un
scatter) — il ne sait pas en **connaître le total** : sous `jit` ce compte est un tracer, et rien ne
peut le lire sur l'hôte. La liste doit donc être dimensionnée par une borne choisie avant de tracer :
trop petite, elle perd des splats en silence ; assez sûre, elle fait payer le pire cas à tout le
monde.

**La frontière, telle qu'on la croyait :** ces deux capacités sont *eager-only*. `ShapeArray`
refuse explicitement un tracer, et `capacity_overflows()` rend `None` sous `jit` — « its content only
exists at execution time, so no Python loop can look at it and try again ». Sous `jit`, il faudrait
donc prescrire la capacité, et l'avantage ne vaudrait que pour du code eager.

> **Correction (2026-09-26).** Cette frontière n'est pas une propriété d'XLA : c'est une propriété
> de faire la relecture **en Python**. Déplacée dans le C++ du handler, elle disparaît. Un handler
> tourne à l'exécution, donc il peut lire un compte que son propre noyau vient d'écrire sur la
> carte, et allouer exactement dessus dans le pool d'XLA — **sous `jit` comme en eager**. C'est
> mesuré, sur GPU, dans `tests/test_scratch_gpu.py` : une fonction compilée **une** fois, des
> tailles allouées de 1989 / 2023 / 2040 / 2074 selon le tirage.
>
> Ce que ça change pour cet exemple : l'index n'a à être ni une sortie ni une capacité devinée,
> **à condition que sa construction et sa consommation tiennent dans un seul appel**. Ce qui reste
> vrai de la limite ci-dessus, c'est ce qui doit traverser la frontière Python : une forme de
> **sortie**, elle, est toujours fixée au traçage.

Le reste de ce que loom apporte ici n'est pas propre au ragged, et c'est tant mieux : `rendu.h` ne
connaît de loom que `HD` et `SI` et s'indexe positionnellement ; l'adjoint est écrit à la main, en
accumulation atomique, et passe `check_grad` sur ses quatre familles de paramètres ; les deux
représentations d'index partagent le **même** calcul par splat (`contribution()`), ce qui est ce qui
rend leur comparaison honnête.

## Les chiffres

**Ce que la borne coûte à XLA**, sur des scènes en amas dont les tailles couvrent une décade — le
régime réel, du détail fin et du fond flou (2000 splats, 512×512) :

| couples (splat, pixel) | | |
|---|---|---|
| ce qui est utile — la somme des empreintes | 9 749 623 | — |
| une fenêtre fixe par splat, R = 90 **dicté par le plus gros** | 65 522 000 | ×6,7 |
| la somme dense, chaque pixel voit tous les splats | 524 288 000 | ×53,8 |

**Les deux représentations d'index**, chiffrées au mieux pour chacune :

| | entiers d'index | index | rendu | total |
|---|---|---|---|---|
| rembourré, capacité idéale 214 | 219 136 | 4,68 ms | 27,32 ms | 32,0 ms |
| **CSR, taille exacte** | **43 147** | 6,61 ms | 22,60 ms | **29,2 ms** |

Sur trois scènes, le rembourré coûte **×2,38, ×2,59, ×5,08** la mémoire du CSR. En temps il gagne la
construction (une passe au lieu de deux, et pas d'aller-retour hôte) et perd le rendu (entre 0 et
17 % selon les exécutions — deux mesures, ×1,00 puis ×0,83 : pas assez pour trancher, probablement
son empreinte douze fois plus grande).

**Et le rendu domine** : 27 des 32 ms. Le choix de représentation pèse quelques pour cent du total.
C'est une décision de **mémoire**, pas de vitesse.

Un dernier chiffre, contre-intuitif. Le coût de la passe d'index croît linéairement avec la capacité
**allouée**, à travail utile constant : 4,7 / 5,8 / 8,8 ms pour ×1 / ×2 / ×4. Ce n'est **pas** le
zérotage du tampon — mesuré en le supprimant, les temps ne bougent pas. C'est l'**allocation**, que
XLA refait à chaque appel. Une capacité trop généreuse ne devient pas gratuite en s'abstenant de
l'écrire.

## Ce que l'exemple a démoli

Sa première version n'avait qu'une représentation — le ragged rembourré — et citait comme argument le
gaspillage d'une « capacité fixe ». C'était se tirer dans le pied : ce gaspillage est celui de *notre*
représentation, pas d'une limite de XLA. **Pour ce problème, le ragged rembourré est la moins bonne
des deux structures.**

Et le constat qui compte le plus pour loom : le ragged (`ShapeVar[ "tuile" ]`, les `dep_axes`) a été
développé pour les cellules de Laguerre, et **`grep dep_axes sdot/src` ne rend rien** — sdot ne s'en
sert nulle part. Cet exemple est le premier usage réel, et il conclut qu'un CSR ferait mieux. C'est
une information sur loom, pas sur le splatting.

## Ce que l'exercice a rapporté à loom

- **Le critère « sortie accumulée » portait sur le mauvais axe** : il était conditionné à
  `dtype.floating_point`, parce que le seul cas connu était un gradient. Un compteur **entier**
  accumulé par tous les splats l'a démenti — sans zérotage il rend un total indéterminé qui devient
  une *taille d'allocation* : `overflow in static extent product: dimensions=[2421069375325856419]`.
  Corrigé : le critère est « partagée ».
- **Le semis automatique est porteur**, et l'exemple l'a prouvé en crashant sans lui.
- **`ShapeVarView` n'a pas de `reserve()`** : réserver une fente atomiquement dans une liste ragged
  est le geste même du scatter ragged, et il faut l'écrire à la main en exposant quatre détails
  internes (`.view.ref()`, `.max`, `.errors`, `.id`).
- **« Qui suis-je ? » revient** : deuxième exemple, même agrégat-prétexte pour porter le rang plat.
- **Pas de scan** : la somme préfixe est sur l'hôte, en numpy. Acceptable sur un vecteur de 1024
  tuiles, à écrire soi-même dès qu'il serait grand.

## Une leçon de modèle, pas de loom

Le premier adjoint était faux, et la **structure** des écarts a donné le diagnostic sans débogage :
exact à onze chiffres sur `couleurs` et `opacités`, faux de 5,4 % sur `centres` et 0,8 % sur
`cov_inv`. Les exacts sont ceux qui n'entrent pas dans l'exponentielle ; les faux sont ceux qui
**déplacent la frontière** de troncature, où le poids sautait de `opacité · 1,1·10⁻²` à zéro — ce
qu'une différence finie mesure comme un saut divisé par 2ε.

Le remède n'est pas d'élargir la tolérance mais de supprimer le saut : on retranche la valeur au
seuil, la troncature devient continue, et les quatre redeviennent exacts. Ce n'était pas un artefact
de test — une frontière qui saute met du bruit dans chaque pas de descente.

À ne pas manquer dans l'adjoint : la **valeur** passe par `(e − E)`, la **dérivée en géométrie** par
`e` seul, la constante retranchée ne dépendant pas de `q`.

## La suite

**Le ragged n'est pas une propriété du `ShapeVar`, c'est un STOCKAGE.** Cet exemple a dû écrire deux
boucles de rendu — `ids( t, k )` contre `ids_plat( offsets( t ) + k )` — pour un calcul identique.
C'est le signe que la représentation fuit dans le corps du noyau, alors que `loom/tensor/storage.py`
existe déjà pour exactement ça : « one object per way a value can be backed ». Un `Padded` et un
`Csr` y seraient deux variantes de plus, et le corps écrirait `ids.row( t )( k )` sans savoir
laquelle. Le test d'acceptation est simple, et l'exemple le rate aujourd'hui : **le corps ne doit pas
changer quand le stockage change.**

Et pour le rendu lui-même, la **composition alpha** : c'est l'algorithme vrai, elle rend la liste
*ordonnée* nécessaire (tri par profondeur, adjoint qui remonte la liste en suivant la transmittance)
et elle exercerait le lancement **coopératif** — un work-group par tuile, la liste en mémoire
locale — qui n'a aujourd'hui qu'un seul usager dans tout le dépôt.
