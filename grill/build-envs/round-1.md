# Environnements de build génériques : round 1

> Laisse **Réponse** vide pour accepter la recommandation. `?` demande plus de détail.
> Écris n'importe quoi d'autre (réponse, objection, commentaire) sous une question, puis Envoyer.

Contexte : aujourd'hui `loom/release/envs.py` choisit pour chaque plateforme un env errand d'après la machine de départ
et des valeurs clé/valeur (`loom_linux_host`, `loom_linux_scratch`, `loom_windows_host`, …) lues dans `errand-vars.py`
puis `~/.config/loom/builders.toml`. Une plateforme impossible n'est pas déclarée, et `envs.missing` dit pourquoi.
La CI (GitHub Actions) appelle `make_wheels` directement : elle n'est pas concernée par ce qui suit.

L'idée de fond que je propose : un env `loom-<platform>` = **un lieu** (les couches de l'utilisateur : où ça tourne)
**+ la chaîne d'outils** (les couches de loom : micromamba + clang, image manylinux, VM Windows). L'utilisateur ne
décrit que le lieu ; loom ajoute le reste.

---

## Q1 : Ce que l'utilisateur déclare

(a) **Un lieu par plateforme, explicitement** : « le Linux, c'est ce `Ssh` ; le Windows, c'est cette machine KVM ».
Rien n'est sondé au chargement d'errand (pas de réseau à chaque `errand envs`).

(b) **Une liste de machines**, loom sonde chacune par ssh (`uname`, `apptainer`, `/dev/kvm`) pour savoir ce qu'elle
peut construire. Plus magique, mais coûte une connexion par machine (à mettre en cache) et rend le choix moins lisible.

(c) Les deux : (a) gagne, (b) pour ce qui n'est pas déclaré.

Sur la machine de départ, la détection locale actuelle reste (Mac → macOS ici ; Linux avec apptainer/docker → Linux ici).

➡️ **Recommandé :** (a). Prévisible, sans réseau au chargement ; la vérification de ce qu'il y a là-bas se fait au
premier usage (Q5).

**Réponse :**

---

## Q2 : Où vivent ces déclarations

(a) **Fichiers utilisateur d'errand**, nouvelle fonction générique d'errand : `~/.config/errand/errand-*.py`, lus pour
**tous** les projets, avant ceux du projet (le `errand-vars.py` du projet peut toujours les surcharger). Remplace
`~/.config/loom/builders.toml`, qui disparaît. Utile au-delà de loom (« mes machines », pour n'importe quel projet).

(b) Un fichier propre à loom, `~/.config/loom/builders.py`.

(c) Seulement le `errand-vars.py` / `errand-envs.py` du projet (à recopier dans chaque projet).

➡️ **Recommandé :** (a), avec le `errand-*.py` du projet qui garde le dernier mot.

**Réponse :**

Je préfère que ça soit par projet dans un fichier non surveillé, avec un message pour dire ce qui est attendu si pas trouvé.

---

## Q3 : La forme d'une déclaration

Des couches errand, sous une clé lue par `errand.value`, par plateforme :

```python
# ~/.config/errand/errand-vars.py
import errand

loom_builders = {
    "linux-x86_64":   [ errand.Ssh( host = "build-box" ),
                        errand.Apptainer( scratch = "/data/tmp", flags = [ "--bind", "/data" ] ) ],
    "windows-x86_64": [ errand.Ssh( host = "vm-host" ),
                        errand.Windows( cpus = 16, ram = "32G" ) ],
    "macos-arm64":    [ errand.Ssh( host = "mac-mini" ) ],
}
```

Règle : loom **complète** les couches qu'il aurait mises et que l'utilisateur a déjà données (ici l'`Apptainer` reçoit
image + recette de loom, en gardant `scratch` et `flags` de l'utilisateur ; le `Windows` reçoit python + provision), et
**ajoute** celles qui manquent. Toutes les options actuelles (`loom_linux_scratch`, `_binds`, `_cache`,
`loom_windows_cpus`, `_ram`, `loom_mamba_root`) deviennent des arguments des couches elles-mêmes, plus aucune clé ad hoc.

Alternative : l'utilisateur ne donne que le lieu (`Ssh`), et les réglages de la chaîne d'outils passent par un dict
d'options à côté (`loom_builder_options = { "linux-x86_64": { "scratch": … } }`).

➡️ **Recommandé :** couches complétées par loom, comme ci-dessus. Une seule notion (la couche), qu'errand documente déjà.

**Réponse :**

Oui pour le principe. Par contre, j'aime bien les dataclass ou tout autre outil qui permet de fonctionner avec de l'autocomplétion pour éviter de devoir aller tout chercher dans la doc

---

## Q4 : Le répertoire distant d'un lieu partagé par plusieurs projets

Un `Ssh( host = … )` déclaré une fois pour tous les projets ne peut pas porter le `root` d'un projet. Proposition :
`root` absent → loom met `loom-release/<projet>` (comme aujourd'hui) ; `root` donné → c'est le **parent** des projets
(`<root>/<projet>`), pour mettre tout ça sur un disque scratch.

Alternative : un gabarit, `root = "scratch/{project}"`.

➡️ **Recommandé :** `root` = parent, `<projet>` ajouté par loom.

**Réponse :**

---

## Q5 : Des couches qui disent ce qui manque (générique, dans errand)

Deux choses distinctes :

1. **Plateforme sans lieu** : au lieu de ne pas déclarer l'env (+ `envs.missing`), loom le déclare avec une nouvelle
   couche errand, `errand.Missing( "un Mac joignable par ssh", hint = "loom_builders[ 'macos-arm64' ] = [ errand.Ssh( host = … ) ] dans ~/.config/errand/errand-vars.py" )`.
   `errand envs` l'affiche (`missing: un Mac …`), et le lancer échoue tout de suite avec ce message, avant tout rsync.
   `dispatch_wheels` n'a plus de logique à lui : il lit les `Missing` des envs demandés.

2. **Lieu qui n'a pas ce qu'il faut** : chaque couche déclare ce qu'elle demande à l'endroit où elle tourne
   (`Apptainer` → la commande `apptainer` ; `Windows` → `qemu-system-x86_64` et `/dev/kvm` ; `Micromamba` → Darwin
   pour le Mac de loom…), vérifié en une ligne shell par lieu avant de construire, avec une erreur du genre
   « sur build-box : apptainer introuvable ». Générique, dans `errand/layers` (`requires( ctx )` ou équivalent).

➡️ **Recommandé :** les deux, dans errand.

**Réponse :**

---

## Q6 : Demander à l'utilisateur

Quand `dispatch_wheels` tombe sur une plateforme `Missing` dans un terminal :

(a) loom **demande** (« Adresse ssh d'un Mac ? ») et **propose d'écrire** la déclaration dans
`~/.config/errand/errand-vars.py` (o/N), puis continue ;

(b) loom **affiche** seulement les lignes exactes à coller, et s'arrête ;

(c) (a) en terminal, (b) sans terminal (CI, détaché).

➡️ **Recommandé :** (c). Écrire du Python dans un fichier utilisateur reste acceptable s'il s'agit d'ajouter un bloc
balisé à la fin, jamais de réécrire.

**Réponse :**

Avant un message, on peut pré-ecrire un fichier, par exemple avec une adresse à compléter ou la possibilité de dire "je ne compile pas sur Mac". L'utilisateur ira compléter lui-même dans le fichier (éditer dans le terminal n'est pas confortable)

---

## Q7 : Windows natif

Aujourd'hui Windows = VM QEMU sur un Linux KVM. Faut-il aussi accepter **une vraie machine Windows** (`Ssh` vers
Windows, ou démarrage depuis Windows : build ici) dans ce chantier ? Il faudrait une chaîne d'outils Windows sans VM
(llvm-mingw + python installés par `windows-tools.ps1` sur la machine elle-même). Même question pour **macOS sans Mac**
(impossible légalement en VM hors Mac) et **Linux depuis un Mac** (Docker en émulation x86 : lent mais possible).

➡️ **Recommandé :** **Windows local oui** (démarré depuis Windows, `windows-x86_64` = `make_wheels` ici, comme le
Mac : une ligne). Windows distant natif et Linux-sur-Mac : pas maintenant, mais le design le permet (la chaîne
d'outils dépend du type de lieu).

**Réponse :**

Il faudrait idéalement tout faire

---

## Q8 : Compatibilité

`loom_linux_host` & co. et `~/.config/loom/builders.toml` : on les supprime sèchement (seul toi les utilises, ton
`errand-vars.py` de sdot est réécrit dans le nouveau format), ou on les lit encore avec un avertissement ?

➡️ **Recommandé :** suppression sèche, je te donne le nouveau bloc pour ton fichier (je ne touche pas à
`sdot-ffi/errand-vars.py` sans ton accord).

**Réponse :**

---
