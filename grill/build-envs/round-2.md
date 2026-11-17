# Environnements de build génériques : round 2

> Laisse **Réponse** vide pour accepter la recommandation. `?` demande plus de détail.

Acquis du round 1 : déclarations par projet dans un fichier non versionné, typées (autocomplétion), pré-écrites par
loom avec des trous à compléter ; couches `Missing` et vérification de ce qu'il faut sur place dans errand ; viser
toutes les combinaisons.

---

## Q1 : Quel fichier

(a) **`loom-builders.py`** à la racine du projet, propre à loom : loom le lit lui-même (dans son provider), l'écrit
s'il n'existe pas (gabarit à trous) et l'ajoute au `.gitignore` s'il y a un git. Toi seul l'édites ensuite ; loom ne le
réécrit jamais.

(b) Dans `errand-vars.py` (existe déjà, non versionné), une valeur `loom_builders = …`. Mais loom y pré-écrirait un bloc
dans un fichier que tu édites aussi, et errand lit ce fichier pour tout.

➡️ **Recommandé :** (a).

**Réponse :**

---

## Q2 : La forme typée

```python
# loom-builders.py -- written by loom, then yours. Not versioned: it names your machines.
from loom.release.builders import Builders, Here, Skip
from errand import Ssh, Apptainer, Docker, Windows

builders = Builders(
    macos_arm64    = [ Ssh( host = "" ) ],                 # a Mac reachable by ssh, or Skip()
    linux_x86_64   = [ Ssh( host = "build-box", root = "/scratch/me" ),
                       Apptainer( scratch = "/data/tmp", flags = [ "--bind", "/data" ] ) ],
    windows_x86_64 = [ Ssh( host = "vm-host" ), Windows( cpus = 16, ram = "32G" ) ],
)
```

- `Builders` : une dataclass, **un champ par plateforme** que loom connaît (autocomplétion sur les noms, faute de
  frappe = erreur au chargement). Valeur : `Here()` (sur cette machine), une liste de couches errand (elles aussi des
  dataclasses), `Skip( "raison" )` (je ne construis pas cette plateforme), ou `None` = ce que loom trouve tout seul.
- Le gabarit écrit par loom met déjà `Here()` pour ce que la machine de départ sait faire, et pour le reste une ligne à
  trou (`host = ""`) avec en commentaire ce qu'il faut et l'alternative `Skip()`. Un `host` vide → `Missing` qui dit
  « complète `macos_arm64` dans loom-builders.py ».
- Les couches données sont **complétées** par loom (image + recette pour `Apptainer`/`Docker`, python + provision pour
  `Windows`) ; il faut donc que `Apptainer( image = … )` ait un défaut (aujourd'hui `image` est obligatoire) : loom
  passe par une petite factory plutôt que de modifier errand ? Je propose : `image` reste obligatoire dans errand, et
  loom fournit ses propres dataclasses de réglages quand c'est plus simple, ex. `Apptainer` de loom = sous-classe avec
  `image = ""` que loom remplit.

➡️ **Recommandé :** tel quel ; pour `image`, errand accepte `image = None` uniquement si quelqu'un le complète avant
l'usage (sinon erreur claire) — plus simple qu'une sous-classe côté loom.

**Réponse :**

ça m'a l'air bien. Cependant, je m'inquiète des cas où les users voudraient installer des trucs "bizarres". Il faudra voir comment l'autoriser et donner accès aux recettes.

---

## Q3 : Ordre de « tout faire »

Faits vérifiés : la couche `Windows` d'errand marche déjà en premier (Linux local avec KVM) ou derrière un `Ssh` ;
`Ssh` d'errand ne parle qu'à des machines POSIX (lignes `sh`, rsync) ; Docker est installé sur ce Mac (pas Apptainer).

| départ \ cible | macOS | Linux | Windows |
|---|---|---|---|
| Mac | ici ✅ | Linux distant ✅ ; **Docker ici** (émulation amd64, lent) 🆕 | VM sur hôte KVM ✅ ; **Windows natif distant** 🆕🆕 |
| Linux | Mac distant ✅ | ici ✅ | **VM ici** (KVM local) 🆕 ; VM distante ✅ ; Windows natif distant 🆕🆕 |
| Windows | Mac distant 🆕🆕 | **Docker/WSL ici** 🆕🆕 ; Linux distant 🆕🆕 | **ici** 🆕 |

🆕 = petit (loom seul, couches existantes). 🆕🆕 = errand doit d'abord savoir parler à Windows par ssh ou tourner sur
Windows (rsync, sh absents) : un vrai chantier errand.

➡️ **Recommandé :** ce chantier fait le design complet + tous les 🆕 ; les 🆕🆕 donnent un `Missing` explicite
(« errand ne sait pas encore … ») et deviennent une session à part (errand sur/vers Windows).

**Réponse :**

---

## Q4 : Le Linux sur Mac par Docker

Docker sur Mac arm64 construit l'image manylinux x86_64 en émulation (Rosetta/QEMU) : faisable, mais la compilation
des noyaux (et nvcc) sera plusieurs fois plus lente. Le gabarit doit-il proposer **par défaut** `Here()` (Docker local)
pour Linux sur un Mac où Docker est installé, ou une ligne `Ssh` à trou avec Docker local en alternative commentée ?

➡️ **Recommandé :** ligne `Ssh` à trou par défaut, Docker local en commentaire (« lent : émulation »).

**Réponse :**

---
