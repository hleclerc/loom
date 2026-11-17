# Generic build environments: the design

## In one sentence

`loom-<platform>` = **a place** ( errand layers the user declares, per project, in an untracked `loom-builders.py`
that loom pre-writes with blanks ) **+ loom's toolchain** ( layers loom adds, or completes when the user already gave
them ). A platform that cannot be reached is still declared, as an errand `Missing` layer that says what to write where.

## loom-builders.py ( project root, untracked, the user's )

```python
# loom-builders.py -- written by loom, then yours: never versioned ( it names your machines ), never rewritten.
from loom.release.builders import Builders, Here, Skip
from errand import Ssh, Apptainer, Docker, Windows

builders = Builders(
    macos_arm64    = Here(),                                  # this Mac
    linux_x86_64   = [ Ssh( host = "" ) ],                    # TO FILL: a Linux machine with Apptainer or Docker; or Skip()
    #                [ Docker( flags = [ "--platform", "linux/amd64" ] ) ]   # here, emulated: several times slower
    windows_x86_64 = [ Ssh( host = "" ), Windows() ],         # TO FILL: a Linux machine with KVM and QEMU; or Skip()
)
```

- `Builders`: a dataclass, one field per platform loom knows ( autocompletion, a typo is an error when read ).
- A field holds `Here()` ( this machine ), a list of errand layers ( a place, plus any toolchain layer to tune ),
  `Skip( "why" )` ( not built from here: `dispatch_wheels all` leaves it out, saying so ), or `None` ( what loom finds
  on its own ).
- Written by loom when absent ( on the first `dispatch_wheels`, or `errand envs` ), and added to `.gitignore` when there
  is a git. The template puts `Here()` where the starting machine can build, a blank `Ssh( host = "" )` elsewhere, the
  alternatives in comments. A blank host = `Missing( "complete linux_x86_64 in loom-builders.py: …" )`.

## Completion rules ( loom )

| platform | place given | toolchain loom adds / completes |
|---|---|---|
| macos-arm64 | `Here()` on a Mac, or `Ssh` to one | `Micromamba` ( conda-forge clang ), `Vars` |
| linux-x86_64 | `Here()` ( Apptainer, else Docker found here ), or `Ssh` | `Apptainer` or `Docker` ( image + recipe filled in; the user's `scratch`, `flags`… kept ) |
| windows-x86_64 | `Here()` on Windows ( native: clang from `windows-tools.ps1` ), `Here()` on a Linux with KVM, or `Ssh` to one | `Windows` VM ( python, provision, `also` filled in; the user's `cpus`, `ram`… kept ) |

- `Ssh.root` absent → `loom-release/<project>`; given → the parent, `<root>/<project>`. `also` is loom's.
- Not reachable yet ( needs errand to run on Windows, or to reach a native Windows by ssh ): an explicit `Missing`
  ( "errand cannot yet …" ). That work is a separate session.
- The old keys ( `loom_linux_host`, … ) and `~/.config/loom/builders.toml` are gone.

## What a project needs on top ( versioned, `pyproject.toml` )

```toml
[tool.loom.release.linux-x86_64]
pip   = [ "pyfftw" ]
setup = "scripts/linux-extra.sh"      # appended to the recipe: %post of the image
[tool.loom.release.macos-arm64]
conda = [ "fftw" ]                    # into the micromamba environment
[tool.loom.release.windows-x86_64]
setup = "scripts/windows-extra.ps1"   # one more provision step of the VM
```

Replacing a recipe entirely: documented as reading loom's ( `loom/release/recipes/` ), and redefining the environment in
the project's `errand-envs.py` ( already possible ).

## errand ( generic )

- **`Missing( what, hint = None )`**: a layer for an environment that cannot be reached from here. `errand envs` shows
  `missing: <what>`; selecting it for a run fails at once, before any push or build, with `what` and `hint`.
  `dispatch_wheels` loses its own bookkeeping ( `envs.missing` ): it reads the `Missing` layers.
- **`requires( ctx )`** on a layer: `[ ( shell test, what is missing ) ]`, checked in one shell line at the place
  ( here, or over ssh ) before anything is built. `Apptainer` → `apptainer`; `Docker` → `docker`; `Windows` →
  `qemu-system-x86_64`, `qemu-img`, `/dev/kvm` writable; `Micromamba` → `micromamba` ( unless `executable` ).
  Error: `on build-box: apptainer not found ( Apptainer )`.
- `Apptainer.image` / `Docker.image` may be left empty in a declaration that something completes; used empty, it is a
  clear error.
