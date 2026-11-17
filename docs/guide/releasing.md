# Releasing a package

A package built on loom ships its kernels precompiled: the wheels carry a catalogue (see [Compilation and catalogue](./compilation)), one wheel per operating system. loom provides the commands that build, check and publish them -- the same for every package, loom itself included:

```bash
python -m loom.release bump_version         # write today's next version
python -m loom.release make_wheels          # the wheel of this machine, into dist/, checked
python -m loom.release publish_wheels       # upload this version's wheels ( TWINE_PASSWORD = the token )
```

They run from the root of the package's checkout. Each one prints what it does.

| command | what it does |
|---|---|
| `show_version` | prints the version ( and, for a package built on loom, the loom it requires and the one in use ); changes nothing |
| `bump_version [--to X]` | writes today's next version ( `year.month.day.number`, `number` counting the releases of the day ), or `X`, in `pyproject.toml` |
| `make_kernels` | the catalogue of this machine: the `catalogue` module, once per framework installed here |
| `make_wheels` | `make_kernels`, then the wheel of this machine, then `check_wheels` |
| `dispatch_wheels all \| P1,P2` | `make_wheels` in the build environment of each platform ( below ), through errand, the wheels brought back into `dist/` |
| `check_wheels [WHEEL...]` | installs the wheel in a clean venv and runs the checks |
| `publish_wheels [--url URL]` | uploads this version's wheels of `dist/` ( PyPI by default ) |

`make_wheels` builds for the machine it runs on, and only for it. A package that ships kernels needs one wheel per platform: one job per platform in a CI ( each runs `make_wheels`, a last one gathers the wheels into `dist/` and runs `publish_wheels` ), or `dispatch_wheels`, which runs it on the machines one has.

The package's `Makefile` can hold one-line reminders of them ( `make version`, `make bump_version`, `make kernels`, `make wheels`, `make check_wheels`, `make publish` ): the work stays in loom.

## What a package says about itself

In its `pyproject.toml`. A pure-Python package needs nothing.

```toml
[tool.loom.release]
package    = "mypkg"                          # the import name ( default: the one package of the wheel, or the project name )
frameworks = ["jax", "numpy", "torch", "cupy"] # those installed on a build machine get kernels
catalogue  = "mypkg.catalogue"                # `python -m mypkg.catalogue` builds the catalogue ( `loom.catalogue.build` )
smoke      = "scripts/smoke.py"               # run on the installed wheel: may compile
checks     = ["docs/snippets/*.py"]           # run on the installed wheel with LOOM_KERNELS=catalogue: nothing may compile
check_with = ["jax"]                          # installed in the clean venv beside the wheels
platforms  = ["macos-arm64", "linux-x86_64", "windows-x86_64"]   # what `dispatch_wheels all` builds
```

## The wheels

A wheel that carries kernels is a platform wheel, `py3-none-<platform>`: the kernels do not depend on the Python version. It carries only the kernels of its operating system, even when the tree holds several ( collected from several machines ). On Linux the platform is `manylinux_2_28`: build in that image. A wheel without kernels stays `py3-none-any`, built once anywhere.

`dist/` keeps the wheels of the current version, whatever their platform, and only those: `publish_wheels` uploads all of them.

## The build environments

`dispatch_wheels` runs through errand, in the environments loom declares for it -- one per platform, under a fixed name: `loom-macos-arm64`, `loom-linux-x86_64`, `loom-windows-x86_64`. Each is **a place** -- where it runs, which only the user knows -- and **loom's toolchain** -- what runs there, the same for every package:

| environment | toolchain | places |
|---|---|---|
| `loom-macos-arm64` | micromamba, clang from conda-forge | this Mac, or a Mac over ssh |
| `loom-linux-x86_64` | an image of loom's recipe ( manylinux_2_28 ), Apptainer or Docker | this machine, or a Linux machine over ssh; on a Mac, Docker emulates x86-64 ( several times slower ) |
| `loom-windows-x86_64` | a Windows virtual machine errand boots, or a Windows machine reached by ssh; clang from llvm-mingw | a Linux machine with KVM and QEMU: this one, or over ssh; a Windows machine over ssh ( `WindowsBox` ); or Windows itself |

The toolchain and its pins are loom's: the compiler, the CUDA compiler, the jax whose kernels every jaxlib accepts.

### Where: `loom-builders.py`

The places go in `loom-builders.py`, at the root of the package. It names the user's machines, so it is never versioned. The first `dispatch_wheels` ( or `errand envs` ) writes it when there is none -- added to the `.gitignore` of a git checkout -- with what this machine builds itself already filled in and blanks for the rest; from then on it is the user's, and loom only reads it:

```python
# loom-builders.py
from loom.release.builders import Builders, Here, Skip
from errand import Ssh, Apptainer, Docker, Micromamba, Windows, WindowsBox, Vars

builders = Builders(
    macos_arm64    = Here(),
    linux_x86_64   = [ Ssh( host = "build-box", root = "/scratch/me" ),
                       Apptainer( scratch = "/scratch/tmp", flags = [ "--bind", "/data" ] ),
                       Vars( { "LOOM_CACHE_DIR": "/scratch/me/loom-cache" } ) ],
    windows_x86_64 = [ Ssh( host = "vm-host" ), Windows( cpus = 16, ram = "32G" ) ],
)
```

`Builders` has one field per platform; each holds

- `Here()`: this machine;
- a list of errand layers: an `Ssh` first to build elsewhere, then, if needed, a layer of loom's toolchain to tune. loom **completes** the toolchain layers it finds -- the image and the recipe of an `Apptainer` or a `Docker`, the name and the packages of a `Micromamba`, the Python and the provisioning of a `Windows` or a `WindowsBox` -- and keeps everything else they say; it adds the ones that are not there. Other layers ( a `Vars` ) come after loom's, and have the last word. The `root` of an `Ssh` is the directory the package's copy goes **in** ( `<root>/<package>`; `loom-release/<package>` when not said; a `root` already ending with the package's name is taken as is ), so one `Ssh` serves every package;
- `Skip( "why" )`: not built from here; `dispatch_wheels all` leaves it out and says so;
- `None`: what loom finds on its own -- this Mac for macOS; Apptainer, Docker or Podman on this Linux machine; KVM and QEMU on this Linux machine, or Windows itself.

A platform with no place -- a blank host, a Mac asked of a Linux machine, no container engine here -- is still declared, as an `errand.Missing` that says what to write where: `errand envs` lists it as `unreachable`, and `dispatch_wheels` stops before anything runs, with that message. A place that lacks a tool is said when it is reached: `on build-box: apptainer, not found`.

A Windows machine of the user's, reached by ssh, is `[ Ssh( host = ... ), WindowsBox() ]`. loom completes the `WindowsBox` as it completes the VM: its Python, a venv of it in `C:\errand`, the packages and loom's toolchain ( into `C:\tools` ). All of it is installed by `errand setup --env loom-windows-x86_64`, run once, and again whenever what it installs changes; `dispatch_wheels` never installs anything there itself. On Windows itself, the environment is the interpreter errand starts with -- holding what `make_wheels` needs, as in a CI -- and loom's clang, which `loom/release/recipes/windows-tools.ps1` unpacks once into `C:\tools`.

### What: the package's extras

What a package needs on top of its dependencies and its `frameworks` is versioned, in `pyproject.toml`, per platform:

```toml
[tool.loom.release.linux-x86_64]
pip   = ["pyfftw"]
setup = "scripts/linux-extra.sh"      # shell lines, at the end of the image's recipe ( as root, in AlmaLinux 8 )

[tool.loom.release.macos-arm64]
conda = ["fftw"]                      # into the micromamba environment, from conda-forge

[tool.loom.release.windows-x86_64]
pip   = ["pywin32"]
setup = "scripts/windows-extra.ps1"   # one more provisioning step of the virtual machine
```

The recipes loom writes for the package go in `.loom/release/`, to be ignored by git. loom's own are in `loom/release/recipes/` ( `manylinux.def`, `manylinux.Dockerfile`, `windows-tools.ps1` ): a package that needs something else entirely starts from one of them, and changes the environment in its own `errand-envs.py`, which errand reads after loom's: `errand.envs[ "loom-linux-x86_64" ] = ...` replaces it, `errand.envs[ "loom-linux-x86_64" ].stack.append( errand.Vars( ... ) )` adds to it, and `loom.release.envs.environment( project, platform, place )` builds a variant.

With errand installed beside loom, `errand envs` and `errand setup --env loom-linux-x86_64` see these environments ( loom declares them under errand's `errand.envs` entry point ). Otherwise `python -m loom.release errand ...` is errand with them declared.

## The checks

`check_wheels` installs the wheel in a fresh venv, with no `PYTHONPATH` to the checkout and an empty cache, then:

1. imports the package, and checks that the package, loom and the C++ include roots all come from the venv -- not from a checkout that would hide a broken wheel;
2. runs `smoke`, which may compile: the headers the wheels ship are enough to build a kernel;
3. runs each of `checks` with `LOOM_KERNELS=catalogue`: a kernel the wheel does not hold is an error that names it. What a user runs first -- the documented examples -- belongs here, so that their kernels are in the catalogue rather than compiled on the user's machine. A check that needs a framework the venv does not have is skipped.

## loom first

A package that ships kernels requires at least the loom they were generated with. `bump_version` writes it: `loom-ffi>=` the version of the loom in use ( the checkout `loom.release` comes from ). `publish_wheels` refuses while that loom is not on PyPI. So when loom changed, release it first:

```bash
cd loom  && make bump_version && make wheels && make publish
cd mypkg && make bump_version && make wheels && make publish     # with the wheels of the other platforms gathered in dist/
```

A newer loom than the one required is fine: a kernel whose source it generates differently is not found in the catalogue, and is compiled.
