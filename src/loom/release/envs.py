"""loom's BUILD ENVIRONMENTS for errand: one per platform, under a fixed name, so that a project has nothing to declare.

    loom-macos-arm64     loom-linux-x86_64     loom-windows-x86_64

`provide` declares them ( an errand provider: `errand.providers`, or the `errand.envs` entry point of loom's wheel ), for a
project whose `pyproject.toml` has a `[tool.loom.release]` with `platforms`. Each runs `python -m loom.release make_wheels`
at a PLACE, with loom's TOOLCHAIN:

    the place      where the user says, in `loom-builders.py` ( see `builders.py` ): `Here()`, errand layers ( `Ssh` ... ),
                   `Skip()`; or, said nowhere, what this machine does on its own
    the toolchain  loom's layers after it, or the user's own layers of the same kinds, completed:
                       macos-arm64     Micromamba ( clang from conda-forge )                  on a Mac
                       linux-x86_64    Apptainer, or Docker, of loom's recipe ( manylinux_2_28 )
                       windows-x86_64  a Windows VM errand boots ( a Linux with KVM ), a Windows machine reached by ssh
                                       ( `WindowsBox` ); on Windows itself, its own clang

A platform with no place is declared anyway, as an `errand.Missing` that says what to write where; `dispatch_wheels`
refuses it before anything runs, and `loom-builders.py` is written with blanks to fill when there is none.

What goes in each environment comes from the project: its dependencies ( `[project] dependencies` ), the frameworks of
`[tool.loom.release] frameworks`, its extras ( `[tool.loom.release.<platform>]`: `pip`, `conda` on macOS, a `setup` script
added to the recipe on Linux and to the provisioning on Windows ), and loom -- the checkout this module comes from ( copied
beside the project, put on `PYTHONPATH` ), or the released version in use. The toolchain and its pins are loom's
( `recipes/` ).

A project changes one in its own `errand-envs.py`, read after: `errand.envs[ "loom-linux-x86_64" ] = ...` replaces it,
`errand.envs[ "loom-linux-x86_64" ].stack.append( errand.Vars( ... ) )` adds to it, and `environment( project, platform,
place )` builds a variant.
"""
from pathlib import Path
import os
import platform as _platform
import shutil
import sys

PLATFORMS = ( "macos-arm64", "linux-x86_64", "windows-x86_64" )

# The Jax kernels declare the XLA FFI API version of the jaxlib they are compiled with, and a jaxlib refuses a NEWER one ( the
# catalogue then compiles them instead, see `compilation/catalogue.py` ): the wheels are built with a jaxlib old enough for
# about everyone. 0.10.2 = API 0.3, which jaxlib 0.10 to 0.11.0 have and any later one accepts.
JAX = "0.10.2"
PYTHON = "3.13"
WINDOWS_PYTHON = "3.13.16"
RECIPES = Path( __file__ ).resolve().parent / "recipes"
GENERATED = Path( ".loom" ) / "release"          # in the project: the recipes written for it, the images built from them


def env_name( platform: str ) -> str:
    return f"loom-{ platform }"


def this_platform():
    """The platform of the machine errand starts on, among `PLATFORMS` ( None for another one )."""
    system, machine = _platform.system(), _platform.machine().lower()
    if system == "Darwin" and machine in ( "arm64", "aarch64" ):
        return "macos-arm64"
    if system == "Linux" and machine in ( "x86_64", "amd64" ):
        return "linux-x86_64"
    if system == "Windows" and machine in ( "x86_64", "amd64" ):
        return "windows-x86_64"
    return None


# ── what goes in an environment ───────────────────────────────────────────────

def extras( project, platform: str ) -> dict:
    """`[tool.loom.release.<platform>]`: `pip`, `conda`, `setup` ( a script of the project )."""
    got = project.extras.get( platform, {} )
    unknown = set( got ) - { "pip", "conda", "setup" }
    if unknown:
        raise ValueError( f"[tool.loom.release.{ platform }]: unknown { ', '.join( sorted( unknown ) ) } ( pip, conda, setup )" )
    return got


def packages( project, platform: str ) -> list:
    """The pip requirements of the project's environments on `platform`: its dependencies ( loom aside ), its frameworks,
    and loom when it is not a checkout."""
    from . import loom_checkout, loom_version
    import re
    deps = [ d for d in project.dependencies if not re.match( r"loom-ffi\b", d ) ]
    linux = platform == "linux-x86_64"
    fw = { "jax": f"jax[cuda13]=={ JAX }" if linux else f"jax=={ JAX }", "numpy": "numpy", "torch": "torch",
           "cupy": "cupy-cuda13x" if linux else None }
    frameworks = [ fw.get( f, f ) for f in project.frameworks if fw.get( f, f ) ]
    loom = [] if loom_checkout() is not None or project.is_loom else [ f"loom-ffi=={ loom_version() }" ]
    return list( dict.fromkeys( [ "numpy", "ninja", "wheel", "twine", *deps, *frameworks, *loom,
                                  *extras( project, platform ).get( "pip", [] ) ] ) )


def _rel( root: Path, path: Path ) -> str:
    return os.path.relpath( path, root ).replace( os.sep, "/" )


def _siblings( project ):
    """What the work needs beside the project, as paths relative to it: the loom checkout ( when it is one ), and errand's
    ( `errand run` starts errand again inside an environment ). Relative paths only: an installed one is not pushed."""
    from . import loom_checkout
    import errand
    out = []
    checkout = loom_checkout()
    if checkout is not None and not project.is_loom:
        out.append( ( "loom", checkout.root, "src" ) )
    errand_root = Path( errand.__file__ ).resolve().parents[ 1 ]
    if ( errand_root / "pyproject.toml" ).is_file():
        out.append( ( "errand", errand_root, "" ) )
    return out


def _write( path: Path, text: str ):
    path.parent.mkdir( parents = True, exist_ok = True )
    if not path.is_file() or path.read_text() != text:
        path.write_text( text )


def _here_python_path( project, sep = os.pathsep ) -> str:
    return sep.join( str( root / sub ) if sub else str( root ) for _, root, sub in _siblings( project ) )


def _there( project ):
    """`also` and `PYTHONPATH` for a copy of the project at `root` over there, the siblings at the same relative places."""
    also, path = [], []
    for _, sibling, sub in _siblings( project ):
        rel = _rel( project.root, sibling )
        if rel.startswith( ".." ) and not os.path.isabs( rel ):
            also.append( rel )
            path.append( f"{ rel }/{ sub }" if sub else rel )
    return also, path




# ── the toolchains ────────────────────────────────────────────────────────────

def _ssh( project, ssh ):
    """The user's `Ssh`, completed: `root` is the parent of the project's copy ( `loom-release` when not said ) -- or the
    copy itself when it already ends with the project's name --, `also` the siblings the work needs."""
    from dataclasses import replace
    also, _ = _there( project )
    parent = ( ssh.root or "loom-release" ).rstrip( "/" )
    root = parent if parent.rsplit( "/", 1 )[ -1 ] == project.root.name else f"{ parent }/{ project.root.name }"
    return replace( ssh, root = root, also = list( dict.fromkeys( [ *ssh.also, *also ] ) ) )


def _take( given: list, *kinds ):
    """The first of the user's layers of one of `kinds`, out of `given`, or None."""
    for i, layer in enumerate( given ):
        if isinstance( layer, kinds ):
            return given.pop( i )
    return None


def macos_arm64( project, ssh = None, given = () ):
    """A micromamba environment with clang from conda-forge ( the adjoint kernels need P2280, which Apple's clang lacks ):
    here on a Mac, or over `ssh` on one. A `Micromamba` in `given` is completed ( its `root_prefix`, `executable` kept )."""
    import errand
    from dataclasses import replace
    given = list( given )
    extra = extras( project, "macos-arm64" )
    mamba = errand.Micromamba( f"loom-{ project.wheel_name }-macos-arm64", python = PYTHON, channels = [ "conda-forge" ],
                               packages = [ "clangxx", *extra.get( "conda", [] ) ], pip = packages( project, "macos-arm64" ) )
    mine = _take( given, errand.Micromamba )
    if mine is not None:
        mamba = replace( mine, name = mine.name or mamba.name, python = mine.python or PYTHON,
                         channels = list( dict.fromkeys( [ *mine.channels, "conda-forge" ] ) ),
                         packages = [ *mamba.packages, *mine.packages ], pip = [ *mamba.pip, *mine.pip ] )
    flags = { "LOOM_CXX": "clang++", "LOOM_CXXFLAGS": "-std=c++23" }
    if ssh is None:
        path = _here_python_path( project )
        return errand.Env( [ mamba, errand.Vars( { **flags, **( { "PYTHONPATH": path } if path else {} ) } ), *given ] )
    _, path = _there( project )
    return errand.Env( [ _ssh( project, ssh ), mamba,
                         errand.Vars( { **flags, **( { "PYTHONPATH": ":".join( path ) } if path else {} ) } ), *given ] )


def _setup( project, platform ) -> str:
    """The text of the project's `setup` script for `platform`, or ''."""
    script = extras( project, platform ).get( "setup" )
    return ( project.root / script ).read_text() if script else ""


def linux_recipe( project, pip = () ) -> str:
    """The project's copy of loom's Apptainer recipe, its packages on the pip line and its `setup` at the end of `%post`:
    written into the project ( `.loom/release/` ), so that it is pushed with it and its content is what the image's
    fingerprint is taken of."""
    text = ( RECIPES / "manylinux.def" ).read_text().replace(
        "{pip}", " ".join( f'"{ p }"' for p in [ *packages( project, "linux-x86_64" ), *pip ] ) )
    setup = _setup( project, "linux-x86_64" )
    if setup:
        text = text.replace( "\n%environment", "    # the project's setup ( [tool.loom.release.linux-x86_64] setup )\n"
                             + "".join( f"    { l }\n" for l in setup.splitlines() ) + "\n%environment", 1 )
    path = GENERATED / "linux-x86_64.def"
    _write( project.root / path, text )
    return path.as_posix()


def linux_dockerfile( project, pip = () ) -> str:
    """The same recipe for Docker ( `.loom/release/linux-x86_64.Dockerfile` ), x86-64 whatever the machine."""
    text = ( RECIPES / "manylinux.Dockerfile" ).read_text().replace(
        "{pip}", " ".join( f'"{ p }"' for p in [ *packages( project, "linux-x86_64" ), *pip ] ) )
    setup = _setup( project, "linux-x86_64" )
    if setup:
        text += "\n# the project's setup ( [tool.loom.release.linux-x86_64] setup )\nRUN <<'LOOM_SETUP'\nset -e\n" + setup.rstrip() + "\nLOOM_SETUP\n"
    path = GENERATED / "linux-x86_64.Dockerfile"
    _write( project.root / path, text )
    return path.as_posix()


def _engine():
    return next( ( e for e in ( "apptainer", "docker", "podman" ) if shutil.which( e ) ), None )


def linux_x86_64( project, ssh = None, given = (), engine = None ):
    """An image of loom's recipe ( manylinux_2_28 ): Apptainer, or Docker / Podman when that is what `given` holds or
    `engine` says; here, or over `ssh`. The user's container layer is completed: image and recipe are loom's, the rest
    ( `scratch`, `flags`, `build_on` ... ) theirs."""
    import errand
    from dataclasses import replace
    given = list( given )
    mine = _take( given, errand.Apptainer, errand.Docker )        # Podman is a Docker
    kind = type( mine ) if mine is not None else { "docker": errand.Docker, "podman": errand.Podman }.get( engine, errand.Apptainer )
    gpu = ssh is not None or shutil.which( "nvidia-smi" )
    if issubclass( kind, errand.Docker ):
        # x86-64 on any machine: emulated on an arm64 one
        flags = [ "--platform", "linux/amd64" ]
        container = replace( mine or kind(), image = f"loom-{ project.wheel_name }-linux-x86_64",
                             recipe = linux_dockerfile( project ),
                             flags = [ *flags, *( mine.flags if mine else [] ) ] )
    else:
        image = ( GENERATED / "linux-x86_64.sif" ).as_posix()
        container = replace( mine or errand.Apptainer(), image = image, recipe = linux_recipe( project ), fakeroot = True,
                             flags = list( dict.fromkeys( [ *( [ "--nv" ] if gpu else [] ), *( mine.flags if mine else [] ) ] ) ) )
    if ssh is None:
        path = _here_python_path( project )
        return errand.Env( [ container, errand.Vars( { "PYTHONPATH": path } if path else {} ), *given ], cuda = True )
    _, path = _there( project )
    return errand.Env( [ _ssh( project, ssh ), container, errand.Vars( { "PYTHONPATH": ":".join( path ) } if path else {} ),
                         *given ], cuda = True )


WINDOWS_TOOLS = r"C:\tools"
_LLVM_MINGW = "llvm-mingw-20261006-ucrt-x86_64"          # as `recipes/windows-tools.ps1` unpacks it


def _windows_vars( path_sep_paths ):
    import errand
    python = rf"C:\Program Files\Python{ WINDOWS_PYTHON.replace( '.', '' )[ :3 ] }"
    return errand.Vars( { "PATH": WINDOWS_TOOLS + rf"\{ _LLVM_MINGW }\bin;" + WINDOWS_TOOLS + r"\mingw64\bin;%PATH%",
                          "LOOM_CXX": "clang++", "LOOM_CXXFLAGS": "-std=c++23",
                          "LOOM_CACHE_DIR": r"E:\loomcache",           # E: is kept from one boot to the next, C: starts clean
                          "SSL_CERT_FILE": python + r"\Lib\site-packages\certifi\cacert.pem",
                          **( { "PYTHONPATH": ";".join( p.replace( "/", "\\" ) for p in path_sep_paths ) } if path_sep_paths else {} ) } )


def windows_x86_64( project, ssh = None, given = () ):
    """Windows Server under QEMU, booted by errand on a Linux machine with KVM: this one, or over `ssh`. clang ( llvm-mingw )
    builds the kernels as DLLs with the C++ runtime linked in, GNU make drives the build: both installed by loom's
    `windows-tools.ps1` when the image is made. The user's `Windows` layer is completed ( `cpus`, `ram`, `data` ... kept )."""
    import errand
    from dataclasses import replace
    given = list( given )
    tools = GENERATED / "windows-tools.ps1"
    _write( project.root / tools, ( RECIPES / "windows-tools.ps1" ).read_text() )
    also, path = _there( project )
    python = rf"C:\Program Files\Python{ WINDOWS_PYTHON.replace( '.', '' )[ :3 ] }"
    pip_line = ( f"& '{ python }\\python.exe' -m pip install --no-warn-script-location certifi "
                 + " ".join( f'"{ p }"' for p in packages( project, "windows-x86_64" ) ) )
    setup = extras( project, "windows-x86_64" ).get( "setup" )
    mine = _take( given, errand.Windows ) or errand.Windows( cpus = 8, ram = "16G" )
    vm = replace( mine, python = mine.python or WINDOWS_PYTHON, also = list( dict.fromkeys( [ *mine.also, *also ] ) ),
                  provision = [ pip_line, tools.as_posix(), *( [ setup ] if setup else [] ), *mine.provision ] )
    return errand.Env( [ *( [ _ssh( project, ssh ) ] if ssh is not None else [] ), vm, _windows_vars( path ), *given ] )


BOX_VENV = r"C:\errand\venv"          # the venv errand makes on a `WindowsBox`, which its commands run with


def windows_box( project, ssh, given = () ):
    """A Windows machine of the user's, reached by `ssh`: errand's `WindowsBox`, completed as the VM is -- loom's Python
    ( installed by uv into C:\\errand, a venv of it made there by errand ), the packages and loom's toolchain
    ( `windows-tools.ps1`, into C:\\tools ) as its provisioning, run by `errand setup`. The user's `Ssh` is kept as
    it is: the box puts the project in C:\\errand\\work."""
    import errand
    from dataclasses import replace
    given = list( given )
    tools = GENERATED / "windows-tools.ps1"
    _write( project.root / tools, ( RECIPES / "windows-tools.ps1" ).read_text() )
    also, path = _there( project )
    pip_line = ( f"& '{ BOX_VENV }\\Scripts\\python.exe' -m pip install --no-warn-script-location certifi "
                 + " ".join( f'"{ p }"' for p in packages( project, "windows-x86_64" ) ) )
    setup = extras( project, "windows-x86_64" ).get( "setup" )
    mine = _take( given, errand.WindowsBox )
    box = replace( mine, python = mine.python or WINDOWS_PYTHON, also = list( dict.fromkeys( [ *mine.also, *also ] ) ),
                   provision = [ pip_line, tools.as_posix(), *( [ setup ] if setup else [] ), *mine.provision ] )
    v = _windows_vars( path )
    v.values.update( { "LOOM_CACHE_DIR": r"C:\errand\loom-cache",
                       "SSL_CERT_FILE": BOX_VENV + r"\Lib\site-packages\certifi\cacert.pem" } )
    return errand.Env( [ ssh, box, v, *given ] )


def windows_native( project, given = () ):
    """On Windows itself: the interpreter errand starts with ( holding what `make_wheels` needs, as in a CI ), and loom's
    clang, unpacked once by `windows-tools.ps1` into C:\\tools."""
    import errand
    if not ( Path( WINDOWS_TOOLS ) / _LLVM_MINGW / "bin" / "clang++.exe" ).is_file():
        return _missing( "windows-x86_64", rf"loom's Windows toolchain in { WINDOWS_TOOLS }",
                         f"run once: powershell -NoProfile -File { RECIPES / 'windows-tools.ps1' }" )
    v = _windows_vars( _here_python_path( project, ";" ).split( ";" ) if _here_python_path( project, ";" ) else [] )
    v.values.pop( "LOOM_CACHE_DIR" )
    v.values.pop( "SSL_CERT_FILE" )
    return errand.Env( [ v, *given ] )


# ── from a place to an environment ────────────────────────────────────────────

def _missing( platform, what, hint = None ):
    import errand
    from . import builders
    return errand.Env( [ errand.Missing( what, hint = hint or builders.hint( platform ) ) ] )


WHAT = { "macos-arm64": "a Mac ( here, or reachable by ssh )",
         "linux-x86_64": "a Linux x86-64 machine with Apptainer or Docker ( here, or reachable by ssh )",
         "windows-x86_64": "a Linux machine with KVM and QEMU, where errand boots Windows ( here, or reachable by ssh ); "
                           "a Windows machine reachable by ssh ( WindowsBox ); or Windows itself" }


WHAT_SSH = { "macos-arm64": "a Mac reachable by ssh",
             "linux-x86_64": "a Linux x86-64 machine reachable by ssh, with Apptainer or Docker",
             "windows-x86_64": "a machine reachable by ssh: a Linux with KVM and QEMU, where errand boots Windows "
                               "( then Windows() ), or a Windows ( then WindowsBox() )" }


def _kvm() -> bool:
    return os.access( "/dev/kvm", os.R_OK | os.W_OK )


def can_here( platform: str ) -> bool:
    """Does this machine build `platform` itself, with nothing said?"""
    here = this_platform()
    if platform == "macos-arm64":
        return here == platform
    if platform == "linux-x86_64":
        return here == platform and _engine() is not None
    if platform == "windows-x86_64":
        return here == platform or ( here == "linux-x86_64" and _kvm() and shutil.which( "qemu-system-x86_64" ) is not None )
    return False


def environment( project, platform: str, place ):
    """`loom-<platform>` at `place` ( a field of `Builders` ): an `errand.Env`, a `Missing` one when it cannot be reached."""
    import errand
    from . import builders as B
    if platform not in PLATFORMS:
        return _missing( platform, f"a platform loom has a recipe for ( it knows { ', '.join( PLATFORMS ) } )",
                         f"[tool.loom.release] platforms" )
    if isinstance( place, B.Skip ):
        return _missing( platform, f"nothing: { platform } is skipped in { B.FILE }" + ( f" ( { place.why } )" if place.why else "" ),
                         f"`{ platform.replace( '-', '_' ) } = Skip()` in { B.FILE }: built elsewhere" )
    here = this_platform()
    if place is None:
        if not can_here( platform ):
            return _missing( platform, WHAT[ platform ] )
        place = B.Here()
    given = [] if isinstance( place, B.Here ) else list( place )
    ssh = given.pop( 0 ) if given and isinstance( given[ 0 ], errand.Ssh ) else None
    if ssh is not None:
        if not ssh.host:
            return _missing( platform, WHAT_SSH[ platform ],
                             f"the host of `{ platform.replace( '-', '_' ) }` is blank in { B.FILE }: fill it in, or Skip()" )
        if platform == "windows-x86_64" and any( isinstance( l, errand.WindowsBox ) for l in given ):
            return windows_box( project, ssh, given )
        if platform == "macos-arm64":
            return macos_arm64( project, ssh, given )
        if platform == "linux-x86_64":
            return linux_x86_64( project, ssh, given )
        return windows_x86_64( project, ssh, given )
    # here
    if platform == "macos-arm64":
        if here != platform:
            return _missing( platform, "a Mac: macOS wheels are built on one", f"`macos_arm64 = [ Ssh( host = ... ) ]` in { B.FILE }" )
        return macos_arm64( project, None, given )
    if platform == "linux-x86_64":
        if any( isinstance( l, ( errand.Apptainer, errand.Docker ) ) for l in given ):
            return linux_x86_64( project, None, given )
        engine = _engine()
        if engine is None:
            return _missing( platform, "Apptainer, Docker or Podman on this machine",
                             f"install one, or `linux_x86_64 = [ Ssh( host = ... ) ]` in { B.FILE }" )
        return linux_x86_64( project, None, given, engine = engine )
    if here == "windows-x86_64" and not any( isinstance( l, errand.Windows ) for l in given ):
        return windows_native( project, given )
    if here != "linux-x86_64":
        return _missing( platform, "KVM: a Windows virtual machine is booted on Linux x86-64",
                         f"`windows_x86_64 = [ Ssh( host = ... ), Windows() ]` in { B.FILE }" )
    return windows_x86_64( project, None, given )


def provide( root: Path ):
    """The errand provider: `loom-<platform>` for each platform the project at `root` asks for, at the place
    `loom-builders.py` says -- written, with blanks, when there is none."""
    from . import Project, builders as B
    import errand
    # installed, loom is asked by every errand project: only one that says `platforms` gets anything
    toml = Path( root ) / "pyproject.toml"
    if not toml.is_file() or "[tool.loom.release]" not in toml.read_text():
        return
    project = Project( root )
    if not project.platforms:
        return
    if not B.inside_an_environment():
        written = B.write_template( project.root, this_platform(), { p: can_here( p ) for p in PLATFORMS } )
        if written is not None:
            print( f"loom: wrote { written }: where each platform's wheel is built, blanks to fill in", file = sys.stderr )
    try:
        said = B.read( project.root ) or B.Builders()
    except Exception as err:
        for platform in project.platforms:
            errand.envs[ env_name( platform ) ] = _missing( platform, f"a { B.FILE } that reads: { err }", "fix it, or delete it for a fresh one" )
        return
    for platform in project.platforms:
        errand.envs[ env_name( platform ) ] = environment( project, platform, said.of( platform ) if platform in PLATFORMS else None )


def skipped( root ) -> dict:
    """platform -> why, for those `loom-builders.py` says `Skip()`."""
    from . import builders as B
    try:
        said = B.read( root )
    except Exception:
        return {}
    if said is None:
        return {}
    return { p: said.of( p ).why for p in PLATFORMS if isinstance( said.of( p ), B.Skip ) }
