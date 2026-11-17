"""WHERE each platform's wheel is built, as the user says it: `loom-builders.py` at the root of the project.

Never versioned -- it names the user's machines -- and the user's: loom writes it when it is absent, with what the machine
it starts on can do already filled in and blanks for the rest, then only reads it.

    from loom.release.builders import Builders, Here, Skip
    from errand import Ssh, Apptainer, Windows

    builders = Builders(
        macos_arm64    = Here(),
        linux_x86_64   = [ Ssh( host = "build-box", root = "/scratch/me" ), Apptainer( scratch = "/data/tmp" ) ],
        windows_x86_64 = Skip( "no KVM machine" ),
    )

Each field is a PLACE: `Here()`, a list of errand layers ( an `Ssh` first to go elsewhere, then any layer of loom's
toolchain to tune: loom completes it rather than adding its own ), `Skip()`, or `None` -- what loom finds on its own.
`envs.py` turns them into the `loom-<platform>` environments.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import os

FILE = "loom-builders.py"


@dataclass
class Here:
    """This machine: the one errand starts on."""


@dataclass
class Skip:
    """This platform is not built from here: `dispatch_wheels all` leaves it out, and says so."""
    why: str = ""


@dataclass
class Builders:
    """One field per platform loom builds wheels for: `Here()`, `[ errand layers ]`, `Skip( "why" )`, or None ( loom
    looks for a way on its own ). A field name is a platform with `_` for `-`."""
    macos_arm64   : Here | Skip | list | None = None
    linux_x86_64  : Here | Skip | list | None = None
    windows_x86_64: Here | Skip | list | None = None

    def of( self, platform: str ):
        return getattr( self, platform.replace( "-", "_" ) )

    def __post_init__( self ):
        for f in fields( self ):
            v = getattr( self, f.name )
            if isinstance( v, tuple ):
                setattr( self, f.name, v := list( v ) )
            if not ( v is None or isinstance( v, ( Here, Skip, list ) ) ):
                raise TypeError( f"Builders( { f.name } = { v!r} ): Here(), a list of errand layers, Skip(), or None" )


def path( root ) -> Path:
    return Path( root ) / FILE


def read( root ) -> Builders | None:
    """What `loom-builders.py` says, or None when there is none. A file that does not define `builders` is an error that
    says what it should hold."""
    p = path( root )
    if not p.is_file():
        return None
    scope = { "__file__": str( p ), "__name__": "loom_builders" }
    exec( compile( p.read_text(), str( p ), "exec" ), scope )
    got = scope.get( "builders" )
    if not isinstance( got, Builders ):
        raise ValueError( f"{ p } must define `builders = Builders( ... )` ( from loom.release.builders ), "
                          f"got { type( got ).__name__ }" )
    return got


# ── the template ──────────────────────────────────────────────────────────────

_HEAD = '''\
# Where loom builds the wheels of this project ( `python -m loom.release dispatch_wheels` ): written by loom, now yours.
# Not versioned -- it names your machines -- and never rewritten: delete it to get a fresh one.
#
# Each platform is `Here()`, a list of errand layers ( `Ssh` first to build elsewhere; loom adds its toolchain after it,
# or completes the toolchain layers you give: `Apptainer( scratch = ... )`, `Windows( cpus = ... )` ), `Skip( "why" )`
# not to build it from here, or None for what loom finds on its own. A blank host is said when it is needed.
from loom.release.builders import Builders, Here, Skip
from errand import Ssh, Apptainer, Docker, Micromamba, Windows, WindowsBox, Vars

builders = Builders(
'''


def template( here: str | None, can_here: dict ) -> str:
    """The text of a fresh `loom-builders.py` for a machine of platform `here`, `can_here[ platform ]` saying whether
    this machine builds that platform itself."""
    lines = []
    # macos-arm64
    if can_here.get( "macos-arm64" ):
        lines.append( "    macos_arm64    = Here()," )
    else:
        lines += [ '    macos_arm64    = [ Ssh( host = "" ) ],                     # a Mac reachable by ssh; or Skip( "no Mac" )' ]
    # linux-x86_64
    if can_here.get( "linux-x86_64" ):
        lines.append( "    linux_x86_64   = Here()," )
    else:
        lines.append( '    linux_x86_64   = [ Ssh( host = "" ) ],                     # a Linux x86-64 machine with Apptainer ( or give Docker() ); or Skip()' )
        if here == "macos-arm64":
            lines.append( "    # linux_x86_64 = [ Docker() ],                              # here, x86-64 emulated: several times slower" )
    lines.append( '    #   with Ssh( host = "box", root = "/scratch/me" ), Apptainer( scratch = "/scratch/tmp", flags = [ "--bind", "/data" ] ),' )
    lines.append( '    #   Vars( { "LOOM_CACHE_DIR": "/scratch/me/loom-cache" } )' )
    # windows-x86_64
    if can_here.get( "windows-x86_64" ):
        lines.append( "    windows_x86_64 = Here()," )
    else:
        lines.append( '    windows_x86_64 = [ Ssh( host = "" ), Windows( cpus = 8, ram = "16G" ) ],   # a Linux machine with KVM and QEMU, where errand boots Windows; or Skip()' )
        lines.append( '    #   more cpus and ram: a faster build' )
        lines.append( '    # windows_x86_64 = [ Ssh( host = "" ), WindowsBox() ],                # a Windows machine reachable by ssh ( OpenSSH server )' )
    return _HEAD + "\n".join( lines ) + "\n)\n"


def write_template( root, here: str | None, can_here: dict ) -> Path | None:
    """`loom-builders.py` when there is none ( -> its path ), added to the `.gitignore` of a git checkout. Never over
    one that is there."""
    p = path( root )
    if p.exists():
        return None
    p.write_text( template( here, can_here ) )
    root = Path( root )
    if ( root / ".git" ).exists():
        ignore = root / ".gitignore"
        text = ignore.read_text() if ignore.is_file() else ""
        if not any( l.strip() in ( FILE, "/" + FILE ) for l in text.splitlines() ):
            ignore.write_text( text + ( "" if not text or text.endswith( "\n" ) else "\n" )
                               + f"# where loom builds the wheels: names this machine's hosts\n/{ FILE }\n" )
    return p


def hint( platform: str ) -> str:
    return f"say it in { FILE } ( `{ platform.replace( '-', '_' ) } = ...` ), at the root of the project"


def inside_an_environment() -> bool:
    """Read by errand started again inside an environment ( over ssh, in a container ): nothing is written there."""
    return bool( os.environ.get( "ERRAND_IN_ENV" ) )
