"""RELEASING a package built on loom -- loom itself included -- with the same commands everywhere:

    python -m loom.release show_version              print the version ( and the loom it requires ), change nothing
    python -m loom.release bump_version [--to X]     write today's next version ( or X ) in pyproject.toml
    python -m loom.release make_kernels              the precompiled kernels of this machine, every framework
    python -m loom.release make_wheels               the wheel of this machine into dist/: kernels, wheel, check
    python -m loom.release dispatch_wheels all       `make_wheels` for every platform, through errand, into dist/
    python -m loom.release dispatch_wheels P1,P2     the same for these platforms only
    python -m loom.release errand ARGS...            errand, with loom's build environments ( `errand setup --env loom-...` )
    python -m loom.release check_wheels [WHEEL...]   install in a clean venv and run the checks
    python -m loom.release publish_wheels [--url U]  upload this version's wheels of dist/ ( TWINE_PASSWORD = the token )

They run from the project's root ( where its `pyproject.toml` is ). What is particular to a project is said there, and a
pure-Python package ( loom ) needs nothing:

    [tool.loom.release]
    package    = "mypkg"                      # the import name ( default: the one package of the wheel, or the project name )
    frameworks = [ "jax", "numpy", "torch" ]  # one catalogue run per framework installed on the build machine
    catalogue  = "mypkg.catalogue"            # the module that builds the catalogue ( `loom.catalogue.build` )
    smoke      = "scripts/smoke.py"           # run in the clean venv, may compile: the wheel's headers work
    checks     = [ "docs/snippets/*.py" ]     # run in the clean venv with `LOOM_KERNELS=catalogue`: nothing may compile
    check_with = [ "jax" ]                    # what the clean venv installs beside the wheels
    platforms  = [ "macos-arm64", "linux-x86_64", "windows-x86_64" ]   # what `dispatch_wheels all` builds ( `envs.py` )

VERSIONS are `year.month.day.number`, the release made today ( `number` counts the releases of the day ). A package that
depends on loom requires at least the loom it is built with: `bump_version` writes `loom-ffi>=` the version of the loom in use (the
checkout this module comes from), and `publish_wheels` refuses while that version is not on PyPI -- loom is released first.

A wheel that carries precompiled kernels is a platform wheel ( `py3-none-<platform>`: the kernels do not depend on the
Python version ), and carries only those of its operating system. A wheel without kernels stays `py3-none-any`.

`make_wheels` builds for the machine it runs on, and only for it: one job per platform in a CI, or a developer's own machine.
`dist/` keeps the wheels of the current version made elsewhere ( brought back next to it ), so that `publish_wheels` sends all.
`dispatch_wheels` runs it in loom's build environments ( `loom-<platform>`, see `envs.py` ) through errand, which brings the
wheels back.
"""
from pathlib import Path
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.request

PYPI_UPLOAD = "https://upload.pypi.org/legacy/"
FRAMEWORK_MODULES = ( "jax", "torch", "cupy" )          # a check of a framework the clean venv does not have is skipped


def _toml( path: Path ) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:                          # python 3.10
        raise SystemExit( "loom.release: needs python >= 3.11 ( tomllib )" )
    return tomllib.loads( path.read_text() )


def normalized( version: str ) -> str:
    """The version as pip writes it ( PEP 440 drops the padding: `2026.11.08.1` -> `2026.11.8.1` )."""
    return ".".join( str( int( p ) ) if p.isdigit() else p for p in version.split( "." ) )


def next_version( current: str, today = None ) -> str:
    """`year.month.day.number`: the release made today, `number` counting the releases of the day ( 1, then 2, ... )."""
    today = today or datetime.date.today()
    stem = f"{ today.year }.{ today.month }.{ today.day }"
    m = re.fullmatch( re.escape( stem ) + r"\.(\d+)", normalized( current ) )
    return f"{ stem }.{ int( m.group( 1 ) ) + 1 if m else 1 }"


def run( *cmd, **kw ):
    print( "+", " ".join( map( str, cmd ) ), flush = True )
    return subprocess.run( [ str( c ) for c in cmd ], check = kw.pop( "check", True ), **kw )


class Project:
    """A package's checkout: its `pyproject.toml`, and what `[tool.loom.release]` says."""

    def __init__( self, root: Path ):
        self.root = Path( root ).resolve()
        self.toml_path = self.root / "pyproject.toml"
        data = _toml( self.toml_path )
        self.name = data[ "project" ][ "name" ]
        self.dependencies = data[ "project" ].get( "dependencies", [] )
        config = data.get( "tool", {} ).get( "loom", {} ).get( "release", {} )
        # the import name: said, or the package hatch puts in the wheel ( `src/loom` ), or the project name
        hatch = data.get( "tool", {} ).get( "hatch", {} ).get( "build", {} ).get( "targets", {} ).get( "wheel", {} ).get( "packages", [] )
        self.package = config.get( "package", Path( hatch[ 0 ] ).name if len( hatch ) == 1 else self.name.replace( "-", "_" ) )
        self.frameworks = list( config.get( "frameworks", [] ) )
        self.catalogue = config.get( "catalogue" )
        self.smoke = config.get( "smoke" )
        self.checks = list( config.get( "checks", [] ) )
        self.check_with = list( config.get( "check_with", [] ) )
        self.platforms = list( config.get( "platforms", [] ) )
        # what one platform's build environment needs on top ( `[tool.loom.release.linux-x86_64]`: pip, conda, setup )
        self.extras = { k: dict( v ) for k, v in config.items() if isinstance( v, dict ) }

    @classmethod
    def here( cls ):
        for d in ( Path.cwd(), *Path.cwd().parents ):
            if ( d / "pyproject.toml" ).is_file():
                return cls( d )
        raise SystemExit( "loom.release: no pyproject.toml here or above" )

    @property
    def version( self ) -> str:
        return re.search( r'(?m)^version = "([^"]*)"', self.toml_path.read_text() ).group( 1 )

    @property
    def wheel_name( self ) -> str:
        """The distribution name as a wheel file spells it."""
        return re.sub( r"[-_.]+", "_", self.name ).lower()

    @property
    def is_loom( self ) -> bool:
        return self.name == "loom-ffi"

    @property
    def depends_on_loom( self ) -> bool:
        return any( re.match( r"loom-ffi\b", d ) for d in self.dependencies )

    @property
    def package_dir( self ) -> Path:
        for d in ( self.root / "src" / self.package, self.root / self.package ):
            if d.is_dir():
                return d
        raise SystemExit( f"loom.release: no package `{ self.package }` under { self.root } ( set `package` in [tool.loom.release] )" )

    @property
    def kernels_dir( self ) -> Path:
        return self.package_dir / "precompiled-kernels"

    def kernel_files( self ):
        return [ f for f in self.kernels_dir.rglob( "*" ) if f.is_file() and f.suffix in ( ".so", ".dylib", ".dll" ) ] \
            if self.kernels_dir.is_dir() else []


def loom_checkout():
    """The loom this module comes from, as a Project, when it is a checkout ( None for an installed wheel )."""
    root = Path( __file__ ).resolve().parents[ 3 ]
    if ( root / "pyproject.toml" ).is_file() and 'name = "loom-ffi"' in ( root / "pyproject.toml" ).read_text():
        return Project( root )
    return None


def loom_version() -> str:
    checkout = loom_checkout()
    if checkout is not None:
        return normalized( checkout.version )
    from importlib.metadata import version
    return normalized( version( "loom-ffi" ) )


def _source_path( p: Project ) -> str:
    """`PYTHONPATH` for what runs from the checkouts: the project's sources, and loom's."""
    checkout = loom_checkout()
    parts = [ str( p.package_dir.parent ) ]
    if checkout is not None and not p.is_loom:
        parts.append( str( checkout.root / "src" ) )
    return os.pathsep.join( [ *parts, *filter( None, [ os.environ.get( "PYTHONPATH" ) ] ) ] )


# ── version ───────────────────────────────────────────────────────────────────

def cmd_show_version( p: Project ):
    """The version, and for a package that depends on loom the loom it requires and the one in use. Changes nothing."""
    print( p.version + ( f"   ( requires loom-ffi >= { required_loom( p ) }, built with { loom_version() } )" if p.depends_on_loom else "" ) )


def cmd_bump_version( p: Project, to = None ):
    """Write today's next version ( or `to` ) in pyproject.toml -- and, for a package that depends on loom, require at least
    the loom in use."""
    new = to or next_version( p.version )
    text = re.sub( r'(?m)^version = "[^"]*"', f'version = "{ new }"', p.toml_path.read_text(), count = 1 )
    if p.depends_on_loom:
        text = re.sub( r'"loom-ffi(\[[^\]]*\])?[^"]*"', lambda m: f'"loom-ffi{ m.group( 1 ) or "" }>={ loom_version() }"', text, count = 1 )
    p.toml_path.write_text( text )
    cmd_show_version( Project( p.root ) )                # what was written


# ── make_kernels ───────────────────────────────────────────────────────────────────

def cmd_make_kernels( p: Project ):
    """`python -m <catalogue>` once per framework, from scratch: a tag of an earlier build ( a level no longer built ) would go
    into the wheel otherwise."""
    if not p.catalogue:
        print( f"{ p.name }: no `catalogue` in [tool.loom.release], no kernels to build" )
        return
    shutil.rmtree( p.kernels_dir, ignore_errors = True )
    from importlib.util import find_spec
    for fw in p.frameworks or [ None ]:
        if fw and find_spec( fw ) is None:                 # a framework this build machine does not have ( cupy without a card )
            print( f"=== kernels for { fw }: skipped, not installed here", flush = True )
            continue
        print( f"=== kernels for { fw or 'the default framework' }", flush = True )
        env = { **os.environ, "PYTHONPATH": _source_path( p ), "LOOM_CATALOGUE_PRUNE": "0", **( { "LOOM_FRAMEWORK": fw } if fw else {} ) }
        run( sys.executable, "-m", p.catalogue, env = env, cwd = p.root )
    # once, with the choices as the last framework left them: each build measures them again ( a kernel the first one sent
    # to a poorer level may be sent back to a richer one by the next ), so a prune per build could empty a level a choice uses
    print( "=== pruning what no machine will ask for", flush = True )
    run( sys.executable, "-c",
         "import sys; from loom.compilation import catalogue as c; removed = c.prune( sys.argv[ 1 ], [ sys.argv[ 2 ] ] ); "
         "print( f'pruned { len( removed ) } kernel(s)' ); print( c.report( sys.argv[ 1 ] ) )",
         str( p.kernels_dir ), str( p.package_dir / "choices" ),
         env = { **os.environ, "PYTHONPATH": _source_path( p ) }, cwd = p.root )


# ── wheels ────────────────────────────────────────────────────────────────────

def default_platform_tag() -> str:
    """This interpreter's platform, manylinux_2_28 on Linux: the image the kernels are built in."""
    tag = sysconfig.get_platform().replace( "-", "_" ).replace( ".", "_" )
    return tag.replace( "linux_", "manylinux_2_28_", 1 ) if tag.startswith( "linux_" ) else tag


def _foreign_kernels( p: Project, platform_tag: str ):
    """The kernels of the other operating systems: the tree may hold several ( collected from several machines )."""
    keep = ".dylib" if platform_tag.startswith( "macosx" ) else ".dll" if platform_tag.startswith( "win" ) else ".so"
    return [ f for f in p.kernel_files() if f.suffix != keep ]


def build_wheel( p: Project, platform_tag = None, validate = True ) -> Path:
    """The wheel of this machine into `dist/`, validated. The wheels of other platforms already there are kept ( `--pull dist`
    brings them back from the other machines ), those of another version or another package are not ( they would be
    uploaded with the rest )."""
    dist = p.root / "dist"
    dist.mkdir( exist_ok = True )
    version = normalized( p.version )
    for old in dist.glob( "*.whl" ):
        if not old.name.startswith( f"{ p.wheel_name }-{ version }-" ):
            old.unlink()
    out = dist / ".build"
    shutil.rmtree( out, ignore_errors = True )
    out.mkdir( parents = True )

    platform_tag = platform_tag or default_platform_tag()
    with_kernels = bool( p.kernel_files() )
    aside, moved = dist / ".aside", []
    for f in _foreign_kernels( p, platform_tag ) if with_kernels else []:
        target = aside / f.relative_to( p.root )
        target.parent.mkdir( parents = True, exist_ok = True )
        f.rename( target )
        moved.append( ( target, f ) )
    try:
        run( sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "-w", out, p.root )
    finally:
        for target, f in moved:
            target.rename( f )
        shutil.rmtree( aside, ignore_errors = True )
    wheel = next( out.glob( f"{ p.wheel_name }-*.whl" ) )
    if with_kernels:
        run( sys.executable, "-m", "pip", "install", "--quiet", "wheel" )
        run( sys.executable, "-m", "wheel", "tags", "--remove", "--python-tag", "py3", "--abi-tag", "none",
             "--platform-tag", platform_tag, wheel )
        wheel = next( out.glob( f"{ p.wheel_name }-*.whl" ) )

    if validate:
        cmd_check_wheels( p, [ wheel ] )
    final = dist / wheel.name
    wheel.replace( final )
    shutil.rmtree( out, ignore_errors = True )
    print( f"wheel: { final }" )
    return final


def cmd_make_wheels( p: Project, kernels = True, platform_tag = None, validate = True ):
    """The wheel of this machine: its kernels ( `make_kernels` ), the wheel, its check."""
    if kernels and p.catalogue:
        cmd_make_kernels( p )
    build_wheel( p, platform_tag, validate )


# ── dispatch_wheels ───────────────────────────────────────────────────────────

def _errand():
    """errand's configuration and command line, with loom's build environments declared ( `envs.provide` ): also when errand
    is not installed in this interpreter's site-packages, where its entry point would have said them."""
    try:
        from errand import config as errand_config
        from errand.cli.main import main as errand_main
    except ImportError:
        raise SystemExit( "loom.release: this runs through errand, which is not importable here ( pip install errand-run )" )
    from . import envs
    if envs.provide not in errand_config.providers:
        errand_config.providers.append( envs.provide )
    return errand_config, errand_main


def cmd_errand( argv ):
    """errand itself, loom's build environments declared: `python -m loom.release errand setup --env loom-linux-x86_64`."""
    _, errand_main = _errand()
    code = errand_main( list( argv ) )
    if code:
        raise SystemExit( code )


def cmd_dispatch_wheels( p: Project, on: str, kernels = True ):
    """`make_wheels` in the build environment of each platform `on` names ( `all`: the project's `platforms` ), through
    errand, `dist/` brought back. The environments are loom's ( `envs.py` ), as the project's `errand-envs.py` may have
    changed them, at the places `loom-builders.py` says ( written with blanks when there is none ); a platform this machine
    has no way to build stops everything before anything runs. `all` leaves out what `loom-builders.py` skips."""
    errand_config, errand_main = _errand()
    from . import envs
    names = p.platforms if on == "all" else [ n for n in on.split( "," ) if n ]
    unknown = [ n for n in names if n not in p.platforms ]
    if not names or unknown:
        raise SystemExit( f"loom.release: dispatch_wheels takes `all` or platforms of `[tool.loom.release] platforms` "
                          f"( { ', '.join( p.platforms ) or 'none declared' } ); unknown: { ', '.join( unknown ) }" )
    if on == "all":
        # what `loom-builders.py` says not to build from here is left out, and said
        skipped = envs.skipped( p.root )
        for n in names:
            if n in skipped:
                print( f"loom.release: { n } skipped ( loom-builders.py{ ': ' + skipped[ n ] if skipped[ n ] else '' } )" )
        names = [ n for n in names if n not in skipped ]
        if not names:
            raise SystemExit( "loom.release: every platform is skipped in loom-builders.py" )
    # a platform with no way to be built from here is an `errand.Missing`: errand refuses it, saying what to write where
    # several environments are a grid for errand, asked for as such ( `--grid=,` )
    code = errand_main( [ "run", *( [ "--grid=," ] if len( names ) > 1 else [] ), "--env", ",".join( envs.env_name( n ) for n in names ), "--pull", "dist",
                          "python", "-m", "loom.release", "make_wheels", *( [] if kernels else [ "--no-kernels" ] ) ] )
    print( "\n".join( map( str, sorted( ( p.root / "dist" ).glob( f"{ p.wheel_name }-{ normalized( p.version ) }-*.whl" ) ) ) ) )
    if code:
        raise SystemExit( code )


# ── check_wheels ──────────────────────────────────────────────────────────────────

_PRECHECK = """
import {package}, loom, loom.compilation as c
print( {package}.__file__ )
print( loom.__file__ )
print( "\\n".join( map( str, c.include_roots() ) ) )
"""


def _venv_python( venv: Path ) -> Path:
    return venv / ( "Scripts/python.exe" if os.name == "nt" else "bin/python" )


def _loom_requirement( p: Project, scratch: Path ):
    """What the clean venv installs as loom: a wheel of the checkout in use ( built aside, not kept ), or the version
    installed here, from the index."""
    if p.is_loom:
        return []
    checkout = loom_checkout()
    if checkout is None:
        return [ f"loom-ffi=={ loom_version() }" ]
    run( sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "-w", scratch / "loom", checkout.root )
    return [ str( next( ( scratch / "loom" ).glob( "loom_ffi-*.whl" ) ) ) ]


def cmd_check_wheels( p: Project, wheels = None, keep = False ):
    """The wheel in a CLEAN venv: no `PYTHONPATH` to a checkout, a fresh cache. Its modules and C++ roots must come from the
    venv; `smoke` runs ( and may compile, with the wheel's headers ); every `checks` script runs with `LOOM_KERNELS=catalogue`,
    where a kernel the wheel does not hold is an error that names it."""
    wheels = [ Path( w ).resolve() for w in wheels ] if wheels else \
        sorted( ( p.root / "dist" ).glob( f"{ p.wheel_name }-{ normalized( p.version ) }-*.whl" ) )[ -1: ]
    if not wheels:
        raise SystemExit( f"loom.release: no { p.name } { p.version } wheel in dist/ to validate" )
    scratch = Path( tempfile.mkdtemp( prefix = f"{ p.wheel_name }-validate-" ) )
    venv, work = scratch / "venv", scratch / "run"
    work.mkdir( parents = True )
    try:
        run( sys.executable, "-m", "venv", venv )
        py = _venv_python( venv )
        run( py, "-m", "pip", "install", "--quiet", *map( str, wheels ), *_loom_requirement( p, scratch ), *p.check_with )
        env = { k: v for k, v in os.environ.items()
                if k not in ( "PYTHONPATH", "LOOM_BUILD_DIR", "LOOM_CACHE_DIR", "LOOM_KERNELS", "SDOT_BUILD_DIR", "SDOT_CACHE_DIR" ) }
        env[ "LOOM_CACHE_DIR" ] = str( scratch / "cache" )

        # what is imported, and the C++ roots, come from the venv ( `.resolve()`: /var is /private/var on macOS )
        r = run( py, "-c", _PRECHECK.format( package = p.package ), cwd = work, env = env, capture_output = True, text = True, check = False )
        if r.returncode:
            raise SystemExit( r.stdout + r.stderr + "\nloom.release: the installed wheel does not import" )
        inside = os.path.normcase( str( venv.resolve() ) )
        for line in filter( None, r.stdout.splitlines() ):
            if not os.path.normcase( str( Path( line ).resolve() ) ).startswith( inside ):
                raise SystemExit( f"loom.release: { line } is outside the venv ( imported from a checkout? )\n{ r.stdout }" )
        print( r.stdout, flush = True )

        if p.smoke:
            r = run( py, p.root / p.smoke, cwd = work, env = env, capture_output = True, text = True, check = False )
            print( r.stdout + r.stderr, flush = True )
            if r.returncode:
                raise SystemExit( f"loom.release: { p.smoke } failed" )

        strict = { **env, "LOOM_KERNELS": "catalogue" }
        for pattern in p.checks:
            for script in sorted( glob.glob( str( p.root / pattern ) ) ) or [ str( p.root / pattern ) ]:
                name = os.path.relpath( script, p.root )
                r = run( py, script, cwd = work, env = strict, capture_output = True, text = True, check = False )
                missing = re.search( r"No module named '(\w+)", r.stderr )
                if r.returncode and missing and missing.group( 1 ) in FRAMEWORK_MODULES:
                    print( f"  { name }: skipped ( { missing.group( 1 ) } is not in the venv )", flush = True )
                elif r.returncode:
                    raise SystemExit( r.stdout + r.stderr + f"\nloom.release: { name } needs a kernel the wheel does not hold, or failed" )
                else:
                    print( f"  { name }: OK, nothing compiled", flush = True )
        print( "\nVALIDATION OK", flush = True )
    finally:
        if keep:
            print( f"--keep: { scratch }" )
        else:
            shutil.rmtree( scratch, ignore_errors = True )


# ── publish_wheels ───────────────────────────────────────────────────────────────────

def _on_pypi( name: str, version: str ) -> bool:
    try:
        with urllib.request.urlopen( f"https://pypi.org/pypi/{ name }/{ version }/json", timeout = 30 ):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def required_loom( p: Project ):
    for d in p.dependencies:
        m = re.match( r"loom-ffi(\[[^\]]*\])?\s*>=\s*([\w.]+)", d )
        if m:
            return normalized( m.group( 2 ) )
    return None


def cmd_publish_wheels( p: Project, url = PYPI_UPLOAD ):
    """This version's wheels of `dist/`, uploaded ( `TWINE_PASSWORD` = the token ). To PyPI, a package that depends on loom is
    refused while the loom it requires is not there."""
    wheels = sorted( ( p.root / "dist" ).glob( f"{ p.wheel_name }-{ normalized( p.version ) }-*.whl" ) )
    if not wheels:
        raise SystemExit( f"loom.release: no { p.name } { p.version } wheel in dist/ -- `python -m loom.release make_wheels` first" )
    needed = required_loom( p )
    if needed and url == PYPI_UPLOAD and not _on_pypi( "loom-ffi", needed ):
        raise SystemExit( f"loom.release: { p.name } requires loom-ffi>={ needed }, which is not on PyPI -- release loom first" )
    print( "\n".join( map( str, wheels ) ) )
    run( sys.executable, "-m", "pip", "install", "--quiet", "twine" )
    run( sys.executable, "-m", "twine", "upload", "--skip-existing", "--repository-url", url, *wheels,
         env = { **os.environ, "TWINE_USERNAME": os.environ.get( "TWINE_USERNAME", "__token__" ) } )


# ── command line ──────────────────────────────────────────────────────────────

def main( argv = None ):
    ap = argparse.ArgumentParser( prog = "python -m loom.release", description = "Release a package built on loom ( see the module's docstring )" )
    sub = ap.add_subparsers( dest = "cmd", required = True )
    sub.add_parser( "show_version", help = "print the version ( and the loom it requires ), change nothing" )
    v = sub.add_parser( "bump_version", help = "write today's next version ( or --to X ) in pyproject.toml" )
    v.add_argument( "--to", default = None, metavar = "X" )
    sub.add_parser( "make_kernels", help = "the precompiled kernels of this machine, every framework" )
    w = sub.add_parser( "make_wheels", help = "the wheel of this machine into dist/: kernels, wheel, check" )
    w.add_argument( "--no-kernels", action = "store_true", help = "the kernels already in the tree" )
    w.add_argument( "--no-check", action = "store_true" )
    w.add_argument( "--platform-tag", default = None, help = "default: this interpreter's ( manylinux_2_28 on Linux )" )
    d = sub.add_parser( "dispatch_wheels", help = "make_wheels for `all` platforms of [tool.loom.release], or the named ones ( P1,P2 ), through errand" )
    d.add_argument( "on", metavar = "all|PLATFORMS" )
    d.add_argument( "--no-kernels", action = "store_true", help = "the kernels already in each copy of the tree" )
    e = sub.add_parser( "errand", help = "errand, with loom's build environments: `errand setup --env loom-linux-x86_64`" )
    e.add_argument( "args", nargs = argparse.REMAINDER )
    c = sub.add_parser( "check_wheels", help = "install in a clean venv and run the checks ( default: this version's wheel of dist/ )" )
    c.add_argument( "wheels", nargs = "*" )
    c.add_argument( "--keep", action = "store_true", help = "keep the venv, for a post-mortem" )
    u = sub.add_parser( "publish_wheels", help = "upload this version's wheels of dist/ ( TWINE_PASSWORD = the token )" )
    u.add_argument( "--url", default = PYPI_UPLOAD )
    a = ap.parse_args( argv )

    if a.cmd == "errand":
        return cmd_errand( a.args )
    p = Project.here()
    if a.cmd == "show_version":
        cmd_show_version( p )
    elif a.cmd == "bump_version":
        cmd_bump_version( p, a.to )
    elif a.cmd == "make_kernels":
        cmd_make_kernels( p )
    elif a.cmd == "make_wheels":
        cmd_make_wheels( p, kernels = not a.no_kernels, platform_tag = a.platform_tag, validate = not a.no_check )
    elif a.cmd == "dispatch_wheels":
        cmd_dispatch_wheels( p, a.on, kernels = not a.no_kernels )
    elif a.cmd == "check_wheels":
        cmd_check_wheels( p, a.wheels, a.keep )
    elif a.cmd == "publish_wheels":
        cmd_publish_wheels( p, a.url )
