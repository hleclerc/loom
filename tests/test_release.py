"""`loom.release`: versions, and the build environments it declares for errand ( `envs.py` ).

The environments are resolved, not run: which ones a project gets from the values the user gave, what goes in them, and
what is said when a platform has no way to be built. A real build is `python -m loom.release dispatch_wheels`.
"""
from pathlib import Path
import datetime
import os
import tempfile

from errand import test
from errand import config as C, local as LOC

from loom import release as R
from loom.release import envs as E


def _project( tmp: Path, platforms, deps = ( "numpy", "loom-ffi>=2026.1.1.1", "scipy" ), frameworks = ( "jax", "torch", "cupy" ) ):
    ( tmp / "src" / "mypkg" ).mkdir( parents = True )
    ( tmp / "pyproject.toml" ).write_text(
        '[project]\nname = "mypkg"\nversion = "2026.11.08.1"\n'
        f'dependencies = [{ ", ".join( repr( d ).replace( chr( 39 ), chr( 34 ) ) for d in deps ) }]\n'
        '[tool.loom.release]\n'
        f'frameworks = [{ ", ".join( chr( 34 ) + f + chr( 34 ) for f in frameworks ) }]\n'
        f'platforms = [{ ", ".join( chr( 34 ) + p + chr( 34 ) for p in platforms ) }]\n' )
    return R.Project( tmp )


def _load( root: Path, builders = None ):
    """errand's configuration of `root`, loom's provider declared, `builders` as its loom-builders.py ( None: none, loom
    writes its template )."""
    if builders is not None:
        ( root / "loom-builders.py" ).write_text( HEAD + builders )
    if E.provide not in C.providers:
        C.providers.append( E.provide )
    C.load( root )


HEAD = "from loom.release.builders import Builders, Here, Skip\nfrom errand import Ssh, Apptainer, Docker, Micromamba, Windows, WindowsBox, Vars\n"


def _restore( kept ):
    C.providers[ : ] = [ p for p in C.providers if p is not E.provide ]
    C.reset()
    LOC._cache.clear()
    LOC._cache.update( kept[ 0 ] )
    LOC._files[ : ], LOC._missing_root = kept[ 1 ], kept[ 2 ]


def _kept():
    return ( dict( LOC._cache ), list( LOC._files ), LOC._missing_root )


if test( "versions are today's next one, written as pip writes them" ):
    today = datetime.date( 2026, 11, 13 )
    assert R.next_version( "2026.11.08.1", today ) == "2026.11.13.1"
    assert R.next_version( "2026.11.13.2", today ) == "2026.11.13.3"
    assert R.normalized( "2026.11.08.1" ) == "2026.11.8.1"


if test( "bump_version writes the version and requires the loom in use" ):
    with tempfile.TemporaryDirectory() as tmp:
        p = _project( Path( tmp ), [] )
        R.cmd_bump_version( p, to = "2026.12.1.1" )
        text = ( Path( tmp ) / "pyproject.toml" ).read_text()
        assert 'version = "2026.12.1.1"' in text
        assert f'"loom-ffi>={ R.loom_version() }"' in text


class Loaded:
    """`with Loaded( platforms, builders ) as ( root, envs ):` a project in a temporary directory, errand loaded on it."""
    def __init__( self, platforms, builders = None, extras = "", files = None ):
        self.platforms, self.builders, self.extras, self.files = platforms, builders, extras, files or {}

    def __enter__( self ):
        self.kept = _kept()
        # these tests run in an errand environment, where loom writes nothing: they are the machine errand starts on
        self.in_env = os.environ.pop( "ERRAND_IN_ENV", None ) if not getattr( self, "inside", False ) else None
        self.tmp = tempfile.TemporaryDirectory()
        root = Path( self.tmp.name )
        _project( root, self.platforms )
        if self.extras:
            with open( root / "pyproject.toml", "a" ) as f:
                f.write( self.extras )
        for name, text in self.files.items():
            ( root / name ).parent.mkdir( parents = True, exist_ok = True )
            ( root / name ).write_text( text )
        _load( root, self.builders )
        return root, C.envs

    def __exit__( self, *exc ):
        if self.in_env is not None:
            os.environ[ "ERRAND_IN_ENV" ] = self.in_env
        _restore( self.kept )
        self.tmp.cleanup()


def _missing( env ):
    from errand import layers as L
    return L.missing_of( env.stack )


if test( "an Ssh gives the platform its environment, a blank host says what to fill in" ):
    with Loaded( [ "linux-x86_64", "windows-x86_64" ],
                 'builders = Builders( linux_x86_64 = [ Ssh( host = "build-box", root = "scratch" ) ],\n'
                 '                     windows_x86_64 = [ Ssh( host = "" ), Windows() ] )\n' ) as ( root, envs ):
        env = envs[ "loom-linux-x86_64" ]
        assert env.ssh.host == "build-box" and env.ssh.root == f"scratch/{ root.name }"   # the parent of the copies
        assert _missing( env ) is None
        recipe = ( root / ".loom" / "release" / "linux-x86_64.def" ).read_text()
        for wanted in ( '"scipy"', '"jax[cuda13]==0.10.2"', '"torch"', '"cupy-cuda13x"' ):
            assert wanted in recipe, wanted
        assert "loom-ffi>=" not in recipe                       # loom comes from the checkout, not the index
        win = _missing( envs[ "loom-windows-x86_64" ] )
        assert win is not None and "blank" in win.hint and "loom-builders.py" in win.hint, win
        assert C.default().name == "default"                     # provided: never the default


if test( "the layers the user gives are completed, theirs kept, loom's added" ):
    from errand import layers as L
    with Loaded( [ "linux-x86_64", "windows-x86_64", "macos-arm64" ],
                 'builders = Builders(\n'
                 '    linux_x86_64   = [ Ssh( host = "box" ), Apptainer( scratch = "/data/tmp", flags = [ "--bind", "/data" ] ),\n'
                 '                       Vars( { "LOOM_CACHE_DIR": "/data/cache" } ) ],\n'
                 '    windows_x86_64 = [ Ssh( host = "vm-host", root = "/big" ), Windows( cpus = 16, ram = "32G" ) ],\n'
                 '    macos_arm64    = [ Ssh( host = "mac" ), Micromamba( root_prefix = "/opt/mamba" ) ] )\n' ) as ( root, envs ):
        box = envs[ "loom-linux-x86_64" ].stack
        assert isinstance( box[ 1 ], L.Apptainer ) and box[ 1 ].scratch == "/data/tmp" and box[ 1 ].fakeroot
        assert box[ 1 ].image.endswith( "linux-x86_64.sif" ) and box[ 1 ].recipe.endswith( "linux-x86_64.def" )
        assert box[ 1 ].flags == [ "--nv", "--bind", "/data" ]
        assert box[ -1 ].values == { "LOOM_CACHE_DIR": "/data/cache" }          # after loom's: has the last word
        win = envs[ "loom-windows-x86_64" ].stack
        assert win[ 0 ].root == f"/big/{ root.name }"
        assert ( win[ 1 ].cpus, win[ 1 ].ram, win[ 1 ].python ) == ( 16, "32G", E.WINDOWS_PYTHON )
        assert win[ 1 ].provision[ 1 ].endswith( "windows-tools.ps1" ) and "scipy" in win[ 1 ].provision[ 0 ]
        mac = envs[ "loom-macos-arm64" ].stack
        assert mac[ 1 ].root_prefix == "/opt/mamba" and mac[ 1 ].name == "loom-mypkg-macos-arm64" and "clangxx" in mac[ 1 ].packages
        # a root that already names the project is the copy itself, not its parent
        project = R.Project( root )
        assert E._ssh( project, L.Ssh( host = "mac", root = f"/x/{ project.root.name }/" ) ).root == f"/x/{ project.root.name }"


if test( "a Windows machine reached by ssh is a WindowsBox, completed as the VM is" ):
    from errand import layers as L
    with Loaded( [ "windows-x86_64" ],
                 'builders = Builders( windows_x86_64 = [ Ssh( host = "winbox", root = "D:/x" ),\n'
                 '                                         WindowsBox( provision = [ "Write-Output mine" ] ) ] )\n' ) as ( root, envs ):
        env = envs[ "loom-windows-x86_64" ]
        assert _missing( env ) is None
        ssh, box, v = env.stack[ : 3 ]
        assert ssh.host == "winbox" and ssh.root == "D:/x"                    # the box's: kept as it is
        assert box.is_box and box.via is ssh and box.python == E.WINDOWS_PYTHON
        assert "C:\\errand\\venv\\Scripts\\python.exe" in box.provision[ 0 ] and "scipy" in box.provision[ 0 ]
        assert box.provision[ 1 ].endswith( "windows-tools.ps1" ) and box.provision[ -1 ] == "Write-Output mine"
        assert v.values[ "LOOM_CACHE_DIR" ].startswith( "C:\\errand" ) and "llvm-mingw" in v.values[ "PATH" ]
        assert env.ssh is None and env.vm is box                              # errand drives it as a Windows guest


if test( "loom-builders.py is written when absent, with what this machine does and blanks for the rest" ):
    from loom.release import builders as B
    with Loaded( [ "macos-arm64", "linux-x86_64", "windows-x86_64" ] ) as ( root, envs ):
        text = ( root / "loom-builders.py" ).read_text()
        said = B.read( root )
        for p in E.PLATFORMS:
            if E.can_here( p ):
                assert isinstance( said.of( p ), B.Here ) and _missing( envs[ E.env_name( p ) ] ) is None, p
            else:
                assert said.of( p )[ 0 ].host == "" and "blank" in _missing( envs[ E.env_name( p ) ] ).hint, p
        ( root / "loom-builders.py" ).write_text( "# mine\n" + text )
        _load( root )
        assert ( root / "loom-builders.py" ).read_text().startswith( "# mine" )        # never rewritten
    # a git checkout ignores it; inside an environment, nothing is written
    with tempfile.TemporaryDirectory() as tmp:
        root = Path( tmp )
        ( root / ".git" ).mkdir()
        ( root / ".gitignore" ).write_text( "build/" )
        assert B.write_template( root, "macos-arm64", {} ) == root / "loom-builders.py"
        assert ( root / ".gitignore" ).read_text().splitlines()[ -1 ] == "/loom-builders.py"
        assert B.write_template( root, "macos-arm64", {} ) is None
    was = os.environ.get( "ERRAND_IN_ENV" )
    os.environ[ "ERRAND_IN_ENV" ] = "loom-linux-x86_64"
    try:
        inside = Loaded( [ "linux-x86_64" ] )
        inside.inside = True
        with inside as ( root, envs ):
            assert not ( root / "loom-builders.py" ).exists()
    finally:
        os.environ.pop( "ERRAND_IN_ENV" )
        if was is not None:
            os.environ[ "ERRAND_IN_ENV" ] = was


if test( "Skip leaves a platform out, a file that does not read says why, a wrong value is refused" ):
    from loom.release import builders as B
    with Loaded( [ "windows-x86_64", "linux-x86_64" ],
                 'builders = Builders( windows_x86_64 = Skip( "no KVM" ), linux_x86_64 = [ Ssh( host = "b" ) ] )\n' ) as ( root, envs ):
        assert "skipped" in _missing( envs[ "loom-windows-x86_64" ] ).what
        assert E.skipped( root ) == { "windows-x86_64": "no KVM" }
    with Loaded( [ "linux-x86_64" ], "builders = 3\n" ) as ( root, envs ):
        assert "must define `builders = Builders( ... )`" in _missing( envs[ "loom-linux-x86_64" ] ).what
    try:
        B.Builders( linux_x86_64 = "box" )
        assert False, "refused"
    except TypeError as err:
        assert "Here(), a list of errand layers" in str( err )


if test( "what a project needs on top goes in its recipes, per platform" ):
    extras = ( '[tool.loom.release.linux-x86_64]\npip = ["pyfftw"]\nsetup = "scripts/extra.sh"\n'
               '[tool.loom.release.macos-arm64]\nconda = ["fftw"]\n'
               '[tool.loom.release.windows-x86_64]\nsetup = "scripts/extra.ps1"\n' )
    with Loaded( [ "linux-x86_64", "macos-arm64", "windows-x86_64" ],
                 'builders = Builders( linux_x86_64 = [ Ssh( host = "b" ), Docker() ], macos_arm64 = [ Ssh( host = "m" ) ],\n'
                 '                     windows_x86_64 = [ Ssh( host = "v" ), Windows() ] )\n',
                 extras = extras, files = { "scripts/extra.sh": "dnf -y install fftw-devel\n", "scripts/extra.ps1": "echo hi\n" } ) as ( root, envs ):
        docker = envs[ "loom-linux-x86_64" ].stack[ 1 ]
        assert docker.flags[ :2 ] == [ "--platform", "linux/amd64" ] and docker.recipe.endswith( ".Dockerfile" )
        text = ( root / docker.recipe ).read_text()
        assert '"pyfftw"' in text and "dnf -y install fftw-devel" in text and "{pip}" not in text
        assert "fftw" in envs[ "loom-macos-arm64" ].stack[ 1 ].packages
        assert "scripts/extra.ps1" in envs[ "loom-windows-x86_64" ].stack[ 1 ].provision
    p = _project( Path( tempfile.mkdtemp() ), [] )
    p.extras = { "linux-x86_64": { "setup": "x.sh" } }
    ( p.root / "x.sh" ).write_text( "echo extra\n" )
    assert "    echo extra\n\n%environment" in ( p.root / E.linux_recipe( p ) ).read_text()
    p.extras = { "linux-x86_64": { "apt": [] } }
    try:
        E.packages( p, "linux-x86_64" )
        assert False, "refused"
    except ValueError as err:
        assert "unknown apt" in str( err )


if test( "a project without platforms gets nothing, and its own file has the last word" ):
    kept = _kept()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path( tmp )
            ( root / "pyproject.toml" ).write_text( '[project]\nname = "other"\nversion = "1"\n' )
            _load( root )
            assert not [ n for n in C.envs if n.startswith( "loom-" ) ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path( tmp )
            _project( root, [ "linux-x86_64" ] )
            ( root / "errand-envs.py" ).write_text(
                "import errand\nerrand.envs[ 'loom-linux-x86_64' ] = errand.Env( [ errand.Vars( { 'MINE': '1' } ) ] )\n" )
            _load( root, 'builders = Builders( linux_x86_64 = [ Ssh( host = "build-box" ) ] )\n' )
            assert C.envs[ "loom-linux-x86_64" ].stack[ 0 ].values == { "MINE": "1" }
    finally:
        _restore( kept )


if test( "on a Mac, the Mac platform is built here" ):
    if E.this_platform() == "macos-arm64":
        kept = _kept()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path( tmp )
                _project( root, [ "macos-arm64" ] )
                _load( root, "builders = Builders()\n" )               # None: what loom finds on its own
                env = C.envs[ "loom-macos-arm64" ]
                assert env.ssh is None
                mamba = env.stack[ 0 ]
                assert mamba.packages == [ "clangxx" ] and "jax==0.10.2" in mamba.pip and "cupy-cuda13x" not in mamba.pip
        finally:
            _restore( kept )
