"""`loom-kernels`: build the CATALOGUE of precompiled kernels, from any build.

This is the interface through which an outside project -- and the build tool of its choice,
cmake, bazel, xmake, meson, a Makefile -- has its kernels produced AHEAD OF TIME rather than at
run time. Loom's model (targets invented while the program runs) fits none of those tools; what
does fit is an ARTIFACT they know how to build and install. This command is that artifact.

    # 1. the RECORD, on a development machine: run whatever triggers the compilations, and
    #    keep the generated sources ( for CUDA, a GPU is needed ).
    loom-kernels record --out catalogue_record -- python -m my_package.catalogue

    # 2. the COMPILATION of the record, anywhere, once per platform and per variant.
    #    `--import` names the modules that register their C++ roots ( `register_include_root` ):
    #    loom does not know its users by name, they have to be told to it.
    loom-kernels compile --record catalogue_record --out my_package/catalogue \\
                         --import my_package --variant x86-64-v3

    # 3. the wheel ships the output directory, registered at import time by
    #    `loom.compilation.catalogue.register_catalogue( ... )`.

The record knows nothing about WHAT is run: the command is given after `--`, as many times as
wanted ( one `--` per command ). This is what makes the thing usable by someone else -- the
previous version hard-coded the test suites of `sdot`.
"""
from pathlib import Path
import argparse
import importlib
import os
import subprocess
import sys
import tempfile

from .. import env


def _split_commands( rest ):
    """`[ "--", "a", "b", "--", "c" ]` -> `[ [ "a", "b" ], [ "c" ] ]`."""
    commands, current = [], None
    for word in rest:
        if word == "--":
            if current:
                commands.append( current )
            current = []
        elif current is not None:
            current.append( word )
    if current:
        commands.append( current )
    return commands


def record( args, commands ):
    """Run the given commands with the record hooked up, and keep what they caused to compile."""
    if not commands:
        raise SystemExit( "loom-kernels record: say WHAT to run, after `--`\n"
                          "  loom-kernels record --out catalogue_record -- python -m my_package.catalogue" )

    out = Path( args.out ).resolve()
    # the record of one kind of device replaces the previous one of the same kind; the other kinds
    # and the generated headers ( merged, written if changed ) stay
    if ( out / args.device ).exists():
        import shutil
        shutil.rmtree( out / args.device )
    out.mkdir( parents = True, exist_ok = True )

    with tempfile.TemporaryDirectory( prefix = "loom-catalogue-record-" ) as build:
        child_env = dict( os.environ )
        child_env.update( LOOM_CATALOGUE_RECORD = str( out ), LOOM_BUILD_DIR = build,
                              LOOM_KERNELS = "compile" )
        for cmd in commands:
            print( "$", " ".join( cmd ), flush = True )
            if subprocess.run( cmd, env = child_env ).returncode:
                raise SystemExit( "the record failed" )

    nb = len( list( ( out / args.device ).glob( "*.c*" ) ) ) if ( out / args.device ).is_dir() else 0
    print( f"record: { nb } { args.device } kernel(s) in { out }" )


def compile_( args ):
    """Compile a record for a platform and a variant, without running anything of the project."""
    with tempfile.TemporaryDirectory( prefix = "loom-catalogue-build-" ) as build:
        # the variant / architectures reach the compiler through the environment, and the build has
        # its own directory: nothing from the machine must get into the library
        env.set_var( "BUILD_DIR", build )
        if args.device == "cpu":
            env.set_var( "CPU_VARIANT", args.variant )
            tag = f"cpu-{ args.variant }"
        else:
            env.set_var( "CUDA_ARCH", args.arch )
            tag = "cuda"

        # the modules that register their C++ roots: without them, the record's `#include`s do not
        # resolve. loom does not guess them -- `register_include_root` is an explicit registration.
        for name in args.import_:
            importlib.import_module( name )

        from ..devices.Device import Device
        from . import catalogue
        nb = catalogue.build( args.record, args.out, Device.factory( args.device ), tag )
    print( f"catalogue `{ tag }`: { nb } kernel(s) in { Path( args.out ) / tag }" )


def prune( args ):
    """Forget kernels: erase their directory, then clean up the shared graph.

    It is the partition that makes the operation trivial ( see `build.py` ): what belongs to one
    kernel lives in its directory and goes away with it, with no graph mutation nor lock. The
    shared graph then cleans itself up, by removing the edges whose output no longer exists.
    """
    import shutil
    import time

    from .build import Manifest, kernels_root
    from . import build_dir

    root = kernels_root()
    cutoff = time.time() - args.older_than * 86400 if args.older_than is not None else None
    erased, nb_bytes = 0, 0
    if root.is_dir():
        for d in sorted( root.iterdir() ):
            if not d.is_dir():
                continue
            if cutoff is not None and d.stat().st_mtime >= cutoff:
                continue
            size = sum( f.stat().st_size for f in d.rglob( "*" ) if f.is_file() )
            if args.dry_run:
                print( f"  to erase: { d.name } ( { size // 1024 } kB )" )
            else:
                shutil.rmtree( d )
            erased += 1
            nb_bytes += size

    verb = "to erase" if args.dry_run else "erased"
    print( f"{ verb }: { erased } kernel(s), { nb_bytes // ( 1024 * 1024 ) } MB" )

    if not args.dry_run:
        m = Manifest( build_dir() )
        removed = m.prune()
        if removed:
            m.save()
            m.write_ninja()
        print( f"shared graph: { removed } orphan edge(s) removed, { len( m.edges ) } remaining" )


def main( argv = None ):
    argv = list( sys.argv[ 1: ] if argv is None else argv )
    # `--` separates our options from the command to run: argparse cannot do it by itself
    cut = argv.index( "--" ) if "--" in argv else len( argv )
    mine, rest = argv[ : cut ], argv[ cut : ]

    p = argparse.ArgumentParser( prog = "loom-kernels", description = __doc__.splitlines()[ 0 ] )
    sub = p.add_subparsers( dest = "cmd", required = True )

    r = sub.add_parser( "record", help = "run what compiles, and keep the generated sources" )
    r.add_argument( "--out", default = "catalogue_record" )
    r.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    r.set_defaults( func = lambda a: record( a, _split_commands( rest ) ) )

    c = sub.add_parser( "compile", help = "compile a record for a platform" )
    c.add_argument( "--record", default = "catalogue_record" )
    c.add_argument( "--out", required = True )
    c.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    c.add_argument( "--variant", default = "x86-64-v3", help = "CPU level: x86-64-v2 / v3 / v4, armv8-a" )
    c.add_argument( "--arch", default = "sm_70,sm_75,sm_80,sm_86,sm_89,sm_90",
                    help = "CUDA architectures, separated by commas" )
    c.add_argument( "--import", dest = "import_", action = "append", default = [],
                    metavar = "MODULE", help = "module to import before compiling ( it registers "
                                               "its C++ root ) ; repeatable" )
    c.set_defaults( func = compile_ )

    g = sub.add_parser( "prune", help = "forget kernels: erase their directory" )
    g.add_argument( "--older-than", type = float, metavar = "DAYS",
                    help = "only erase the kernels unused for this many days "
                           "( without the option: all )" )
    g.add_argument( "--dry-run", action = "store_true", help = "say what would be erased" )
    g.set_defaults( func = prune )

    a = p.parse_args( mine )
    a.func( a )


if __name__ == "__main__":
    main()
