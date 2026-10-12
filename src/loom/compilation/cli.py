"""Housekeeping of the kernel cache:  `python -m loom.compilation.cli prune [--older-than DAYS] [--dry-run]`.

Producing a CATALOGUE is not done from here: it is a Python script of the package that ships it, which loops on the
hardware (`loom.hardware`) and collects what it compiles (`loom.compilation.catalogue.collect`).
"""
import argparse
import sys


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
    p = argparse.ArgumentParser( prog = "python -m loom.compilation.cli", description = __doc__.splitlines()[ 0 ] )
    sub = p.add_subparsers( dest = "cmd", required = True )
    g = sub.add_parser( "prune", help = "forget kernels: erase their directory" )
    g.add_argument( "--older-than", type = float, metavar = "DAYS",
                    help = "only erase the kernels unused for this many days "
                           "( without the option: all )" )
    g.add_argument( "--dry-run", action = "store_true", help = "say what would be erased" )
    g.set_defaults( func = prune )
    a = p.parse_args( sys.argv[ 1: ] if argv is None else argv )
    a.func( a )


if __name__ == "__main__":
    main()
