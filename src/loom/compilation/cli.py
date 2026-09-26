"""`loom-kernels` : fabriquer le CATALOGUE des noyaux précompilés, depuis n'importe quel build.

C'est l'interface par laquelle un projet extérieur -- et l'outil de construction de son choix,
cmake, bazel, xmake, meson, un Makefile -- fait produire ses noyaux À L'AVANCE plutôt qu'à
l'exécution. Le modèle de loom (des cibles inventées pendant que le programme tourne) ne rentre
dans aucun de ces outils ; ce qui rentre, c'est un ARTEFACT qu'ils savent fabriquer et installer.
Cette commande est cet artefact.

    # 1. le RELEVÉ, sur une machine de développement : on lance ce qui provoque les
    #    compilations, et on garde les sources engendrées ( pour CUDA, il faut un GPU ).
    loom-kernels record --out catalogue_record -- python -m mon_paquet.catalogue

    # 2. la COMPILATION du relevé, n'importe où, une fois par plateforme et par variante.
    #    `--import` nomme les modules qui enregistrent leurs racines C++ ( `register_include_root` ) :
    #    loom ne connaît pas ses usagers par leur nom, il faut les lui dire.
    loom-kernels compile --record catalogue_record --out mon_paquet/catalogue \\
                         --import mon_paquet --variant x86-64-v3

    # 3. le wheel embarque le répertoire de sortie, enregistré à l'import par
    #    `loom.compilation.catalogue.register_catalogue( ... )`.

Le relevé ne sait rien de CE qu'on lance : la commande est donnée après `--`, autant de fois
qu'on veut ( `--` une fois par commande ). C'est ce qui rend la chose utilisable par quelqu'un
d'autre -- la version d'avant nommait en dur les suites de tests de `sdot`.
"""
from pathlib import Path
import argparse
import importlib
import os
import subprocess
import sys
import tempfile


def _decoupe_commandes( restes ):
    """`[ "--", "a", "b", "--", "c" ]` -> `[ [ "a", "b" ], [ "c" ] ]`."""
    commandes, courante = [], None
    for mot in restes:
        if mot == "--":
            if courante:
                commandes.append( courante )
            courante = []
        elif courante is not None:
            courante.append( mot )
    if courante:
        commandes.append( courante )
    return commandes


def record( args, commandes ):
    """Lance les commandes données avec le relevé branché, et garde ce qu'elles ont fait compiler."""
    if not commandes:
        raise SystemExit( "loom-kernels record : dire QUOI lancer, après `--`\n"
                          "  loom-kernels record --out catalogue_record -- python -m mon_paquet.catalogue" )

    out = Path( args.out ).resolve()
    # le relevé d'un genre de device remplace le précédent du même genre ; les autres genres et
    # les en-têtes engendrés ( fusionnés, écrits si changés ) restent
    if ( out / args.device ).exists():
        import shutil
        shutil.rmtree( out / args.device )
    out.mkdir( parents = True, exist_ok = True )

    with tempfile.TemporaryDirectory( prefix = "loom-catalogue-record-" ) as build:
        env = dict( os.environ )
        env.update( SDOT_CATALOGUE_RECORD = str( out ), SDOT_BUILD_DIR = build,
                    SDOT_KERNELS = "atelier" )
        for cmd in commandes:
            print( "$", " ".join( cmd ), flush = True )
            if subprocess.run( cmd, env = env ).returncode:
                raise SystemExit( "le relevé a échoué" )

    nb = len( list( ( out / args.device ).glob( "*.c*" ) ) ) if ( out / args.device ).is_dir() else 0
    print( f"relevé : { nb } noyau(x) { args.device } dans { out }" )


def compile_( args ):
    """Compile un relevé pour une plateforme et une variante, sans rien exécuter du projet."""
    with tempfile.TemporaryDirectory( prefix = "loom-catalogue-build-" ) as build:
        # la variante / les architectures vont au compilateur par l'environnement, et le build a
        # son propre répertoire : rien de la machine ne doit entrer dans la bibliothèque
        os.environ[ "SDOT_BUILD_DIR" ] = build
        if args.device == "cpu":
            os.environ[ "SDOT_CPU_VARIANT" ] = args.variant
            tag = f"cpu-{ args.variant }"
        else:
            os.environ[ "SDOT_CUDA_ARCH" ] = args.arch
            tag = "cuda"

        # les modules qui enregistrent leurs racines C++ : sans eux, les `#include` du relevé ne
        # résolvent pas. loom ne les devine pas -- `register_include_root` est une inscription.
        for nom in args.import_:
            importlib.import_module( nom )

        from ..devices.Device import Device
        from . import catalogue
        nb = catalogue.build( args.record, args.out, Device.factory( args.device ), tag )
    print( f"catalogue `{ tag }` : { nb } noyau(x) dans { Path( args.out ) / tag }" )


def prune( args ):
    """Oublier des noyaux : effacer leur repertoire, puis nettoyer le graphe commun.

    C'est la partition qui rend l'operation triviale ( voir `build.py` ) : ce qui est propre a un
    noyau vit dans son repertoire et part avec lui, sans mutation de graphe ni verrou. Le graphe
    commun se nettoie ensuite tout seul, en retirant les aretes dont la sortie n'existe plus.
    """
    import shutil
    import time

    from .build import Manifest, kernels_root
    from . import build_dir

    racine = kernels_root()
    limite = time.time() - args.older_than * 86400 if args.older_than is not None else None
    efface, octets = 0, 0
    if racine.is_dir():
        for d in sorted( racine.iterdir() ):
            if not d.is_dir():
                continue
            if limite is not None and d.stat().st_mtime >= limite:
                continue
            taille = sum( f.stat().st_size for f in d.rglob( "*" ) if f.is_file() )
            if args.dry_run:
                print( f"  a effacer : { d.name } ( { taille // 1024 } ko )" )
            else:
                shutil.rmtree( d )
            efface += 1
            octets += taille

    verbe = "a effacer" if args.dry_run else "efface"
    print( f"{ verbe } : { efface } noyau(x), { octets // ( 1024 * 1024 ) } Mo" )

    if not args.dry_run:
        m = Manifest( build_dir() )
        retirees = m.prune()
        if retirees:
            m.save()
            m.write_ninja()
        print( f"graphe commun : { retirees } arete(s) orpheline(s) retiree(s), { len( m.edges ) } restantes" )


def main( argv = None ):
    argv = list( sys.argv[ 1: ] if argv is None else argv )
    # `--` sépare nos options de la commande à lancer : argparse ne sait pas le faire lui-même
    coupe = argv.index( "--" ) if "--" in argv else len( argv )
    miens, restes = argv[ : coupe ], argv[ coupe : ]

    p = argparse.ArgumentParser( prog = "loom-kernels", description = __doc__.splitlines()[ 0 ] )
    sub = p.add_subparsers( dest = "cmd", required = True )

    r = sub.add_parser( "record", help = "lancer ce qui compile, et garder les sources engendrées" )
    r.add_argument( "--out", default = "catalogue_record" )
    r.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    r.set_defaults( func = lambda a: record( a, _decoupe_commandes( restes ) ) )

    c = sub.add_parser( "compile", help = "compiler un relevé pour une plateforme" )
    c.add_argument( "--record", default = "catalogue_record" )
    c.add_argument( "--out", required = True )
    c.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    c.add_argument( "--variant", default = "x86-64-v3", help = "niveau CPU : x86-64-v2 / v3 / v4, armv8-a" )
    c.add_argument( "--arch", default = "sm_70,sm_75,sm_80,sm_86,sm_89,sm_90",
                    help = "architectures CUDA, séparées par des virgules" )
    c.add_argument( "--import", dest = "import_", action = "append", default = [],
                    metavar = "MODULE", help = "module à importer avant de compiler ( il enregistre "
                                               "sa racine C++ ) ; répétable" )
    c.set_defaults( func = compile_ )

    g = sub.add_parser( "prune", help = "oublier des noyaux : effacer leur repertoire" )
    g.add_argument( "--older-than", type = float, metavar = "JOURS",
                    help = "n'effacer que les noyaux inutilises depuis ce nombre de jours "
                           "( sans l'option : tous )" )
    g.add_argument( "--dry-run", action = "store_true", help = "dire ce qui serait efface" )
    g.set_defaults( func = prune )

    a = p.parse_args( miens )
    a.func( a )


if __name__ == "__main__":
    main()
