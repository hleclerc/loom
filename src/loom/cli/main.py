#!/usr/bin/env python3
"""`./run` : ce qui FABRIQUE la machine sur laquelle le travail tourne.

Installer les paquets en editable, diagnostiquer la chaîne de compilation, bâtir les images
Apptainer, créer les environnements micromamba. C'est tout.

**Ce qui lance le travail est parti chez `errand`** -- un paquet à part, qui ne sait rien de
loom et qu'un autre projet peut donc utiliser :

    ./run test Cell::batch          devient    errand test_Cell::batch
    ./run bench --nb-diracs=5000               errand -k bench --nb-diracs 5000
    ./run experiment lung                      errand -k experiment lung
    ./run test --env lmo-cuda-jax              errand --env lmo-cuda-jax
    ./run test --driver torch                  errand -t driver=torch
                                             errand --tui

Ce qui était déclaré ici l'est dans `errandfile.py`, à la racine : les environnements, les
chemins des sources, et le provider qui compile les tests C++ de loom.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path

from . import layers

ROOT = Path(__file__).resolve().parents[4]

BOLD = "\033[1m"; DIM = "\033[2m"; GREEN = "\033[32m"; CYAN = "\033[36m"; RED = "\033[31m"; RESET = "\033[0m"
def _hdr(s): return f"{BOLD}{CYAN}{s}{RESET}"
def _ok(s): return f"{GREEN}{s}{RESET}"
def _err(s): return f"{RED}{s}{RESET}"
def _dim(s): return f"{DIM}{s}{RESET}"

def run(cmd, *, env=None, cwd=None):
    merged = os.environ.copy()
    if env: merged.update(env)
    print(_dim(f"  $ {' '.join(cmd)}"))
    return subprocess.run(cmd, env=merged, cwd=cwd or ROOT).returncode

def python(): return sys.executable


def _env_banner(seq):
    """Grey status line: which machine this is about to run on and with
    which driver -- rsync push/pull get their own line from `layers.Remote
    .run` itself, since only it knows whether/what it actually synced."""
    remote = seq[0] if seq and isinstance(seq[0], layers.Remote) else None
    driver_layer = next((l for l in seq if isinstance(l, layers.Driver)), None)
    machine = f"{remote.host} ({remote.remote_dir})" if remote else "local"
    parts = [f"machine={machine}"]
    if driver_layer:
        parts.append(f"driver={driver_layer.name}")
    # flush explicitly: stdout is block-buffered (not line-buffered) once
    # piped/redirected, and the point of this banner is to show up BEFORE
    # the actual work starts -- unflushed, it sits behind everything else
    # (including a remote ssh subprocess's own unbuffered output) until
    # this process exits, printing dead last instead of first.
    print(_dim(f"  → {'  '.join(parts)}"), flush=True)


def run_in_env(seq, argv, env_vars=None, pull=None):
    """Run `argv` through the layer sequence `seq`: a local subprocess
    (relative paths in `argv`/PYTHONPATH resolve against ROOT), or shipped
    over ssh when `seq` starts with a Remote layer (same relative paths
    resolve against the remote checkout, via `cd`) -- `pull` paths (relative
    to root), if given, are rsynced back afterwards; meaningless and ignored
    when running locally."""
    _env_banner(seq)
    cmd = layers.Command(list(argv), dict(env_vars or {}))
    result = layers.resolve(seq, cmd, layers.Context(root=ROOT), pull=pull)
    if isinstance(result, int):
        return result
    return run(result.argv, env=result.env)


# ── per-entry output directories ────────────────────────────────────────────────
#
# tmp/{kind}/{file}__{name}/[param_hash]/{env}/{date}/ -- one leaf directory
# per (case, resolved param set, env, date). `param_hash` is only
def cmd_build_sif(args):
    """Build Apptainer .sif images out of .envs.py's Apptainer layers.

    Iterates every env (or just --env NAME); envs without an Apptainer layer
    are skipped. Builds locally, or on the env's Remote (rsync -> ssh ->
    apptainer build) when it has one.
    """
    from . import envs

    all_envs = envs.load_envs()
    if args.env_name and args.env_name not in all_envs:
        print(_err(f"Unknown env: {args.env_name}"))
        print(_dim(f"  Available: {', '.join(all_envs) or '(none)'}"))
        return 1
    targets = [all_envs[args.env_name]] if args.env_name else list(all_envs.values())

    with_image = [(e, envs.apptainer_of(e)) for e in targets]
    with_image = [(e, a) for e, a in with_image if a is not None]
    if not with_image:
        scope = f"matching '{args.env_name}'" if args.env_name else "in .envs.py"
        print(_err(f"No Apptainer-based env found {scope}"))
        return 1

    rc = 0
    for env_cfg, apptainer in with_image:
        def_file = envs.def_for_image(apptainer.image)
        if not (ROOT / def_file).exists():
            print(_err(f"  {env_cfg.name}: .def file not found: {def_file}"))
            rc = 1
            continue

        remote = envs.remote_of(env_cfg)
        if remote:
            print(_hdr(f"\nbuild-sif: {env_cfg.name} → {apptainer.image} (on {remote.host})"))
            sd = args.scratch_dir or remote.apptainer_scratch or ""
            if sd:
                print(_dim(f"  scratch: {sd}"))
            commands = envs.remote_build_sif_commands(
                remote, apptainer,
                force=args.force, fakeroot=args.fakeroot,
                scratch_dir=args.scratch_dir,
            )
            for cmd in commands:
                if run(cmd) != 0:
                    rc = 1
                    break
        else:
            print(_hdr(f"\nbuild-sif: {env_cfg.name} → {apptainer.image}"))
            cmd = envs.build_sif_command(apptainer, force=args.force, fakeroot=args.fakeroot)
            if args.scratch_dir:
                cmd_env = {"APPTAINER_TMPDIR": args.scratch_dir, "APPTAINER_CACHEDIR": args.scratch_dir}
                print(_dim(f"  APPTAINER_TMPDIR={args.scratch_dir}"))
                if run(cmd, env=cmd_env) != 0:
                    rc = 1
            else:
                if run(cmd) != 0:
                    rc = 1

    return rc


def cmd_env(args):
    if getattr(args, "env_action", "list") == "create":
        return _cmd_env_create(args)

    from . import envs
    all_envs = envs.load_envs()

    print(_hdr("\nEnvironments (.envs.py):"))
    if not all_envs:
        print(_dim("  (none configured)"))
        return 0

    default_name = "default" if "default" in all_envs else next(iter(all_envs), None)

    for name, e in all_envs.items():
        marker = " ← default" if name == default_name else ""
        parts = []
        for layer in e.seq:
            if isinstance(layer, layers.Remote):
                parts.append(f"remote={layer.host}")
            elif isinstance(layer, layers.Micromamba):
                parts.append(f"micromamba={layer.name}")
            elif isinstance(layer, layers.Venv):
                parts.append(f"venv={layer.python}")
            elif isinstance(layer, layers.Apptainer):
                parts.append(f"apptainer={layer.image}")
        print(f"  {name:18s}  driver={e.driver or '?':6s}  {' '.join(parts)}{marker}")
        remote = envs.remote_of(e)
        if remote and remote.apptainer_scratch:
            print(f"  {'':18s}  scratch: {remote.apptainer_scratch}")

    drivers = sorted({e.driver for e in all_envs.values() if e.driver})
    print(_dim(f"\n  Select with: --env <name>  (or --driver <{', '.join(drivers)}>)"))
    return 0


def _cmd_env_create(args):
    """`./run env create` : fabrique ce que l'env DÉCLARE et qui n'est pas encore là.

    Une couche sait se sonder (`probe_shell`) et se créer (`create_shell`) ou ne sait pas :
    une qui ne sait pas est simplement ignorée, sans branche par type ici. `Apptainer` est
    volontairement de celles-là -- son image se construit avec `./run build-sif`, qui a ses
    propres options (--fakeroot, scratch) et n'a rien à faire dans une boucle générique.
    """
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    if env_cfg is None:
        print(_err("env create: aucun environnement de ce nom (voir `./run env`)"))
        return 1

    remote = envs.remote_of(env_cfg)
    ctx = layers.Context(root=Path(remote.remote_dir) if remote else ROOT, remote=bool(remote))
    todo = [l for l in env_cfg.seq if hasattr(l, "create_shell")]
    if not todo:
        print(_dim(f"  '{env_cfg.name}' : rien à créer"))
        return 0

    rc = 0
    for layer in todo:
        what = layer.describe()
        if layers.run_shell(layer.probe_shell(ctx), remote, quiet=True) == 0:
            print(_dim(f"  {what} : déjà présent"))
            continue
        print(_hdr(f"\ncréation: {what}"))
        if layers.run_shell(layer.create_shell(ctx), remote) != 0:
            print(_err(f"  échec: {what}"))
            rc = 1
    if rc == 0:
        print(_dim("\n  puis: ./run install"))
    return rc


def cmd_install(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    # le driver résolu, visible du processus fils : c'est ce que `loom.testing.driver_is` lit pour
    # qu'un fichier puisse sortir AVANT d'importer un backend qui n'est pas celui de cette
    # exécution (et qui peut être cassé sur cette machine).
    seq = env_cfg.seq if env_cfg else []
    remote = envs.remote_of(env_cfg)
    rc = 0
    driver_layer = env_cfg.driver_layer if env_cfg else None
    if driver_layer and driver_layer.pip:
        print(_hdr(f"\ninstall: {driver_layer.pip}"))
        # installed first so it's already satisfied when a project's own
        # dependencies (e.g. otrec -> optax -> jax) pull in the plain package
        if run_in_env(seq, [python(), "-m", "pip", "install", driver_layer.pip]) != 0: rc = 1
    # `errand` en premier : c'est lui qui lance tout le reste, et il n'a aucune dépendance --
    # donc il s'installe avant que quoi que ce soit d'autre existe, ce qui est exactement ce
    # qu'on lui demande.
    for proj in ["errand", "loom", "sdot", "otrec"]:
        if not remote and not (ROOT / proj / "pyproject.toml").exists(): continue
        print(_hdr(f"\ninstall: {proj}"))
        if run_in_env(seq, [python(), "-m", "pip", "install", "-e", proj]) != 0: rc = 1
    return rc


def cmd_toolchain(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    # le driver résolu, visible du processus fils : c'est ce que `loom.testing.driver_is` lit pour
    # qu'un fichier puisse sortir AVANT d'importer un backend qui n'est pas celui de cette
    # exécution (et qui peut être cassé sur cette machine).
    if env_cfg and env_cfg.driver:
        os.environ.setdefault("SDOT_DRIVER", env_cfg.driver)
    seq = env_cfg.seq if env_cfg else []
    return run_in_env(seq, [python(), "-m", "loom.toolchain"])


GONE = {
    "test"      : "errand",
    "bench"     : "errand -k bench",
    "experiment": "errand -k experiment",
}


def _say_gone(name):
    print(_err(f"`./run {name}` n'existe plus : c'est `{GONE[name]}`."))
    print(_dim("  errand --help    les motifs, les environnements, les paramètres"))
    print(_dim("  errand --tui     l'écran pour chercher, lancer et suivre"))
    print(_dim("  les environnements sont déclarés dans errandfile.py, à la racine"))
    return 2


def main(argv=None):
    EPILOG = """
Ce qui lance le travail est passé chez `errand` :
    errand                      tout ce qui doit passer
    errand -k bench "OtPlan*"   ce qui doit être rapide, dans ces fichiers
    errand --env lmo-cuda-jax   ailleurs
    errand --tui                l'écran

Ici il ne reste que de quoi fabriquer la machine :
    ./run install               les trois paquets en editable
    ./run toolchain             ce que la compilation trouve
    ./run env / env create      les environnements de .envs.py
    ./run build-sif             les images Apptainer
"""
    parser = argparse.ArgumentParser(prog="run", description="nsdot: la machine, pas le travail",
                                     epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    def add_shared(p):
        p.add_argument("--env", default=None, help="Environment from .envs.py")
        p.add_argument("--driver", default=None, help="Framework driver: jax, torch")
        p.add_argument("--fp", default=None, help="Floating-point precision (FP32, FP64)")
        p.add_argument("--device", default=None, help="Target device (cpu, cuda)")
        p.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")
    for name in GONE:
        p_gone = sub.add_parser(name, help=f"parti chez errand: {GONE[name]}", add_help=False)
        p_gone.add_argument("rest", nargs="*")

    p_inst = sub.add_parser("install", help="Editable install all packages"); add_shared(p_inst)
    p_tool = sub.add_parser("toolchain", help="Toolchain diagnostic"); add_shared(p_tool)
    p_sif = sub.add_parser("build-sif", help="Build Apptainer .sif images from .envs.py")
    p_sif.add_argument("--env", dest="env_name", help="Env to build (default: all Apptainer ones)")
    p_sif.add_argument("--force", action="store_true", help="Force rebuild")
    p_sif.add_argument("--fakeroot", action="store_true", help="Use --fakeroot for apptainer build")
    p_sif.add_argument("--scratch-dir", help="Scratch directory (APPTAINER_TMPDIR / CACHEDIR)")
    p_env = sub.add_parser("env", help="Environment management")
    p_env.add_argument("env_action", nargs="?", default="list", choices=["list", "create"])
    add_shared(p_env)

    args, _ = parser.parse_known_args(argv)
    if args.command in GONE:
        return _say_gone(args.command)

    from . import envs as _envs
    os.environ.update(_envs.build_env_vars(args))

    dispatch = {"install": cmd_install, "toolchain": cmd_toolchain,
                "build-sif": cmd_build_sif, "env": cmd_env}
    if args.command not in dispatch:
        parser.print_help()
        return 1
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
