#!/usr/bin/env python3
"""nsdot unified dev runner — CLI entry point. See README.md for usage."""

from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

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


# ── test ──────────────────────────────────────────────────────────────────────

def cmd_test(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    env_vars = envs.build_env_vars(args)
    filters = [args.name] if args.name else []
    failures = _run_cpp_tests(filters) + _run_python_tests(filters, args.project)
    print("\n" + "=" * 48)
    if failures:
        for label, name, phase in failures:
            print(f"  [{label}] {name}: {phase} FAILED")
        return 1
    print("  all good")
    return 0


def _run_python_tests(filters, project_filter):
    import importlib, importlib.util, traceback
    projects = ["loom", "sdot", "otrec"]
    if project_filter: projects = [p for p in projects if p in project_filter]
    files = []
    for proj in projects:
        td = ROOT / proj / "tests"
        if td.is_dir(): files += sorted(td.glob("test_*.py"))
    if not files: return []
    filter_names = [f for f in filters if "[" not in f]
    if filter_names:
        files = [f for f in files if any(n.split("::")[0].lower() in f.stem.lower() for n in filter_names)]
    if not files: return []
    tm = importlib.import_module("loom.testing")
    tm.test_phase = tm.PHASE_COLLECT
    tm.all_the_tests.clear()
    def _import_test(path):
        name = "_sdot_test__" + "__".join(str(path.resolve().relative_to(ROOT).with_suffix("")).split("/"))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod, name
    file_modules = {}
    for f in files: _, mname = _import_test(f); file_modules[mname] = f
    filter_tags = [f for f in filters if "[" in f]
    selected = [t for t in tm.all_the_tests if tm.matches(t, filter_names, filter_tags)]
    if not selected: return []
    print(f"\n{'='*12} [python] {len(selected)} test(s) {'='*12}", flush=True)
    tm.test_phase = tm.PHASE_RUN
    failures = []
    try:
        for t in selected:
            tm.test_filter = t
            where = f"{Path(t.file).name}:{t.line}"
            try:
                _import_test(file_modules[t.module])
                print(f"{tm.GREEN}PASS:{tm.RESET} {t.name} ({where})", flush=True)
            except Exception as e:
                print(f"{tm.RED}FAIL:{tm.RESET} {t.name} ({where}) - {e}", flush=True)
                traceback.print_exc(); sys.stdout.flush()
                failures.append(("python", t.name, "run"))
    finally:
        tm.test_phase = tm.PHASE_COLLECT; tm.test_filter = None
    return failures


def _run_cpp_tests(filters):
    from loom.compilation.adaptive_cpp import make_executable
    from loom.devices.Device import Device
    cpp_dir = ROOT / "loom" / "tests" / "cpp"
    if not cpp_dir.is_dir(): return []
    files = sorted(cpp_dir.glob("test_*.*"))
    if not files: return []
    filter_names = [f for f in filters if "[" not in f]
    if filter_names: files = [f for f in files if any(k.lower() in f.stem.lower() for k in filter_names)]
    failures = []
    for f in files:
        for dev_name in ["cuda", "cpu"]:
            device = Device.factory(dev_name)
            if not device.device_is_present: continue
            print(f"\n{'='*12} [{device}] {f.stem} {'='*12}", flush=True)
            try:
                exe = make_executable(f"{f.stem}_{device}", [f], device)
            except Exception as e:
                print(f"  BUILD-FAIL: {e}", flush=True); failures.append((device, f.stem, "build")); continue
            if subprocess.run([str(exe), *filters]).returncode: failures.append((device, f.stem, "run"))
    return failures


# ── experiment / bench ────────────────────────────────────────────────────────

def cmd_experiment(args): return _run_entry("experiment", args)
def cmd_bench(args): return _run_entry("benchmark", args)


def _run_entry(kind, args):
    from .harness import collect, lookup
    from . import envs
    name = getattr(args, f"{kind}_name", None)
    if not name:
        print(_err(f"Usage: ./run {kind} <name>")); _list_available(kind); return 1

    roots = [str(ROOT / p / "src" / p / "experiments") for p in ("loom", "sdot", "otrec")
             if (ROOT / p / "src" / p / "experiments").is_dir()]
    roots += [str(ROOT / p / "src" / p / "benchmarks") for p in ("loom", "sdot", "otrec")
              if (ROOT / p / "src" / p / "benchmarks").is_dir()]
    registry = collect(roots)

    entry_id = f"exp_{name}" if kind == "experiment" else f"bench_{name}"
    entry = lookup(entry_id)
    if entry is None:
        for eid, edata in registry.items():
            if name.lower() in edata["description"].lower(): entry = edata; break
    if entry is None:
        msg = _err(f"No {kind} found matching '{name}'")
        print(msg, file=sys.stderr, flush=True); print(msg, flush=True)
        _list_available(kind); return 1

    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    env_vars = envs.build_env_vars(args)
    env_vars["SDOT_RUN_PHASE"] = "1"

    py_root = ROOT
    if args.host:
        host = envs.get_host(args.host)
        if host: py_root = Path(host["remote_dir"])
    pp_parts = [str(py_root / p / "src") for p in ("loom", "sdot", "otrec")
                if (py_root / p / "src").is_dir() or args.host]
    existing_pp = os.environ.get("PYTHONPATH", "")
    if existing_pp: pp_parts.insert(0, existing_pp)
    env_vars["PYTHONPATH"] = ":".join(pp_parts)

    arg_overrides = envs.arg_overrides_to_env(args, entry["params"])
    env_vars.update(arg_overrides)

    if args.host: return _run_remote(kind, entry, args, env_vars, env_cfg)

    print(_hdr(f"\n{kind}: {entry['description']}"))
    print(_dim(f"  file: {entry['file']}"))
    if arg_overrides: print(_dim(f"  overrides: {arg_overrides}"))
    cmd = [python(), entry["file"]]
    cmd = envs.wrap_command(cmd, env_cfg)
    return run(cmd, env=env_vars)


def _run_remote(kind, entry, args, env_vars, env_cfg):
    from . import envs
    host = envs.get_host(args.host)
    if host is None: print(_err(f"Host '{args.host}' not found in .hosts.toml")); return 1
    hostname = host["hostname"]; remote_dir = host["remote_dir"]
    excludes = ["--exclude=build","--exclude=*.so","--exclude=*.o","--exclude=*.dylib",
                "--exclude=node_modules","--exclude=.venv","--exclude=tmp",
                "--exclude=__pycache__","--exclude=*.pyc","--exclude=.git",
                "--exclude=dist","--exclude=*.egg-info"]

    print(_hdr(f"syncing → {hostname}:{remote_dir}"))
    run(["rsync", "-a", "--delete", *excludes, f"{ROOT}/", f"{hostname}:{remote_dir}/"])

    remote_python = host.get("python", sys.executable)
    entry_name = getattr(args, f"{kind}_name")
    runner = f"{remote_dir}/run"
    env_assignments = []
    extra_flags = []
    for k, v in env_vars.items():
        env_assignments.append(f"{k}={v}")
        if k.startswith("SDOT_ARG_"):
            extra_flags.append(f"--{k[9:].lower().replace('_', '-')}={v}")
    env_str = " ".join(env_assignments); flags_str = " ".join(extra_flags)
    remote_cmd = f"cd {remote_dir} && mkdir -p tmp && {env_str} {remote_python} {runner} {kind} {entry_name} {flags_str}"
    ssh_cmd = ["ssh", hostname, remote_cmd]

    print(_hdr(f"executing on {hostname}"))
    print(_dim(f"  {remote_cmd}"))

    import selectors
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_files = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("OUTPUT:"):
            output_files.append(line.split("OUTPUT:", 1)[1].strip())
            print(_ok(f"  ← {line.partition('OUTPUT:')[2]}"))
        else:
            print(line)
    rc = proc.wait()

    if output_files:
        print(_hdr("retrieving outputs"))
        for path in output_files:
            local_path = ROOT / path; local_path.parent.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", f"{hostname}:{remote_dir}/{path}", str(local_path)])
    return rc


def _list_available(kind):
    from .harness import entries
    matching = [e for e in entries() if e["kind"] == kind]
    if matching:
        print(_dim(f"\n  Available {kind}s:"))
        for e in matching:
            eid = Path(e["file"]).stem
            ps = ", ".join(f"{n}={p.default!r}" for n, p in e["params"].items())
            print(_dim(f"    {eid:30s}  {e['description']}"))
            if e["params"]: print(_dim(f"    {'':30s}  params: {ps}"))


# ── other commands ────────────────────────────────────────────────────────────

def cmd_build_sif(args):
    """Build Apptainer .sif images from .def files.

    Reads .hosts.toml for apptainer-type environments and derives the .def file
    from the image path. Builds locally or remotely (via --host).
    """
    from . import envs

    apptainer_envs = envs.list_apptainer_envs()

    # Filter by --env if specified
    if args.env_name:
        if args.env_name not in apptainer_envs:
            print(_err(f"Unknown apptainer env: {args.env_name}"))
            print(_dim(f"  Available: {', '.join(apptainer_envs) or '(none)'}"))
            return 1
        apptainer_envs = {args.env_name: apptainer_envs[args.env_name]}

    if not apptainer_envs:
        print(_err("No apptainer environments found in .hosts.toml"))
        print(_dim("  Add an [envs.<name>] with type = \"apptainer\" and an image = \"containers/<name>.sif\" path."))
        return 1

    host_cfg = None
    if args.host:
        host_cfg = envs.get_host(args.host)
        if host_cfg is None:
            print(_err(f"Unknown host: {args.host}"))
            return 1

    rc = 0
    for env_name, env_cfg in apptainer_envs.items():
        def_file = envs.def_for_image(env_cfg["image"])
        if not (ROOT / def_file).exists():
            print(_err(f"  {env_name}: .def file not found: {def_file}"))
            rc = 1
            continue

        if host_cfg:
            print(_hdr(f"\nbuild-sif: {env_name} → {env_cfg['image']} (on {args.host})"))
            # Show which scratch dir will be used
            sd = args.scratch_dir or host_cfg.get("apptainer_scratch", "")
            if sd:
                print(_dim(f"  scratch: {sd}"))
            commands = envs.remote_build_sif(
                host_cfg, env_cfg,
                force=args.force, fakeroot=args.fakeroot,
                scratch_dir=args.scratch_dir,
            )
            for cmd in commands:
                if run(cmd) != 0:
                    rc = 1
                    break
        else:
            print(_hdr(f"\nbuild-sif: {env_name} → {env_cfg['image']}"))
            cmd = envs.build_sif_command(env_cfg, force=args.force, fakeroot=args.fakeroot)
            if args.scratch_dir:
                cmd_env = os.environ.copy()
                cmd_env["APPTAINER_TMPDIR"] = args.scratch_dir
                cmd_env["APPTAINER_CACHEDIR"] = args.scratch_dir
                print(_dim(f"  APPTAINER_TMPDIR={args.scratch_dir}"))
                if run(cmd, env=cmd_env) != 0:
                    rc = 1
            else:
                if run(cmd) != 0:
                    rc = 1

    return rc


def cmd_env(args):
    from . import envs
    cfg = envs.load_config()
    all_envs = cfg.get("envs", {})

    print(_hdr("\nEnvironments (.hosts.toml → [envs]):"))
    if not all_envs:
        print(_dim("  (none configured)"))
    else:
        # Determine the default env (replicate get_env resolution)
        if "default" in all_envs:
            default_name = "default"
        elif all_envs:
            default_name = next(iter(all_envs))
        else:
            default_name = None

        for name, ecfg in all_envs.items():
            marker = " ← default" if name == default_name else ""
            etype = ecfg.get("type", "?")
            driver = ecfg.get("driver", "?")
            extra = ""
            if etype == "micromamba":
                extra = f'name={ecfg.get("name","?")}'
            elif etype == "apptainer":
                extra = f'image={ecfg.get("image","?")}'
            print(f"  {name:18s}  driver={driver:6s}  type={etype:12s}  {extra}{marker}")
            # Show mounts for apptainer envs
            mounts = ecfg.get("mounts", {})
            if mounts:
                for src, dst in mounts.items():
                    print(f"  {'':18s}  mount: {src} → {dst}")

        print(_dim(f"\n  Select with: --env <name>  (or --driver <{', '.join(sorted(set(e.get('driver','?') for e in all_envs.values())))})>"))

    print(_hdr("\nHosts (.hosts.toml → [hosts]):"))
    hosts = cfg.get("hosts", {})
    if not hosts:
        print(_dim("  (none configured)"))
    else:
        for name, hcfg in hosts.items():
            scratch = hcfg.get("apptainer_scratch", "")
            extra = f'  scratch={scratch}' if scratch else ""
            print(f"  {name:18s}  hostname={hcfg.get('hostname','?')}{extra}")
        print(_dim("\n  Select with: --host <name>  |  builds sync via rsync → ssh"))

    return 0


def cmd_install(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver); rc = 0
    for proj in ["loom", "sdot", "otrec"]:
        proj_dir = ROOT / proj
        if not (proj_dir / "pyproject.toml").exists(): continue
        print(_hdr(f"\ninstall: {proj}"))
        cmd = [python(), "-m", "pip", "install", "-e", str(proj_dir)]
        cmd = envs.wrap_command(cmd, env_cfg)
        if run(cmd) != 0: rc = 1
    return rc


def cmd_toolchain(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    cmd = [python(), "-m", "loom.toolchain"]
    cmd = envs.wrap_command(cmd, env_cfg)
    return run(cmd)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    EPILOG = """
Environment selection (--env / --driver / --host):
  --env NAME      Use a specific environment from .hosts.toml (e.g. "cuda-jax", "torch")
  --driver jax    Auto-select the first env whose driver matches
  --host HOST     Run on the remote machine (uses the remote's python + env)
  --device cuda   Set SDOT_DEVICE and JAX_PLATFORMS for the child process
  --fp FP64       Set SDOT_FTYPE (FP32 or FP64)

  When neither --env nor --driver is given, the env named "default" is used.
  Run `./run env list` to see all configured environments and hosts.

  Examples:
    ./run test --env cuda-jax              # Run tests in a specific env
    ./run test --driver torch              # Auto-select the torch env
    ./run test --host lmo                  # Run tests remotely on lmo
    ./run test --env cuda-jax --host lmo   # Specific env on a remote host
    ./run build-sif --env cuda-jax         # Build a specific container image
    ./run build-sif --host lmo             # Build all containers on a remote host
"""
    parser = argparse.ArgumentParser(prog="run", description="nsdot unified dev runner",
                                     epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    def add_shared(p):
        p.add_argument("--env", default=None, help="Environment from .hosts.toml (default: auto by driver, then 'default')")
        p.add_argument("--driver", default=None, help="Framework driver: jax, torch")
        p.add_argument("--fp", default=None, help="Floating-point precision (FP32, FP64)")
        p.add_argument("--device", default=None, help="Target device (cpu, cuda)")
        p.add_argument("--host", default=None, help="Remote host (from .hosts.toml)")
        p.add_argument("-v", "--verbose", action="store_true")

    p_test = sub.add_parser("test", help="Run tests")
    add_shared(p_test)
    p_test.add_argument("--name", help="Test name filter")
    p_test.add_argument("--project", help="Restrict to project (loom, sdot, otrec)")

    p_exp = sub.add_parser("experiment", help="Run an experiment", add_help=False)
    add_shared(p_exp)
    p_exp.add_argument("experiment_name", nargs="?", help="Experiment name")
    p_exp.add_argument("-h", "--help", action="store_true", help="Show help with dynamic params")

    p_bench = sub.add_parser("bench", help="Run a benchmark", add_help=False)
    add_shared(p_bench)
    p_bench.add_argument("bench_name", nargs="?", help="Benchmark name")
    p_bench.add_argument("-h", "--help", action="store_true", help="Show help with dynamic params")

    p_inst = sub.add_parser("install", help="Editable install all packages"); add_shared(p_inst)
    p_tool = sub.add_parser("toolchain", help="Toolchain diagnostic"); add_shared(p_tool)
    p_sif = sub.add_parser("build-sif", help="Build Apptainer .sif images from .def files")
    p_sif.add_argument("--env", dest="env_name", help="Apptainer env to build from .hosts.toml (default: all)")
    p_sif.add_argument("--host", help="Remote host to build on (rsync → ssh → apptainer build)")
    p_sif.add_argument("--force", action="store_true", help="Force rebuild (pass --force to apptainer)")
    p_sif.add_argument("--fakeroot", action="store_true", help="Use --fakeroot for apptainer build")
    p_sif.add_argument("--scratch-dir", help="Scratch directory (APPTAINER_TMPDIR / APPTAINER_CACHEDIR)")
    p_env = sub.add_parser("env", help="Environment management")
    p_env.add_argument("env_action", nargs="?", default="list", choices=["list"])

    # Two-pass for dynamic params
    known, remaining = parser.parse_known_args(argv)
    wants_help = getattr(known, "help", False)
    if known.command in ("experiment", "bench") and (getattr(known, f"{known.command}_name", None) or wants_help):
        from .harness import collect, lookup
        roots = [str(ROOT / p / "src" / p / "experiments") for p in ("loom","sdot","otrec")
                 if (ROOT / p / "src" / p / "experiments").is_dir()]
        roots += [str(ROOT / p / "src" / p / "benchmarks") for p in ("loom","sdot","otrec")
                  if (ROOT / p / "src" / p / "benchmarks").is_dir()]
        registry = collect(roots)
        name = getattr(known, f"{known.command}_name")
        entry_id = f"exp_{name}" if known.command == "experiment" else f"bench_{name}"
        entry = lookup(entry_id) or next((e for eid, e in registry.items() if name.lower() in e["description"].lower()), None)
        if entry:
            target = p_exp if known.command == "experiment" else p_bench
            for pname, p in entry["params"].items():
                flag = f"--{pname.replace('_', '-')}"
                if p.ptype is bool: target.add_argument(flag, action="store_true", default=None, help=p.help)
                else: target.add_argument(flag, type=p.ptype, default=None, help=p.help)
                if p.choices: target._actions[-1].choices = p.choices
        if wants_help:
            target = p_exp if known.command == "experiment" else p_bench
            target.print_help()
            if entry: print(_hdr(f"\n  {entry['description']}")+_dim(f"\n  file: {entry['file']}"))
            return 0
        args = parser.parse_args(argv)
    else:
        args = known

    dispatch = {"test":cmd_test,"experiment":cmd_experiment,"bench":cmd_bench,
                "install":cmd_install,"toolchain":cmd_toolchain,"build-sif":cmd_build_sif,"env":cmd_env}
    if args.command not in dispatch: parser.print_help(); return 1
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
