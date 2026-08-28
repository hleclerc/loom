#!/usr/bin/env python3
"""nsdot unified dev runner — CLI entry point. See README.md for usage."""

from __future__ import annotations
import argparse, contextlib, copy, datetime, fnmatch, hashlib, io, itertools
import os, resource, shutil, subprocess, sys, time, traceback
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
# present when the entry has params (a short hash rather than a stringified
# param set, to keep paths short). Only the leaf is cleared+recreated before
# each run -- its ancestors accumulate history across envs/dates, which is
# exactly what the two rollup levels below summarize.
#
# ... EXCEPT for an experiment, which stops at {env}/: no date level. What a
# date buys is a HISTORY to compare (yesterday's timing against today's), and
# an experiment produces nothing comparable -- its output is a file a human
# opens. What it costs, there, is the one thing that matters: a path that
# changes under you, so the tab you left open on tmp/.../scene.html points at
# yesterday's run. Stable path, reload, done. One rollup level instead of two
# follows from that (see `_refresh_rollups`).
#
# Pulled back from a remote host as a single deterministic `tmp/{kind}`
# rsync (see cmd_test/cmd_bench) -- no runtime-declared marker mechanism.

_DATED_KINDS = ( "test", "bench" )


def _date_for(kind):
    """The `{date}` path component for `kind`, or None when it has none."""
    return _today_utc() if kind in _DATED_KINDS else None

def _slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s)).strip("_") or "_"


def _param_hash(resolved_params):
    if not resolved_params:
        return None
    blob = repr(sorted(resolved_params.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _entry_dirs(env_name, date_str, kind, entry, resolved_params):
    """Returns (leaf_dir, hash_dir): leaf_dir is where result.yaml/output.txt
    land for this run; hash_dir is its .../[param_hash]/ ancestor -- the root
    the rollup levels are computed under (.../[hash]/{env}/ and .../[hash]/).

    `date_str` None (an experiment, see above) makes .../[hash]/{env}/ itself
    the leaf."""
    label = f"{_slug(Path(entry.file).stem)}__{_slug(entry.name)}"
    hash_dir = ROOT / "tmp" / kind / label
    h = _param_hash(resolved_params)
    if h: hash_dir = hash_dir / h
    leaf = hash_dir / _slug(env_name)
    return (leaf / date_str if date_str else leaf), hash_dir


def _clear_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _today_utc():
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _resolve_for_path(params, overrides_env):
    """Same resolution as loom.testing.resolve_params, against an explicit
    {SDOT_ARG_X: val} dict rather than os.environ -- lets the (local)
    controller predict the exact hash_dir a --env remote run will use,
    *before* it has actually run, to pull back only that."""
    resolved = {}
    for pname, p in params.items():
        raw = overrides_env.get(f"SDOT_ARG_{pname.upper()}")
        if raw is not None:
            try: resolved[pname] = p.ptype(raw)
            except (ValueError, TypeError): resolved[pname] = p.default
        else:
            resolved[pname] = p.default
    return resolved


def _pull_dirs_for(kind, entries, env_name, overrides_env):
    """The exact set of hash_dir's (leaf's grandparent -- includes the
    refreshed rollups, not just the raw result) this invocation's entries
    will write to. tmp/ isn't touched by the repo push/--delete, so a remote
    host can carry old, unrelated runs -- pulling this precise set instead of
    the whole tmp/{kind} avoids dragging that back."""
    date_str = _date_for(kind)
    dirs = set()
    for e in entries:
        resolved = _resolve_for_path(e.params, overrides_env)
        _, hash_dir = _entry_dirs(env_name, date_str, kind, e, resolved)
        dirs.add(str(hash_dir.relative_to(ROOT)))
    return sorted(dirs)


def _ram_mb():
    """Peak RSS of this process so far -- a running high-water-mark (tests run
    in-process, reimported one after another), not an isolated per-entry
    measurement: it won't go back down between entries."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class _Tee:
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams: st.write(s)
        return len(s)
    def flush(self):
        for st in self._streams: st.flush()


@contextlib.contextmanager
def _capture_output():
    """Mirror stdout/stderr to a buffer while still printing live, so a run's
    text output can be saved to output.txt without losing the console feed."""
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, buf), _Tee(old_err, buf)
    try:
        yield buf
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _yaml_safe(v):
    """Best-effort sanitizer so an unpicklable value stuffed into p.results
    (a numpy/jax scalar, say) degrades to its repr() instead of crashing the
    write of an otherwise-successful run's result.yaml."""
    if isinstance(v, dict): return {k: _yaml_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_yaml_safe(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)): return v
    return repr(v)


def _write_result_yaml(out_dir, *, kind, entry, env_name, status, error,
                        duration_s, ram_mb, params, results, output_text):
    import yaml
    output_file = None
    if output_text.strip():
        output_file = "output.txt"
        (out_dir / output_file).write_text(output_text)
    data = {
        "name": entry.name,
        "file": str(Path(entry.file).relative_to(ROOT)),
        "line": entry.line,
        "kind": kind,
        "env": env_name,
        "status": status,
        "error": error,
        "duration_s": round(duration_s, 3),
        "ram_mb": round(ram_mb, 1),
        "params": _yaml_safe(params),
        "results": _yaml_safe(results),
        "output_file": output_file,
    }
    with open(out_dir / "result.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return data


def _numeric_summary(row):
    """min/max across p.results' numeric values (what a bench put there);
    falls back to duration_s when there's nothing numeric in results (e.g. a
    plain test) so every row still gets a meaningful pair."""
    vals = [v for v in row.get("results", {}).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        vals = [row["duration_s"]]
    return min(vals), max(vals)


def _write_rollup(dir_path, rows_by_label):
    """rows_by_label: {label: result.yaml-dict}. One row per label: status +
    min/max of that single run's numeric values (see _numeric_summary)."""
    import yaml
    entries = {}
    for label, row in sorted(rows_by_label.items()):
        lo, hi = _numeric_summary(row)
        entries[label] = {"status": row["status"], "min": lo, "max": hi}
    data = {
        "ok": sum(1 for r in rows_by_label.values() if r["status"] == "PASS"),
        "not_ok": sum(1 for r in rows_by_label.values() if r["status"] != "PASS"),
        "entries": entries,
    }
    with open(dir_path / "summary.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _refresh_rollups(hash_dir, env_name, dated=True):
    """Recompute the ancestor summaries after a result.yaml changed under
    `hash_dir` (re-scanning the filesystem, not an in-memory ledger -- self-
    healing, and correct however many separate invocations contributed):

    hash_dir/{env}/summary.yaml   rows = dates, this env only ("sans la date")
    hash_dir/summary.yaml         rows = "env/date", across all envs ("sans l'env")

    `dated=False` (an experiment: {env} IS the leaf, see `_entry_dirs`) leaves
    only the second level, its rows being the envs themselves -- there is no
    per-env history to summarize when each env holds exactly one run.
    """
    import yaml

    def _load(rf):
        return yaml.safe_load(rf.read_text()) if rf.exists() else None

    if not dated:
        by_env = {p.name: _load(p / "result.yaml")
                  for p in sorted(hash_dir.iterdir()) if p.is_dir()}
        _write_rollup(hash_dir, {k: v for k, v in by_env.items() if v})
        return

    env_dir = hash_dir / env_name
    by_date = {p.name: _load(p / "result.yaml") for p in sorted(env_dir.iterdir()) if p.is_dir()}
    _write_rollup(env_dir, {k: v for k, v in by_date.items() if v})

    by_env_date = {}
    for one_env_dir in sorted(p for p in hash_dir.iterdir() if p.is_dir()):
        for date_dir in sorted(p for p in one_env_dir.iterdir() if p.is_dir()):
            row = _load(date_dir / "result.yaml")
            if row:
                by_env_date[f"{one_env_dir.name}/{date_dir.name}"] = row
    _write_rollup(hash_dir, by_env_date)


# ── test / bench discovery ──────────────────────────────────────────────────────
#
# A pattern is a comma-separated list of `file[::name]` specs (both globbable
# with `*`). The file part locates a *.py anywhere in the repo by its FULL
# stem, `test_`/`bench_` prefix included ("test_Cell", not "Cell"), no
# project/directory distinction -- with no `*` it must resolve to exactly one
# file. The name part, if given, filters that file's test()/bench() entries by
# name (fnmatch; no `*` means exact). No pattern at all means "everything".

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "build", "dist", ".venv", "tmp",
    ".private", ".cache", "graphify-out",
}


def _iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info")]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


# Marker substrings identifying a file as plausibly declaring an entry. All
# three kinds register through `loom.testing` (test/bench/experiment), so one
# marker would do -- `loom.cli` is the second spelling only because the
# existing exp_*.py files import `experiment` from there (loom/cli/__init__.py
# re-exports it). Same repo-wide discovery for all three (see
# `_candidates_for`): no directory restricted to one kind.
_ENTRY_MARKERS = ("loom.testing", "from loom.cli import")


def _looks_like_entry_file(path):
    """Cheap text check (no import): does this file plausibly declare an
    entry? Without this, searching the whole repo makes any
    source file that happens to share its test/experiment's name (Cell.py
    vs. test_Cell.py -- the common case) a false ambiguity, or a wrong
    match, for what should be an unambiguous lookup."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return any(m in text for m in _ENTRY_MARKERS)


_CLI_DIR = Path(__file__).resolve().parent  # loom/src/loom/cli itself -- excluded below: its own
                                             # source necessarily contains _ENTRY_MARKERS' literal
                                             # strings (and usage-example mentions of them), so it
                                             # would otherwise self-match the marker check


def _candidates_for(project_filter=None):
    candidates = sorted(
        p for p in _iter_py_files(ROOT)
        if _CLI_DIR not in p.parents and _looks_like_entry_file(p)
    )
    if project_filter:
        candidates = [p for p in candidates if p.relative_to(ROOT).parts[0] == project_filter]
    return candidates


def _resolve_spec(spec, candidates):
    """spec: 'file[::name]', either side globbable with `*`, either side optional.
    Returns (matched_files: list[Path], name_glob: str | None)."""
    file_part, _, name_glob = spec.partition("::")
    file_part = file_part.strip()
    name_glob = name_glob.strip() if "::" in spec else None
    name_glob = name_glob or None

    if not file_part:
        return list(candidates), name_glob

    has_wild = "*" in file_part or "?" in file_part
    if has_wild:
        matched = [p for p in candidates if fnmatch.fnmatchcase(p.stem, file_part)]
    else:
        matched = [p for p in candidates if p.stem == file_part]
        if len(matched) > 1:
            names = ", ".join(str(p.relative_to(ROOT)) for p in matched)
            raise ValueError(f"'{file_part}' matches several files ({names}) -- use a glob (e.g. '{file_part}*') to select them all")
        # 0 matches is NOT an error here: the file part may name a C++ test
        # (loom/tests/cpp/, a separate discovery path) instead of a Python
        # one -- the caller decides what "nothing at all matched" means.

    return matched, name_glob


def _resolve_pattern(kind, pattern, project_filter):
    """Returns [(matched_files, name_glob), ...], one per comma-separated spec."""
    candidates = _candidates_for(project_filter)
    specs = [s.strip() for s in pattern.split(",")] if pattern else [""]
    return [_resolve_spec(s, candidates) for s in specs]


_SCAN_PKG = "_sdot_scan"  # synthetic namespace root, mirrors real directories -- never collides
                          # with an installed package, unlike using the real dir names directly
                          # (a file under sdot/tests/ is NOT part of the real `sdot` package, whose
                          # __path__ is sdot/src/sdot -- reusing "sdot" here would shadow/conflict).


def _import_test_file(path):
    """Import `path` under `_sdot_scan.<dir>.<dir>...<stem>`, ancestor
    directories registered as namespace packages with their real __path__ --
    so a relative import inside it (`from .sibling import x`) resolves
    against the file's actual directory. A flat synthetic name (the old
    scheme) has no package context at all, so those always failed. Ancestor
    packages are cached; only the leaf module is freshly re-executed on every
    call, needed for the test/bench reimport-per-entry model."""
    import importlib.util
    rel_parts = path.resolve().relative_to(ROOT).with_suffix("").parts
    pkg_name, pkg_dir = _SCAN_PKG, ROOT
    for part in rel_parts[:-1]:
        pkg_name, pkg_dir = f"{pkg_name}.{part}", pkg_dir / part
        if pkg_name not in sys.modules:
            pkg_spec = importlib.util.spec_from_loader(pkg_name, loader=None, is_package=True)
            pkg_mod = importlib.util.module_from_spec(pkg_spec)
            pkg_mod.__path__ = [str(pkg_dir)]
            sys.modules[pkg_name] = pkg_mod

    name = f"{pkg_name}.{rel_parts[-1]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, name


def _select_entries(kind, resolved_specs):
    """Import every file matched by any spec (collect phase), then keep the
    `kind` entries whose file+name matches the spec that selected that file."""
    from loom import testing as tm

    files = sorted({f for matched, _ in resolved_specs for f in matched}, key=str)
    tm.test_phase = tm.PHASE_COLLECT
    tm.all_entries.clear()
    file_modules = {}
    for f in files:
        _, mname = _import_test_file(f)
        file_modules[mname] = f

    selected = []
    for e in tm.all_entries:
        if e.kind != kind:
            continue
        efile = Path(e.file)
        for matched, name_glob in resolved_specs:
            if efile not in matched:
                continue
            if name_glob is None or fnmatch.fnmatchcase(e.name, name_glob):
                selected.append(e)
                break
    return selected, file_modules


def _entries_and_overrides(kind, args):
    """Resolve args.pattern -> (entries, file_modules, union-of-declared-params)."""
    resolved_specs = _resolve_pattern(kind, getattr(args, "pattern", None), getattr(args, "project", None))
    entries, file_modules = _select_entries(kind, resolved_specs)
    seen_params = {}
    for e in entries:
        seen_params.update(e.params)
    return entries, file_modules, seen_params


def _print_entries_help(kind, entries):
    if not entries:
        print(_dim(f"  no {kind} matched"))
        return
    from loom import testing as tm
    for e in entries:
        where = f"{Path(e.file).relative_to(ROOT)}:{e.line}"
        print(_hdr(f"\n{e.name}") + _dim(f"  ({where})"))
        if e.tags: print(_dim(f"  tags: {e.tag_text}"))
        for pname, p in e.params.items():
            flag = f"--{pname.replace('_', '-')}"
            print(f"  {flag:24s} default={p.default!r}  {p.help}")


def _run_entries(kind, entries, file_modules, env_name):
    """Run `entries` (already filtered to `kind`) one at a time, reimporting
    each entry's module so bodies execute isolated -- exactly the C++ harness's
    model. Each run gets its own tmp/{kind}/... leaf directory (see above), a
    result.yaml (status, timing, RAM, params, p.results, captured output), and
    the ancestor rollups get refreshed right after."""
    from loom import testing as tm
    if not entries:
        return []

    # The kernel build cache keys on the GENERATED .cpp alone -- the hand-written headers it
    # includes (sdot/include, loom/include) are not part of the hash (see
    # `compilation.adaptive_cpp.make_library`). So editing one of those and re-running the
    # tests would silently reuse the previous .dylib, and the run would say nothing about the
    # new code. Tests are where that matters most, so they default to level 1 (rebuild only
    # when those sources actually changed, via `_cpp_sources_hash` -- cheap on a cache hit)
    # unless the caller said otherwise: `SDOT_FORCE_BUILD=0` to trust the cache outright,
    # `=2` to force every kernel to rebuild regardless of the hash.
    os.environ.setdefault("SDOT_FORCE_BUILD", "1")

    print(f"\n{'='*12} [{kind}] {len(entries)} entrie(s) {'='*12}", flush=True)
    tm.test_phase = tm.PHASE_RUN
    failures = []
    date_str = _date_for(kind)
    try:
        for e in entries:
            tm.test_filter = e
            where = f"{Path(e.file).name}:{e.line}"
            resolved = tm.resolve_params(e.params)
            if e.params:
                summary = ", ".join(f"{k}={v!r}" for k, v in resolved.items())
                print(_dim(f"  {e.name} ({where}) -- {summary}"), flush=True)

            leaf_dir, hash_dir = _entry_dirs(env_name, date_str, kind, e, resolved)
            _clear_dir(leaf_dir)
            os.environ["SDOT_OUT_DIR"] = str(leaf_dir.relative_to(ROOT))

            t0 = time.perf_counter()
            status, error = "PASS", None
            with _capture_output() as buf:
                try:
                    _import_test_file(file_modules[e.module])
                except Exception as exc:
                    status, error = "FAIL", str(exc)
                    traceback.print_exc()
            duration_s = time.perf_counter() - t0

            if status == "PASS":
                print(f"{tm.GREEN}PASS:{tm.RESET} {e.name} ({where})", flush=True)
                # an experiment IS its output files -- name them, and their directory, so the
                # thing to open is in the terminal rather than to be reconstructed from the
                # naming scheme.
                if kind == "experiment":
                    written = sorted(f.name for f in leaf_dir.iterdir()
                                     if f.name not in ("result.yaml", "output.txt"))
                    print(_dim(f"  {leaf_dir.relative_to(ROOT)}/"), flush=True)
                    for name in written:
                        print(_dim(f"    {name}"), flush=True)
            else:
                print(f"{tm.RED}FAIL:{tm.RESET} {e.name} ({where}) - {error}", flush=True)
                sys.stdout.flush()
                failures.append((kind, e.name, "run"))

            _write_result_yaml(
                leaf_dir, kind=kind, entry=e, env_name=env_name,
                status=status, error=error, duration_s=duration_s, ram_mb=_ram_mb(),
                params=resolved, results=dict(tm.current_results), output_text=buf.getvalue(),
            )
            _refresh_rollups(hash_dir, _slug(env_name), dated=date_str is not None)
    finally:
        tm.test_phase = tm.PHASE_COLLECT; tm.test_filter = None
    return failures


def _cpp_filters(pattern):
    """Best-effort translation into cpp test filters: the C++ side (test_main.h)
    has its own, separate substring matcher, so `*`/`::` markup is stripped
    rather than interpreted -- only enough to narrow which test_*.cpp files
    to build (by filename substring) and what to pass the executable."""
    if not pattern:
        return []
    out = []
    for spec in pattern.split(","):
        file_part, _, name_part = spec.partition("::")
        out.append((file_part or name_part).strip().strip("*"))
    return [f for f in out if f]


# ── test ──────────────────────────────────────────────────────────────────────

def cmd_test(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    seq = env_cfg.seq if env_cfg else []
    # SDOT_ENV_NAME, if set, means we're the remote-side half of a dispatch:
    # use the ORIGINATING env's name for output-path labeling rather than
    # "default" (no --env crosses the ssh hop, precisely to avoid
    # re-resolving the same Remote layer and ssh'ing from the remote host
    # back out to itself -- see the run_in_env call below).
    env_name = os.environ.get("SDOT_ENV_NAME") or (env_cfg.name if env_cfg else "default")

    try:
        entries, file_modules, seen_params = _entries_and_overrides("test", args)
    except ValueError as e:
        print(_err(str(e)))
        return 1

    if envs.remote_of(env_cfg):
        overrides = envs.arg_overrides_to_env(args, seen_params)
        env_vars = envs.build_env_vars(args)
        env_vars.update(overrides)
        pattern_arg = [args.pattern] if getattr(args, "pattern", None) else []
        pull = _pull_dirs_for("test", entries, env_name, overrides)
        # NOT "--env {env_name}": the remote side re-resolves .envs.py itself,
        # and that name's seq still starts with the same Remote layer -- it'd
        # try to ssh from the remote host back out to itself. SDOT_ENV_NAME
        # only overrides the output-path label, without touching dispatch.
        env_vars["SDOT_ENV_NAME"] = env_name
        return run_in_env(seq, ["python", "./run", "test", *pattern_arg], env_vars, pull=pull)

    # SDOT_ENV_NAME set means we're the remote HALF of a dispatch: the
    # originating side already printed the correct banner before ssh'ing in
    # (see run_in_env) -- re-resolving --env-less here would just print a
    # wrong, confusing "local"/default one (the env actually active on this
    # process comes from the outer Micromamba/Apptainer wrapping, not from
    # re-reading .envs.py).
    if not os.environ.get("SDOT_ENV_NAME"):
        _env_banner(seq)
    if seen_params:
        os.environ.update(envs.arg_overrides_to_env(args, seen_params))

    cpp_ran, cpp_failures = _run_cpp_tests(_cpp_filters(getattr(args, "pattern", None)))
    failures = cpp_failures + _run_entries("test", entries, file_modules, env_name)
    if getattr(args, "pattern", None) and not entries and not cpp_ran:
        print(_err(f"No test matched '{args.pattern}'"))
        return 1
    print("\n" + "=" * 48)
    if failures:
        for label, name, phase in failures:
            print(f"  [{label}] {name}: {phase} FAILED")
        return 1
    print("  all good")
    return 0


def _run_cpp_tests(filters):
    """Returns (nb_files_matched, failures) -- the count lets the caller tell
    "nothing matched" apart from "matched, ran, all passed" (both look like
    an empty failures list otherwise)."""
    from loom.compilation.adaptive_cpp import make_executable
    from loom.devices.Device import Device
    cpp_dir = ROOT / "loom" / "tests" / "cpp"
    if not cpp_dir.is_dir(): return 0, []
    files = sorted(cpp_dir.glob("test_*.*"))
    filter_names = [f for f in filters if "[" not in f]
    if filter_names: files = [f for f in files if any(k.lower() in f.stem.lower() for k in filter_names)]
    if not files: return 0, []
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
    return len(files), failures


# ── bench ─────────────────────────────────────────────────────────────────────

def cmd_bench(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    seq = env_cfg.seq if env_cfg else []
    # see cmd_test's comment on SDOT_ENV_NAME
    env_name = os.environ.get("SDOT_ENV_NAME") or (env_cfg.name if env_cfg else "default")

    try:
        entries, file_modules, seen_params = _entries_and_overrides("bench", args)
    except ValueError as e:
        print(_err(str(e)))
        return 1

    if envs.remote_of(env_cfg):
        overrides = envs.arg_overrides_to_env(args, seen_params)
        env_vars = envs.build_env_vars(args)
        env_vars.update(overrides)
        pattern_arg = [args.pattern] if getattr(args, "pattern", None) else []
        pull = _pull_dirs_for("bench", entries, env_name, overrides)
        # see cmd_test's comment on SDOT_ENV_NAME vs "--env": passing --env
        # here would make the remote side re-resolve the same Remote layer
        # and try to ssh back out to itself.
        env_vars["SDOT_ENV_NAME"] = env_name
        return run_in_env(seq, ["python", "./run", "bench", *pattern_arg], env_vars, pull=pull)

    # see cmd_test's comment on SDOT_ENV_NAME vs re-printing this banner
    if not os.environ.get("SDOT_ENV_NAME"):
        _env_banner(seq)
    if seen_params:
        os.environ.update(envs.arg_overrides_to_env(args, seen_params))

    if not entries:
        print(_err(f"No bench matched '{args.pattern}'" if getattr(args, "pattern", None) else "No bench found"))
        return 1

    failures = _run_entries("bench", entries, file_modules, env_name)
    print("\n" + "=" * 48)
    if failures:
        for label, name, phase in failures:
            print(f"  [{label}] {name}: {phase} FAILED")
        return 1
    print("  all good")
    return 0


# ── experiment ────────────────────────────────────────────────────────────────

def cmd_experiment(args):
    """Same entries, same runner as test/bench (see `_run_entries`) -- an
    experiment differs only in what it is FOR: a file to look at, written to
    `p.out_dir`, whose path carries no date (see `_entry_dirs`).

    The one thing it has that the other two don't is the param SWEEP:
    `--nb-diracs=1000,2000` runs every combination, each into its own
    param_hash directory. It lives here rather than in `_run_entries` because
    that is where a sweep makes sense -- comparing pictures, not asserting.
    """
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    seq = env_cfg.seq if env_cfg else []
    # see cmd_test's comment on SDOT_ENV_NAME
    env_name = os.environ.get("SDOT_ENV_NAME") or (env_cfg.name if env_cfg else "default")

    try:
        entries, file_modules, seen_params = _entries_and_overrides("experiment", args)
    except ValueError as e:
        print(_err(str(e)))
        return 1

    if not entries:
        print(_err(f"No experiment matched '{args.pattern}'" if getattr(args, "pattern", None)
                   else "No experiment found"))
        _list_available()
        return 1

    combos = _expand_param_combos(args, seen_params)
    if combos is None:
        return 1

    if envs.remote_of(env_cfg):
        pattern_arg = [args.pattern] if getattr(args, "pattern", None) else []
        rc = 0
        for variants, combo_args in combos:
            overrides = envs.arg_overrides_to_env(combo_args, seen_params)
            env_vars = envs.build_env_vars(args)
            env_vars.update(overrides)
            pull = _pull_dirs_for("experiment", entries, env_name, overrides)
            # see cmd_test's comment on SDOT_ENV_NAME vs "--env"
            env_vars["SDOT_ENV_NAME"] = env_name
            # the sweep is expanded HERE, one plain run per combination -- the
            # remote side receives each value already split, as SDOT_ARG_*
            # env vars (same as test/bench), so it never re-splits an "a,b".
            rc = run_in_env(seq, ["python", "./run", "experiment", *pattern_arg], env_vars, pull=pull) or rc
        return rc

    # see cmd_test's comment on SDOT_ENV_NAME vs re-printing this banner
    if not os.environ.get("SDOT_ENV_NAME"):
        _env_banner(seq)

    failures = []
    for i, (variants, combo_args) in enumerate(combos):
        if len(combos) > 1:
            label = ", ".join(f"{k.replace('_','-')}={v}" for k, v in variants.items())
            # flush explicitly: stdout is fully buffered (not line-buffered) when
            # redirected/piped, so without this every sweep header would print only
            # at process exit -- all bunched after every child's own (unbuffered) output.
            print(_hdr(f"\n=== sweep [{i+1}/{len(combos)}] {label} ==="), flush=True)
        if seen_params:
            os.environ.update(envs.arg_overrides_to_env(combo_args, seen_params))
        failures += _run_entries("experiment", entries, file_modules, env_name)

    print("\n" + "=" * 48)
    if failures:
        for label, name, phase in failures:
            print(f"  [{label}] {name}: {phase} FAILED")
        return 1
    print("  all good")
    return 0


def _list_available():
    """Lists the files that plausibly declare an entry -- no import (a marker
    is a text check, not a guarantee), so just the names to pick from."""
    candidates = _candidates_for()
    if candidates:
        print(_dim("\n  Files declaring entries:"))
        for p in candidates:
            print(_dim(f"    {p.stem}  ({p.relative_to(ROOT)})"))


def _expand_param_combos(args, params):
    """Split any `--flag=a,b,c` dynamic param into a cartesian-product sweep.

    Dynamic params are parsed as raw strings (see main()'s two-pass argument
    registration) precisely so a `,` can be detected here, before type
    coercion. Any param whose raw value contains `,` becomes a multi-valued
    axis; the cartesian product of all such axes yields one argparse.Namespace
    per combination (a shallow copy of `args` with each axis pinned to one
    value). Params without `,` keep their single (coerced) value in every
    combination unchanged.

    Returns a list of (variants, Namespace) pairs, `variants` being just the
    {pname: value} that varied in that combination (for the run header) --
    empty/singleton when nothing was swept, so the non-sweep case is silent.
    Returns None (having already printed an error) if any value fails to
    parse or violates `choices` -- the caller should abort with rc=1.
    """
    axis_values = {}  # pname -> list[coerced values], only for non-bool params the user passed
    ok = True
    for pname, p in params.items():
        raw = getattr(args, pname, None)
        if raw is None or p.ptype is bool:
            continue
        parts = raw.split(",") if isinstance(raw, str) else [raw]
        try:
            values = [p.ptype(x) for x in parts]
        except (TypeError, ValueError):
            print(_err(f"--{pname.replace('_','-')}: cannot parse {raw!r} as {p.ptype.__name__}"))
            ok = False
            continue
        if p.choices:
            bad = [v for v in values if v not in p.choices]
            if bad:
                print(_err(f"--{pname.replace('_','-')}: invalid value(s) {bad} (choices: {p.choices})"))
                ok = False
                continue
        axis_values[pname] = values

    if not ok:
        return None

    names = list(axis_values)
    combos = list(itertools.product(*(axis_values[n] for n in names))) if names else [()]
    out = []
    for combo in combos:
        a = copy.copy(args)
        variants = {}
        for pname, val in zip(names, combo):
            setattr(a, pname, val)
            if len(axis_values[pname]) > 1:
                variants[pname] = val
        out.append((variants, a))
    return out


# ── other commands ────────────────────────────────────────────────────────────

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


def cmd_install(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    seq = env_cfg.seq if env_cfg else []
    remote = envs.remote_of(env_cfg)
    rc = 0
    driver_layer = env_cfg.driver_layer if env_cfg else None
    if driver_layer and driver_layer.pip:
        print(_hdr(f"\ninstall: {driver_layer.pip}"))
        # installed first so it's already satisfied when a project's own
        # dependencies (e.g. otrec -> optax -> jax) pull in the plain package
        if run_in_env(seq, [python(), "-m", "pip", "install", driver_layer.pip]) != 0: rc = 1
    for proj in ["loom", "sdot", "otrec"]:
        if not remote and not (ROOT / proj / "pyproject.toml").exists(): continue
        print(_hdr(f"\ninstall: {proj}"))
        if run_in_env(seq, [python(), "-m", "pip", "install", "-e", proj]) != 0: rc = 1
    return rc


def cmd_toolchain(args):
    from . import envs
    env_cfg = envs.get_env(name=args.env, driver=args.driver)
    seq = env_cfg.seq if env_cfg else []
    return run_in_env(seq, [python(), "-m", "loom.toolchain"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    EPILOG = """
Environment selection (--env / --driver):
  --env NAME      Use a specific environment from .envs.py (e.g. "cuda-jax", "lmo-cuda-jax")
  --driver jax    Auto-select the first env whose driver matches
  --device cuda   Set SDOT_DEVICE and JAX_PLATFORMS for the child process
  --fp FP64       Set SDOT_FTYPE (FP32 or FP64)

  When neither --env nor --driver is given, the env named "default" is used.
  A remote machine is just an env whose seq starts with a Remote layer --
  select it like any other env (no separate --host flag).
  Run `./run env` to see all configured environments.

Test / bench selection (positional pattern, comma-separated file[::name] specs):
  ./run test                             # everything
  ./run test Cell                        # everything in test_Cell.py
  ./run test Cell::batch                 # just "batch" in test_Cell.py
  ./run test "Cell::batch*,OtPlan1d::*"  # multiple specs
  A bare file part with no `*` must resolve to exactly one file.

  Examples:
    ./run test --env cuda-jax              # Run tests in a specific env
    ./run test --driver torch              # Auto-select the torch env
    ./run test --env lmo-cuda-jax          # Run tests on lmo, in a container
    ./run bench "OtPlan1d::*" --nb-diracs=5000
    ./run build-sif --env cuda-jax         # Build a specific container image
    ./run build-sif                        # Build every Apptainer-backed env
"""
    parser = argparse.ArgumentParser(prog="run", description="nsdot unified dev runner",
                                     epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    def add_shared(p):
        p.add_argument("--env", default=None, help="Environment from .envs.py (default: auto by driver, then 'default')")
        p.add_argument("--driver", default=None, help="Framework driver: jax, torch")
        p.add_argument("--fp", default=None, help="Floating-point precision (FP32, FP64)")
        p.add_argument("--device", default=None, help="Target device (cpu, cuda)")
        p.add_argument("-v", "--verbose", action="store_true")

    p_test = sub.add_parser("test", help="Run tests", add_help=False)
    add_shared(p_test)
    p_test.add_argument("pattern", nargs="?", help="file[::name] spec(s), comma-separated, `*` globbable")
    p_test.add_argument("--project", help="Restrict to a top-level directory (e.g. loom, sdot, otrec)")
    p_test.add_argument("-h", "--help", action="store_true", help="Show matched tests and their params")

    p_bench = sub.add_parser("bench", help="Run benchmarks", add_help=False)
    add_shared(p_bench)
    p_bench.add_argument("pattern", nargs="?", help="file[::name] spec(s), comma-separated, `*` globbable")
    p_bench.add_argument("--project", help="Restrict to a top-level directory (e.g. loom, sdot, otrec)")
    p_bench.add_argument("-h", "--help", action="store_true", help="Show matched benchmarks and their params")

    p_exp = sub.add_parser("experiment", help="Run an experiment", add_help=False)
    add_shared(p_exp)
    p_exp.add_argument("pattern", nargs="?", help="file[::name] spec(s), comma-separated, `*` globbable")
    p_exp.add_argument("--project", help="Restrict to a top-level directory (e.g. loom, sdot, otrec)")
    p_exp.add_argument("-h", "--help", action="store_true", help="Show matched experiments and their params")

    p_inst = sub.add_parser("install", help="Editable install all packages"); add_shared(p_inst)
    p_tool = sub.add_parser("toolchain", help="Toolchain diagnostic"); add_shared(p_tool)
    p_sif = sub.add_parser("build-sif", help="Build Apptainer .sif images from .envs.py")
    p_sif.add_argument("--env", dest="env_name", help="Env to build from .envs.py (default: all with an Apptainer layer)")
    p_sif.add_argument("--force", action="store_true", help="Force rebuild (pass --force to apptainer)")
    p_sif.add_argument("--fakeroot", action="store_true", help="Use --fakeroot for apptainer build")
    p_sif.add_argument("--scratch-dir", help="Scratch directory (APPTAINER_TMPDIR / APPTAINER_CACHEDIR)")
    p_env = sub.add_parser("env", help="Environment management")
    p_env.add_argument("env_action", nargs="?", default="list", choices=["list"])

    # Two-pass for dynamic params
    known, remaining = parser.parse_known_args(argv)
    wants_help = getattr(known, "help", False)

    if known.command in ("test", "bench", "experiment") and (getattr(known, "pattern", None) or wants_help):
        target = {"test": p_test, "bench": p_bench, "experiment": p_exp}[known.command]
        try:
            entries, _, seen_params = _entries_and_overrides(known.command, known)
        except ValueError as e:
            print(_err(str(e)))
            return 1
        for pname, p in seen_params.items():
            flag = f"--{pname.replace('_', '-')}"
            if p.ptype is bool: target.add_argument(flag, action="store_true", default=None, help=p.help)
            else:
                # type=str (not p.ptype): argparse's own type= runs too early
                # to see the raw string if a sweep-style "a,b,c" needs splitting.
                target.add_argument(flag, type=str, default=None, help=p.help)
        if wants_help:
            _print_entries_help(known.command, entries)
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
