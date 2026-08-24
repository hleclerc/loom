"""
Shared harness for experiments — the "leveled-up" cousin of `test()`/`bench()`
(loom.testing), for the one-per-file, heavier, explicitly-named case.

Two-phase protocol (same as loom.testing.test()/bench()):
  PHASE_COLLECT (SDOT_RUN_PHASE=0): module is imported, `experiment()`
  registers its metadata and returns None.
  PHASE_RUN (SDOT_RUN_PHASE=1): the selected entry runs with parsed CLI args.

Usage (exp_lung.py):
    from loom.cli import experiment, Param
    if p := experiment("lung BFGS", nb_diracs=Param(1000, help="Number of diracs")):
        print(f"Running with {p.nb_diracs} diracs")

CLI:
    ./run experiment lung --nb-diracs=5000
    ./run experiment lung --help     # auto-generated from Param declarations

For multiple tests/benchmarks per file (call-site granular, mixable in one
file), see `loom.testing.test`/`bench` instead — this module is only for the
one-entry-per-file, param-sweep case.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loom.testing import Args, Param

# ── phases ────────────────────────────────────────────────────────────────────
PHASE_COLLECT = 0
PHASE_RUN = 1

def _current_phase() -> int:
    """Read the CURRENT phase from the environment (re-evaluated on every call).

    NOT a module-level constant: _import_file() changes SDOT_RUN_PHASE between
    imports, so the phase must be read dynamically.
    """
    return int(os.environ.get("SDOT_RUN_PHASE", str(PHASE_RUN)))

# ── registry ──────────────────────────────────────────────────────────────────
# Populated during PHASE_COLLECT: entry_id → {kind, description, params, file}
_registry: dict[str, dict] = {}

# Parameter overrides from env vars (injected by the CLI runner via SDOT_ARG_*)
_arg_overrides: dict[str, str] = {}
for k, v in os.environ.items():
    if k.startswith("SDOT_ARG_"):
        _arg_overrides[k[9:]] = v  # strip SDOT_ARG_ prefix


# ── registration & dispatch ───────────────────────────────────────────────────

def experiment(description: str, **params: Param) -> Args | None:
    """Register an experiment (collect) or return its parsed args (run)."""
    return _register("experiment", description, params)


def _register(kind: str, description: str, params: dict[str, Param]) -> Args | None:
    # Determine caller's file via frame inspection
    frame = sys._getframe(2)  # _register → experiment/benchmark → caller module
    caller_file = frame.f_globals.get("__file__", "unknown")

    entry_id = Path(caller_file).stem  # e.g. "exp_lung"

    if _current_phase() == PHASE_COLLECT:
        _registry[entry_id] = dict(
            kind=kind, description=description, params=params, file=caller_file,
        )
        return None

    # ── PHASE_RUN: parse overrides ─────────────────────────────────────────
    parsed: dict[str, Any] = {}
    for pname, p in params.items():
        raw = _arg_overrides.get(pname.upper())
        if raw is not None:
            try:
                parsed[pname] = p.ptype(raw)
            except (ValueError, TypeError):
                print(f"  ⚠ Invalid --{pname}: {raw!r} (expected {p.ptype.__name__})",
                      file=sys.stderr)
                parsed[pname] = p.default
        else:
            parsed[pname] = p.default

    return Args(**parsed)


# ── CLI-facing helpers (used by main.py) ──────────────────────────────────────

def collect(roots: list[str]) -> dict[str, dict]:
    """Import all exp_*.py from `roots` in COLLECT mode.

    Returns the populated registry: entry_id → {kind, description, params, file}.
    """
    _registry.clear()
    for root_str in roots:
        root = Path(root_str)
        for pyfile in sorted(root.rglob("exp_*.py")):
            _import_file(pyfile)
    return dict(_registry)


def entries() -> list[dict]:
    return list(_registry.values())


def lookup(entry_id: str) -> dict | None:
    return _registry.get(entry_id)


def _import_file(path: Path) -> None:
    import importlib.util
    old = os.environ.get("SDOT_RUN_PHASE")
    os.environ["SDOT_RUN_PHASE"] = str(PHASE_COLLECT)
    try:
        spec = importlib.util.spec_from_file_location(
            f"_collect_{path.stem}", str(path)
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  ⚠ Failed to import {path}: {e}", file=sys.stderr)
    finally:
        if old is not None:
            os.environ["SDOT_RUN_PHASE"] = old
        else:
            os.environ.pop("SDOT_RUN_PHASE", None)
