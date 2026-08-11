"""Environment management — micromamba and apptainer/sif wrappers.

Reads .hosts.toml for [envs] and [hosts] configuration.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[4]  # loom/src/loom/cli/envs.py → repo root
HOSTS_FILE = ROOT / ".hosts.toml"


def load_config() -> dict[str, Any]:
    if not HOSTS_FILE.exists():
        return {}
    with open(HOSTS_FILE, "rb") as f:
        return tomllib.load(f)


def get_env(name: str | None = None, driver: str | None = None) -> dict | None:
    """Find an environment by name or by driver requirement."""
    cfg = load_config()
    envs = cfg.get("envs", {})

    if name:
        return envs.get(name)

    if driver:
        for env_name, env_cfg in envs.items():
            if env_cfg.get("driver") == driver:
                return env_cfg

    # Default: first env, or the one named "default"
    if "default" in envs:
        return envs["default"]
    if envs:
        return next(iter(envs.values()))

    return None


def get_host(name: str) -> dict | None:
    cfg = load_config()
    return cfg.get("hosts", {}).get(name)


def wrap_command(cmd: list[str], env_cfg: dict | None) -> list[str]:
    """Wrap `cmd` with the appropriate launcher (micromamba / apptainer / none)."""
    if env_cfg is None:
        return cmd

    etype = env_cfg.get("type", "")

    if etype == "micromamba":
        # Already in the target env? Skip wrapping
        env_name = env_cfg["name"]
        if env_name in os.environ.get("CONDA_DEFAULT_ENV", ""):
            return cmd
        # Micromamba not installed (e.g., remote venvs)? Skip
        if shutil.which("micromamba") is None:
            return cmd
        return ["micromamba", "-n", env_name, "run", *cmd]

    if etype == "apptainer":
        image = env_cfg["image"]
        flags = env_cfg.get("flags", [])
        mounts = env_cfg.get("mounts", {})
        mount_flags = []
        for src, dst in mounts.items():
            mount_flags += ["--bind", f"{ROOT / src}:{dst}"]
        # PYTHONPATH for editable installs inside the container
        py_paths = env_cfg.get("pythonpath", [])
        full_cmd = list(cmd)
        if py_paths:
            pp = ":".join(str(ROOT / p) for p in py_paths)
            full_cmd = ["env", f"PYTHONPATH={pp}", *full_cmd]
        return ["apptainer", "exec", *flags, *mount_flags, image, *full_cmd]

    # Unknown env type — run bare
    return cmd


def build_env_vars(args) -> dict[str, str]:
    """Build env vars dict from CLI args (for the child process)."""
    env = {"PYTHONUNBUFFERED": "1"}  # always: real-time output over pipes
    if hasattr(args, "fp") and args.fp:
        env["SDOT_FTYPE"] = args.fp
    if hasattr(args, "device") and args.device:
        env["SDOT_DEVICE"] = args.device
        if args.device == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
    return env


def arg_overrides_to_env(args, params: dict) -> dict[str, str]:
    """Convert CLI --arg=value overrides to SDOT_ARG_* env vars for the harness.

    `params` is the dict of Param descriptors from the experiment/benchmark registry.
    """
    env = {}
    for pname, p in params.items():
        # Check if user passed --pname on CLI
        val = getattr(args, pname.replace("-", "_"), None)
        if val is not None:
            # Convert to string (the harness will parse back to the right type)
            env[f"SDOT_ARG_{pname.upper()}"] = str(val)
    return env
