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


def wrap_remote_command(remote_python: str, env_cfg: dict | None, remote_dir: str) -> str:
    """Build the command prefix for remote execution (SSH).

    Returns a shell string that replaces `remote_python` on the remote side.
    For apptainer envs: `apptainer exec <flags> --bind ... <image> python`
      (the container has its own python; host.python is irrelevant inside the container)
    For micromamba / unknown: returns `remote_python` unchanged (host's python is used directly).

    The caller should compose this into the SSH command, e.g.:
        ... {wrap_remote_command(python, env_cfg, dir)} {runner} {kind} ...
    """
    if env_cfg is None:
        return remote_python

    etype = env_cfg.get("type", "")

    if etype == "apptainer":
        image = env_cfg["image"]
        flags = env_cfg.get("flags", [])
        mounts = env_cfg.get("mounts", {})
        mount_flags = []
        for src, dst in mounts.items():
            mount_flags += ["--bind", f"{remote_dir}/{src}:{dst}"]
        # Use the container's own python (PATH is set by the image's %environment).
        parts = ["apptainer", "exec", *flags, *mount_flags, f"{remote_dir}/{image}", "python"]
        return " ".join(parts)

    return remote_python


def sif_exists_on_host(host_cfg: dict, env_cfg: dict) -> bool:
    """Check whether the .sif image exists on the remote host (via SSH)."""
    import subprocess
    hostname = host_cfg["hostname"]
    remote_dir = host_cfg["remote_dir"]
    image = env_cfg["image"]
    sif_path = f"{remote_dir}/{image}"
    rc = subprocess.run(
        ["ssh", hostname, "test", "-f", sif_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode
    return rc == 0


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


# ── build-sif ─────────────────────────────────────────────────────────────────

def list_apptainer_envs() -> dict[str, dict]:
    """Return {env_name: env_cfg} for all apptainer-type environments in .hosts.toml."""
    cfg = load_config()
    envs = cfg.get("envs", {})
    return {name: ec for name, ec in envs.items() if ec.get("type") == "apptainer"}


def def_for_image(image_path: str) -> str:
    """Derive the .def file path from the .sif image path.

    >>> def_for_image('containers/cuda-jax.sif')
    'containers/cuda-jax.def'
    """
    p = Path(image_path)
    if p.suffix == ".sif":
        return str(p.with_suffix(".def"))
    raise ValueError(f"Expected a .sif image path, got: {image_path}")


def build_sif_command(
    env_cfg: dict,
    *,
    force: bool = False,
    fakeroot: bool = False,
) -> list[str]:
    """Build the `apptainer build` command line for an apptainer environment.

    The .def file is derived from env_cfg["image"] (e.g. containers/cuda.sif → .def).
    """
    image = env_cfg["image"]
    def_file = env_cfg.get("def_file", def_for_image(image))
    cmd = ["apptainer", "build"]
    if fakeroot:
        cmd.append("--fakeroot")
    if force:
        cmd.append("--force")
    cmd += [image, def_file]
    return cmd


def remote_build_sif(
    host_cfg: dict,
    env_cfg: dict,
    *,
    force: bool = False,
    fakeroot: bool = False,
    scratch_dir: str | None = None,
) -> list[list[str]]:
    """Return the sequence of commands to build a .sif on a remote host.

    Steps: rsync the repo → ssh + apptainer build.

    scratch_dir overrides the host's apptainer_scratch.  When None, the host's
    apptainer_scratch is used automatically.  Set to "" to suppress both.

    Returns a list of commands; each command is a list[str] suitable for subprocess.
    """
    hostname = host_cfg["hostname"]
    remote_dir = host_cfg["remote_dir"]
    image = env_cfg["image"]
    def_file = env_cfg.get("def_file", def_for_image(image))

    # Resolve scratch dir: explicit arg → host config → none
    sd = scratch_dir
    if sd is None:
        sd = host_cfg.get("apptainer_scratch", "")
    tmpdir_flag = f"APPTAINER_TMPDIR={sd}" if sd else ""
    cachedir_flag = f"APPTAINER_CACHEDIR={sd}" if sd else ""

    rsync_excludes = "--exclude='build' --exclude='*.so' --exclude='node_modules' --exclude='.venv' --exclude='tmp' --exclude='*.sif'"

    rsync_cmd = f"rsync -a {rsync_excludes} . {hostname}:{remote_dir}"
    build_cmd_parts = ["apptainer", "build"]
    if fakeroot:
        build_cmd_parts.append("--fakeroot")
    if force:
        build_cmd_parts.append("--force")
    build_cmd_parts += [str(Path(remote_dir) / image), str(Path(remote_dir) / def_file)]
    build_cmd = " ".join(build_cmd_parts)

    env_parts = []
    if tmpdir_flag:
        env_parts.append(tmpdir_flag)
    if cachedir_flag:
        env_parts.append(cachedir_flag)
    env_prefix = " ".join(env_parts)
    ssh_cmd = f"ssh -t {hostname} \"cd {remote_dir} && {env_prefix} {build_cmd}\""

    return [["bash", "-c", rsync_cmd], ["bash", "-c", ssh_cmd]]
