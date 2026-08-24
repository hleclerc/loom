"""Environment loading and dispatch.

Reads .envs.py (see loom/src/loom/cli/layers.py) for the [env name -> Env]
registry, and runs commands through it -- local subprocess, or shipped over
ssh when the env's `seq` starts with a Remote layer.
"""

from __future__ import annotations

from pathlib import Path

from . import layers

ROOT = Path(__file__).resolve().parents[4]  # loom/src/loom/cli/envs.py → repo root
ENVS_FILE = ROOT / ".envs.py"


def load_envs() -> dict[str, layers.Env]:
    if not ENVS_FILE.exists():
        return {}
    return layers.load(ENVS_FILE)


def get_env(name: str | None = None, driver: str | None = None) -> layers.Env | None:
    """Find an environment by name or by driver requirement."""
    envs = load_envs()

    if name:
        return envs.get(name)

    if driver:
        for e in envs.values():
            if e.driver == driver:
                return e

    # Default: the one named "default", or the first one
    if "default" in envs:
        return envs["default"]
    if envs:
        return next(iter(envs.values()))

    return None


def remote_of(env_cfg: layers.Env | None) -> layers.Remote | None:
    """The Remote layer at the front of `env_cfg.seq`, if any."""
    if env_cfg is None or not env_cfg.seq:
        return None
    first = env_cfg.seq[0]
    return first if isinstance(first, layers.Remote) else None


def apptainer_of(env_cfg: layers.Env) -> layers.Apptainer | None:
    return next((l for l in env_cfg.seq if isinstance(l, layers.Apptainer)), None)


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
        val = getattr(args, pname.replace("-", "_"), None)
        if val is not None:
            env[f"SDOT_ARG_{pname.upper()}"] = str(val)
    return env


# ── build-sif ─────────────────────────────────────────────────────────────────

def def_for_image(image_path: str) -> str:
    """Derive the .def file path from the .sif image path.

    >>> def_for_image('containers/cuda-jax.sif')
    'containers/cuda-jax.def'
    """
    p = Path(image_path)
    if p.suffix == ".sif":
        return str(p.with_suffix(".def"))
    raise ValueError(f"Expected a .sif image path, got: {image_path}")


def build_sif_command(apptainer: layers.Apptainer, *, force: bool = False, fakeroot: bool = False) -> list[str]:
    """Build the `apptainer build` command line for an Apptainer layer.

    Never goes through Apptainer.wrap() -- that runs a command *inside* the
    image, which doesn't exist yet while it's being built.
    """
    def_file = def_for_image(apptainer.image)
    cmd = ["apptainer", "build"]
    if fakeroot:
        cmd.append("--fakeroot")
    if force:
        cmd.append("--force")
    cmd += [apptainer.image, def_file]
    return cmd


def remote_build_sif_commands(
    remote: layers.Remote,
    apptainer: layers.Apptainer,
    *,
    force: bool = False,
    fakeroot: bool = False,
    scratch_dir: str | None = None,
) -> list[list[str]]:
    """The sequence of commands to build a .sif on a remote host: rsync the
    repo, then ssh + apptainer build. Returns commands for subprocess.run.

    scratch_dir overrides remote.apptainer_scratch; pass "" to suppress both.
    """
    def_file = def_for_image(apptainer.image)

    sd = scratch_dir if scratch_dir is not None else (remote.apptainer_scratch or "")
    env_prefix = " ".join(f"APPTAINER_{v}DIR={sd}" for v in ("TMP", "CACHE")) if sd else ""

    build_cmd_parts = ["apptainer", "build"]
    if fakeroot:
        build_cmd_parts.append("--fakeroot")
    if force:
        build_cmd_parts.append("--force")
    build_cmd_parts += [apptainer.image, def_file]
    build_cmd = " ".join(build_cmd_parts)

    rsync_excludes = " ".join(f"--exclude={e}" for e in layers.RSYNC_EXCLUDES)
    rsync_cmd = f"rsync -a {rsync_excludes} . {remote.host}:{remote.remote_dir}"
    ssh_cmd = f"ssh -t {remote.host} \"cd {remote.remote_dir} && {env_prefix} {build_cmd}\""

    return [["bash", "-c", rsync_cmd], ["bash", "-c", ssh_cmd]]
