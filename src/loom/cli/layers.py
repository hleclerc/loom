"""Composable execution layers for ./run.

An environment is a `seq=[...]` of layers describing how to get from "run this
python program" to the actual subprocess — e.g. `seq=[Apptainer(...), Micromamba(...)]`
reads outside-in: apptainer wraps micromamba wraps the command.

Most layers just rewrite a `Command` (argv/env) and compose locally. `Remote` is
different: it must be the first layer in `seq`. It resolves the rest of the
sequence against the REMOTE root, serializes it to a shell string, and runs it
over ssh — with an rsync push before, and an rsync pull after of whatever
`pull` paths the caller asked for (deterministic, known ahead of time — no
runtime-declared marker mechanism; see `resolve`/`Remote.run`).

There's always a "current python" (`sys.executable` locally, `Remote.python`
as the default once inside a `Remote`) — `Micromamba`, `Venv` and `Apptainer`
are all just ways to override it, local or remote alike: a env=cuda-jax +
host=lmo combo picks up lmo's default python via `Remote`, then Apptainer
swaps in the container's own, e.g. `seq=[Remote(...), Apptainer(...)]`. To use
a specific micromamba/venv env on a remote host instead, stack it the same
way: `seq=[Remote(...), Micromamba("name")]` or `seq=[Remote(...), Venv(python=...)]`.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DIM = "\033[2m"
RESET = "\033[0m"


def _dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


@dataclass
class Command:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)

    def shell(self) -> str:
        assigns = [f"{k}={shlex.quote(v)}" for k, v in self.env.items()]
        return " ".join(assigns + [shlex.quote(a) for a in self.argv])


@dataclass
class Context:
    root: Path
    remote: bool = False


class Layer(Protocol):
    def wrap(self, cmd: Command, ctx: Context) -> Command: ...


def compose(seq: list[Layer], cmd: Command, ctx: Context) -> Command:
    """Fold layers innermost-first: seq=[A, B, C] -> A(B(C(cmd)))."""
    for layer in reversed(seq):
        cmd = layer.wrap(cmd, ctx)
    return cmd


def resolve(seq: list[Layer], cmd: Command, ctx: Context, pull: list[str] | None = None) -> Command | int:
    """Run `seq` against `cmd`.

    If `seq` starts with a `Remote` layer, it takes over execution entirely
    (ssh + rsync, then pulls back `pull` paths if given) and this returns the
    process's return code. Otherwise the layers are folded locally and the
    fully-wrapped `Command` is returned for the caller to run as a subprocess
    (`pull` is meaningless locally and ignored).
    """
    if seq and isinstance(seq[0], Remote):
        return seq[0].run(seq[1:], cmd, ctx.root, pull=pull)
    return compose(seq, cmd, ctx)


# ── layers ───────────────────────────────────────────────────────────────────

@dataclass
class Driver:
    """Pure metadata — records which framework (jax/torch) this env targets.
    A no-op in the command itself; `Env.driver` reads it back out of `seq`.

    `pip`, if set, is the exact requirement `cmd_install` pip-installs for
    this driver (e.g. `pip="jax[cuda13]"`) instead of relying on whatever
    plain `jax`/`torch` happens to come in transitively through a project's
    pyproject.toml dependencies — the CUDA/ROCm/... extra differs per env and
    hardware, so it can't live in a single shared pyproject.toml."""
    name: str
    pip: str | None = None

    def wrap(self, cmd: Command, ctx: Context) -> Command:
        return cmd


@dataclass
class Micromamba:
    name: str

    def wrap(self, cmd: Command, ctx: Context) -> Command:
        # These are local-only shortcuts (checking *this* process's env/PATH) —
        # meaningless once ctx.remote, since the remote shell's state is
        # invisible from here. Remote just always wraps; a missing micromamba
        # over there surfaces as a normal command failure (proper availability
        # checking is future work, see is_available()/install()).
        if not ctx.remote:
            if self.name in os.environ.get("CONDA_DEFAULT_ENV", ""):
                return cmd
            if shutil.which("micromamba") is None:
                return cmd
        # Use the env's own `python` on PATH rather than whatever interpreter
        # came in as argv[0] — matches Apptainer/Venv: each python-selecting
        # layer picks its own interpreter, overriding the incoming default.
        return Command(["micromamba", "-n", self.name, "run", "python", *cmd.argv[1:]], cmd.env)


@dataclass
class Apptainer:
    image: str
    flags: list[str] = field(default_factory=list)
    mounts: dict[str, str] = field(default_factory=dict)

    def wrap(self, cmd: Command, ctx: Context) -> Command:
        mount_flags = []
        for src, dst in self.mounts.items():
            mount_flags += ["--bind", f"{ctx.root / src}:{dst}"]
        image_path = str(ctx.root / self.image)
        # The container has its own `python` on PATH (baked in at image build
        # time) — swap it in for whatever interpreter came in as argv[0],
        # local host paths won't generally resolve inside the container.
        argv = ["python", *cmd.argv[1:]]
        return Command(
            ["apptainer", "exec", *self.flags, *mount_flags, image_path, *argv],
            cmd.env,
        )


@dataclass
class Venv:
    """Selects a specific interpreter directly — a plain venv's python, or
    any already-installed one. Works local or remote, same as Micromamba;
    overrides whatever default python it's stacked on top of."""
    python: str

    def wrap(self, cmd: Command, ctx: Context) -> Command:
        return Command([self.python, *cmd.argv[1:]], cmd.env)


@dataclass
class Remote:
    """Must be the outermost (first) layer in a `seq`. Runs the rest of the
    sequence on `host` over ssh, syncing the repo there first and pulling back
    `pull` paths (relative to root) after -- deterministic, known by the
    caller ahead of time; nothing declared at runtime by the remote process."""
    host: str
    remote_dir: str
    python: str = "python3"  # default interpreter on this host; inner layers (Micromamba/Venv/Apptainer) may override it
    apptainer_scratch: str | None = None

    def run(self, inner_seq: list[Layer], cmd: Command, local_root: Path, pull: list[str] | None = None) -> int:
        remote_ctx = Context(root=Path(self.remote_dir), remote=True)
        argv = [self.python, *cmd.argv[1:]] if cmd.argv else cmd.argv
        wrapped = compose(inner_seq, Command(argv, cmd.env), remote_ctx)
        # flush explicitly: see main.py's `_env_banner` for why (block
        # buffering once piped means an unflushed print here would appear
        # AFTER the ssh subprocess's own unbuffered output, not before it).
        print(_dim(f"  rsync push → {self.host}:{self.remote_dir}"), flush=True)
        push(local_root, self.host, self.remote_dir)
        inner = f"cd {self.remote_dir} && mkdir -p tmp && {wrapped.shell()}"
        # ssh runs this non-interactively, so the remote shell never sources
        # .bashrc/.zshrc -- anything installed via a shell-rc PATH addition
        # (micromamba, cargo, nvm, ...) is invisible. bash only reads .bashrc
        # for INTERACTIVE shells (login or not); zsh's .zshrc is the same.
        # -l alone (login) isn't enough -- force -i (interactive) through the
        # user's own $SHELL, not a hard-coded bash, to match what they'd get
        # from an actual interactive ssh session.
        remote_shell = f"$SHELL -ic {shlex.quote(inner)}"
        rc = subprocess.run(["ssh", self.host, remote_shell]).returncode
        if pull:
            print(_dim(f"  rsync pull ← {self.host}:{self.remote_dir} [{', '.join(pull)}]"), flush=True)
            pull_paths(pull, self.host, self.remote_dir, local_root)
        return rc


# ── sync / ssh plumbing ─────────────────────────────────────────────────────

RSYNC_EXCLUDES = [
    "build", "*.so", "*.o", "*.dylib", "node_modules", ".venv", "tmp",
    "__pycache__", "*.pyc", ".git", "dist", "*.egg-info", "*.sif",
]


def push(local_root: Path, host: str, remote_dir: str) -> None:
    cmd = ["rsync", "-a", "--delete", *[f"--exclude={e}" for e in RSYNC_EXCLUDES],
           f"{local_root}/", f"{host}:{remote_dir}/"]
    subprocess.run(cmd, check=True)


def pull_paths(paths: list[str], host: str, remote_dir: str, local_root: Path) -> None:
    """Best-effort: a `path` that never got created on the remote side (e.g.
    a test run that matched nothing) is not an error, just nothing to pull."""
    for p in paths:
        local_path = local_root / p
        local_path.mkdir(parents=True, exist_ok=True)
        # trailing slash on BOTH sides: syncs contents into local_path,
        # regardless of whether it already exists locally (e.g. from an
        # earlier same-hash run). Without it, rsync nests the remote dir
        # *inside* local_path whenever local_path already exists (dst-exists
        # ambiguity, same as `cp -r`) -- e.g. .../<hash>/.../<hash>/...
        subprocess.run(["rsync", "-a", f"{host}:{remote_dir}/{p}/", f"{local_path}/"])


# ── config (.envs.py) ────────────────────────────────────────────────────────
#
# .envs.py declares environments as function calls, not dict literals —
# `env(name, seq)` autocompletes its layers (a dict-of-dicts
# `envs = {"name": {"seq": ...}}` doesn't), and can be called from a loop to
# declare several at once. There's no separate "host" concept: a remote
# machine is just an env whose `seq` starts with `Remote(...)`, and no
# separate `driver` field either — it's a `Driver(...)` layer like any other.
# Share pieces across envs with plain Python (a variable, a small helper
# function, list concatenation) instead of a second registry, e.g.:
#
#   JAX = [Driver("jax")]
#   def lmo(*seq): return [Remote(host="lmo", remote_dir=..., python=...), *seq]
#   env("lmo", lmo() + JAX)
#   env("lmo-cuda-jax", lmo(Apptainer(image=...)) + JAX)

@dataclass
class Env:
    name: str
    seq: list[Layer]

    @property
    def driver_layer(self) -> Driver | None:
        return next((l for l in self.seq if isinstance(l, Driver)), None)

    @property
    def driver(self) -> str | None:
        layer = self.driver_layer
        return layer.name if layer else None


_envs: dict[str, Env] = {}


def env(name: str, seq: list[Layer]) -> None:
    """Register an environment. Call from .envs.py."""
    _envs[name] = Env(name, seq)


def load(path: Path) -> dict[str, Env]:
    """Exec `path` (a .envs.py) and return the envs it registered."""
    _envs.clear()
    spec = importlib.util.spec_from_file_location("_envs_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(_envs)
