# Generic build environments: decisions

- **Env = place + toolchain**: the user declares where (layers), loom adds / completes its toolchain layers.
- **Declared explicitly per platform** (no ssh probing at load); local detection on the starting machine stays.
- **Per project, in an untracked file**, with a message saying what is expected when nothing is found (not a user-wide file).
- **Typed declarations** (dataclasses, autocompletion), not loose keys; loom completes layers the user already gave.
- **Remote root**: `root` given = parent directory, loom appends `<project>`; absent = `loom-release/<project>`.
- **Missing layer + per-layer requirements check**, both generic in errand.
- **Asking**: loom pre-writes a file with placeholders (an address to fill in, or "not built here"); the user edits it; then a message points at it. No terminal prompts.
- **Scope**: ideally every starting machine × every target.
- **Compatibility**: old keys and `builders.toml` dropped; Hugo's file rewritten only with his consent.
- **File**: `loom-builders.py` at the project root, read by loom's provider, written as a template when absent, added to `.gitignore`; never rewritten.
- **Form**: `Builders` dataclass, one field per platform: `Here()`, list of errand layers (completed by loom), `Skip()`, `None`. Concern (Hugo): unusual installs must be possible, recipes accessible.
- **Scope**: full design + every case reachable with existing layers; cases needing errand on/to native Windows → explicit `Missing`, separate session.
- **Linux on a Mac**: template proposes a blank `Ssh`, local Docker (emulated, slow) commented.
- **Project extras**: `[tool.loom.release.<platform>]` in pyproject (`pip`, `conda`, `setup` script appended to the recipe / provision).
- **Whole recipe replaced**: documented location of loom's recipes; redefine the env in `errand-envs.py`.
