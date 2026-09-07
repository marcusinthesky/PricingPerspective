"""Run LeanBlueprint 0.0.20 from this repository's nested Lake workspace.

Upstream assumes the Git and Lake roots coincide. Here Git is rooted three
levels above ``src/lean``, so its import-time repository discovery would choose
the wrong directory. This adapter changes only that discovery result. For
``checkdecls`` it delegates to the separately pinned checker project kept under
this task-owned directory, avoiding a mutation of the production Lake manifest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import git.repo

LEAN_DIR = Path(__file__).resolve().parent.parent
CHECKDECLS_DIR = LEAN_DIR / "blueprint" / "checkdecls"
DECLARATIONS = LEAN_DIR / "blueprint" / "lean_decls"


class _NestedLakeRepository:
    """Minimal GitPython-compatible repository view used by LeanBlueprint."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.working_dir = str(LEAN_DIR)


git.repo.Repo = _NestedLakeRepository  # type: ignore[misc,assignment]

import leanblueprint.client as client  # noqa: E402


def _check_declarations() -> None:
    if not DECLARATIONS.exists():
        raise FileNotFoundError(
            f"LeanBlueprint web extraction did not produce declaration list: {DECLARATIONS}"
        )
    subprocess.run(
        ["lake", "exe", "checkdecls", str(DECLARATIONS)],
        cwd=CHECKDECLS_DIR,
        check=True,
    )


client.do_checkdecls = _check_declarations


if __name__ == "__main__":
    client.safe_cli()
