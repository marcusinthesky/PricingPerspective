#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 - "$repository_root" <<'PYTHON'
"""Check shared flake/devenv pins and the executable Lean version declaration."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(sys.argv[1])


def main() -> None:
    flake = json.loads((ROOT / "infra/nix/flake.lock").read_text())
    devenv = json.loads((ROOT / "devenv.lock").read_text())
    for name in ("nixpkgs", "lean4-nix", "flake-utils", "systems"):
        if flake["nodes"][name]["locked"] != devenv["nodes"][name]["locked"]:
            raise ValueError(f"flake/devenv input differs: {name}")
    expected = {"nixpkgs", "lean4-nix", "flake-utils"}
    if set(flake["nodes"]["root"]["inputs"]) != expected:
        raise ValueError("flake contains unexpected environment inputs")
    if set(devenv["nodes"]["pricing-flake"]["inputs"]) != expected:
        raise ValueError("devenv imports a different flake input graph")
    provider = (ROOT / "infra/nix/pkgs/lean-toolchain.nix").read_text()
    tag = re.search(r'tag = "(v[0-9.]+)";', provider)
    if tag is None:
        raise ValueError("Lean provider tag is missing")
    expected_toolchain = f"leanprover/lean4:{tag.group(1)}"
    if (ROOT / "src/lean/lean-toolchain").read_text().strip() != expected_toolchain:
        raise ValueError("Lean provider and project toolchains differ")
    workspace = tomllib.loads((ROOT / "src/python/pyproject.toml").read_text())
    if workspace["tool"]["uv"]["workspace"]["members"] != [
        "apps/pipeline",
        "apps/hotelling",
        "packages/dotell",
    ]:
        raise ValueError("unexpected Python workspace members")
    print(f"Environment pins agree; {expected_toolchain}; three Python members.")


if __name__ == "__main__":
    main()
PYTHON
