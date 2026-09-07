#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "~=3.13.0"
# dependencies = []
# ///
"""Fail when a declared CLI tool is not provisioned by the Nix dev shell.

The repository declares the tools it expects on ``PATH`` in two machine-readable
places: the harness permission allowlist (``.agents/claude/settings.json``) and
every prek ``language: system`` hook's ``entry``. Neither declaration provisions
anything -- ``infra/nix/flake.nix`` does. When the two drift, the gap is
invisible on a maintainer's machine, because an interactive Nix shell inherits
the ambient ``PATH`` and a home-manager profile silently satisfies the lookup.
It then fails in CI and in fresh worktrees.

This contract closes that gap by classifying the provenance of every declared
tool. Rather than scrubbing ``PATH`` and re-running the repo's commands -- slow,
noisy, and non-deterministic -- it asks the narrower question that actually
discriminates: does the resolved path live under the Nix store?

    store     resolved under /nix/store   -- provisioned by the flake, portable
    ambient   resolved elsewhere          -- a machine-local profile answered
    missing   did not resolve at all      -- stale declaration, or absent package

Both failing classes are real defects, and they are fixed in opposite
directions: ``ambient``/``missing`` for a tool still in use means add it to the
flake; for a tool since retired it means delete the dead declaration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


type Provenance = Literal["store", "ambient", "missing"]

_SETTINGS: Final = Path(".agents/claude/settings.json")
_PRE_COMMIT_GLOB: Final = ".pre-commit-config.yaml"
_STORE_PREFIX: Final = "/nix/store/"

# Tools that are legitimately not provisioned by the flake. Every entry needs a
# reason: an unexplained exemption is indistinguishable from an unnoticed gap,
# which is the failure this contract exists to prevent.
_EXEMPT: Final[Mapping[str, str]] = {
    "nix": "bootstraps the dev shell; cannot be provided by the shell it builds",
    "bash": "POSIX shell interpreter, present on every supported platform",
    "sh": "POSIX shell interpreter, present on every supported platform",
}


class Tool(NamedTuple):
    """A declared tool together with the file that declares it."""

    name: str
    source: str


class Finding(NamedTuple):
    """The resolved provenance of one declared tool."""

    name: str
    provenance: Provenance
    resolved: str
    sources: tuple[str, ...]


def _iter_strings(node: object) -> Iterator[str]:
    """Yield every string anywhere in a decoded JSON document."""
    match node:
        case str():
            yield node
        case dict():
            for value in node.values():
                yield from _iter_strings(value)
        case list():
            for value in node:
                yield from _iter_strings(value)


def _is_tool_name(token: str) -> bool:
    """Reject shell metacharacters and paths, keeping bare executable names."""
    return bool(token) and all(char.isalnum() or char in "-_." for char in token)


def settings_tools(root: Path) -> list[Tool]:
    """Extract tool names from ``Bash(<tool> ...)`` permission entries."""
    path = root / _SETTINGS
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for entry in _iter_strings(document):
        if not entry.startswith("Bash("):
            continue
        # `Bash(gh pr create:*)` -> `gh`; `Bash(just:*)` -> `just`.
        head = entry.removeprefix("Bash(").split(")")[0].split(":")[0].split()
        if head and _is_tool_name(head[0]):
            names.add(head[0])
    return [Tool(name, _SETTINGS.as_posix()) for name in sorted(names)]


def system_hook_entries(document: str) -> Iterator[str]:
    """Yield the ``entry:`` of each local ``language: system`` hook.

    Deliberately a line scanner rather than a YAML parse: this contract must not
    itself depend on a package that might be missing, which would make the gate
    fail for the very reason it exists to report.
    """
    entry: str | None = None
    language: str | None = None
    # Set when `entry:` opened a YAML block scalar (`>-`, `|`, ...), so the next
    # more-indented line carries the command rather than the `entry:` line.
    awaiting_block = False
    for raw in document.splitlines():
        stripped = raw.strip()
        if awaiting_block:
            awaiting_block = False
            if stripped:
                entry = stripped
                continue
        if stripped.startswith("- id:"):
            if entry and language == "system":
                yield entry
            entry, language = None, None
        elif stripped.startswith("entry:"):
            value = stripped.removeprefix("entry:").strip().strip("\"'")
            if value.rstrip("+-") in {">", "|"}:
                entry, awaiting_block = None, True
            else:
                entry = value
        elif stripped.startswith("language:"):
            language = stripped.removeprefix("language:").strip().strip("\"'")
    if entry and language == "system":
        yield entry


def pre_commit_tools(root: Path) -> list[Tool]:
    """Extract the leading binary of every ``language: system`` hook entry."""
    tools: list[Tool] = []
    for path in sorted(root.rglob(_PRE_COMMIT_GLOB)):
        if ".git/" in path.as_posix():
            continue
        source = path.relative_to(root).as_posix()
        for entry in system_hook_entries(path.read_text(encoding="utf-8")):
            head = entry.split()
            if head:
                tools.append(Tool(head[0], source))
    return tools


def classify(name: str) -> tuple[Provenance, str]:
    """Resolve ``name`` on ``PATH`` and label where it came from."""
    resolved = shutil.which(name)
    if resolved is None:
        return "missing", "-"
    return ("store" if resolved.startswith(_STORE_PREFIX) else "ambient"), resolved


def findings(tools: Iterable[Tool]) -> list[Finding]:
    """Collapse declarations by tool name and resolve each one once."""
    sources: dict[str, set[str]] = {}
    for tool in tools:
        sources.setdefault(tool.name, set()).add(tool.source)
    resolved: list[Finding] = []
    for name in sorted(sources):
        provenance, path = classify(name)
        resolved.append(Finding(name, provenance, path, tuple(sorted(sources[name]))))
    return resolved


def violations(results: Iterable[Finding]) -> list[Finding]:
    """Return findings that are neither store-provided nor explicitly exempt."""
    return [
        finding
        for finding in results
        if finding.provenance != "store" and finding.name not in _EXEMPT
    ]


def main(root: Path = Path()) -> int:
    """Fail when a declared tool is not provisioned by the Nix dev shell."""
    root = root.resolve()
    results = findings([*settings_tools(root), *pre_commit_tools(root)])
    if not results:
        sys.stderr.write("toolchain-provenance: no declared tools found\n")
        return 1

    failed = violations(results)
    for finding in failed:
        detail = finding.resolved if finding.provenance == "ambient" else "not on PATH"
        sys.stderr.write(
            f"toolchain-provenance: {finding.name}: {finding.provenance} ({detail})\n"
            f"    declared by: {', '.join(finding.sources)}\n"
            "    fix: add it to infra/nix/flake.nix, or delete the dead declaration\n"
        )

    counts = {
        state: sum(finding.provenance == state for finding in results)
        for state in ("store", "ambient", "missing")
    }
    exempt = sum(finding.name in _EXEMPT for finding in results)
    sys.stdout.write(
        f"toolchain-provenance: {'FAIL' if failed else 'OK'} — "
        f"{len(results)} declared tools; {counts['store']} from the store, "
        f"{counts['ambient']} ambient, {counts['missing']} missing, "
        f"{exempt} exempt.\n"
    )
    return 1 if failed else 0


def cli() -> int:
    """Parse the repository root and run the provenance audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    args = parser.parse_args()
    return main(args.root)


if __name__ == "__main__":
    raise SystemExit(cli())
