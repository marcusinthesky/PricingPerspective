"""Build and inspect every first-party Lean module; compare the named gap inventory."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOL = Path(__file__).resolve().parent
LEAN = TOOL.parents[1]
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
KINDS = frozenset(
    {"sorry", "true-result", "true-iff", "true-definition", "true-premise"}
)
MARKER = "SORRY_AUDIT_JSON "
COUNT = "SORRY_AUDIT_COUNT "


def lean_tokens(source: str) -> list[tuple[str, int]]:
    """Lex debug preflight tokens, ignoring nested comments, strings and raw strings.

    This is not a theorem parser: Lean itself classifies placeholders. Quoted identifiers
    remain visible, so `set_option «debug».skipKernelTC` cannot evade the preflight.
    """
    tokens: list[tuple[str, int]] = []
    i = 0
    line = 1
    while i < len(source):
        start = i
        if source.startswith("--", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/-", i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith("/-", i):
                    depth += 1
                    i += 2
                elif source.startswith("-/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError("unterminated Lean block comment")
        elif source[i] == '"' or (source[i] == "r" and re.match(r'r#*"', source[i:])):
            raw = re.match(r'r(#+|)"', source[i:]) if source[i] == "r" else None
            if raw:
                i += raw.end()
                end = source.find('"' + raw[1], i)
                if end < 0:
                    raise ValueError("unterminated Lean raw string")
                i = end + 1 + len(raw[1])
            else:
                i += 1
                while i < len(source) and source[i] != '"':
                    i += 2 if source[i] == "\\" else 1
                if i >= len(source):
                    raise ValueError("unterminated Lean string")
                i += 1
        elif source[i] == "«":
            end = source.find("»", i + 1)
            if end < 0:
                raise ValueError("unterminated Lean quoted identifier")
            tokens.append((source[i + 1 : end], line))
            i = end + 1
        elif source[i] == "'" and (
            char := re.match(
                r"'(?:\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", source[i:]
            )
        ):
            i += char.end()
        elif source[i].isalpha() or source[i] == "_":
            i += 1
            while i < len(source) and (source[i].isalnum() or source[i] in "_'!?"):
                i += 1
            tokens.append((source[start:i], line))
        else:
            if not source[i].isspace():
                tokens.append((source[i], line))
            i += 1
        line += source[start:i].count("\n")
    return tokens


def check_debug_options(sources: list[Path]) -> None:
    """Reject debug options before any source is built, including multiline commands."""
    for path in sources:
        tokens = lean_tokens(path.read_text())
        for i, (token, line) in enumerate(tokens):
            if token != "set_option" or i + 1 == len(tokens):
                continue
            name = tokens[i + 1][0]
            j = i + 2
            while j + 1 < len(tokens) and tokens[j][0] == ".":
                name += "." + tokens[j + 1][0]
                j += 2
            name = name.removeprefix("_root_.")
            if name == "debug" or name.startswith("debug."):
                raise ValueError(f"{path}:{line}: forbidden set_option {name}")


def discover(lean: Path = LEAN) -> tuple[dict[str, Path], list[Path]]:
    """Discover modules independently of roll-up imports and Lake default targets."""
    modules: dict[str, Path] = {}
    sources = sorted(
        p for p in (lean / "packages").rglob("*.lean") if ".lake" not in p.parts
    )
    for path in sources:
        if path.name == "lakefile.lean":
            continue
        relative = path.relative_to(lean / "packages")
        module = ".".join(relative.with_suffix("").parts[1:])
        if not module or module in modules:
            raise ValueError(f"ambiguous source module: {path}")
        modules[module] = path
    main = lean / "Main.lean"
    if main.exists():
        modules["Main"] = main
        sources.append(main)
    # Native publication annotations are downstream of the mathematical packages.
    # They are still executable first-party Lean and receive the same preflight/census.
    publication = [lean / "Publication.lean", *sorted((lean / "Publication").rglob("*.lean"))]
    for path in publication:
        if path.is_file():
            module = ".".join(path.relative_to(lean).with_suffix("").parts)
            if module in modules:
                raise ValueError(f"ambiguous source module: {path}")
            modules[module] = path
            sources.append(path)
    if (lean / "lakefile.lean").exists():
        sources.append(lean / "lakefile.lean")
    if not modules:
        raise ValueError("no first-party Lean source modules found")
    return modules, sources


def parse_records(stdout: str) -> list[dict[str, Any]]:
    """Require a complete, unique census rather than accepting partial output."""
    records = [
        json.loads(line.split(MARKER, 1)[1])
        for line in stdout.splitlines()
        if MARKER in line
    ]
    counts = [
        int(line.split(COUNT, 1)[1]) for line in stdout.splitlines() if COUNT in line
    ]
    if len(counts) != 1 or counts[0] != len(records) or not records:
        raise ValueError("missing or incomplete compiled declaration census")
    if len({r["declaration"] for r in records}) != len(records):
        raise ValueError("duplicate compiled declaration")
    return records


def inventory_for(records: list[dict[str, Any]], modules: dict[str, Path]) -> list[str]:
    """Enforce trust unconditionally, then return named sorry/True inventory rows."""
    rows: set[str] = set()
    failures = []
    for record in records:
        name = record["declaration"]
        if record["module"] not in modules:
            raise ValueError(f"unexpected census module: {record['module']}")
        bad = set(record["axioms"]) - ALLOWED_AXIOMS - {"sorryAx"}
        if record["axiom"] or bad:
            failures.append(f"{name}: custom/nonstandard axiom {sorted(bad)}")
        if "sorryAx" in record["axioms"]:
            rows.add(f"sorry {name}")
        for kind in record["flags"]:
            if kind not in KINDS - {"sorry", "true-premise"}:
                raise ValueError(f"unexpected placeholder classification: {kind}")
            rows.add(f"{kind} {name}")
        for binder in record["premises"]:
            rows.add(f"true-premise {name}:{binder}")
    if failures:
        raise ValueError(
            "unallowlistable trust violation:\n" + "\n".join(sorted(failures))
        )
    return sorted(rows)


def read_allowlist(path: Path) -> list[str]:
    """Reject malformed/duplicate entries; prose is permitted only on comment lines."""
    rows = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        text = text.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2 or parts[0] not in KINDS:
            raise ValueError(f"{path}:{line}: malformed inventory row: {text}")
        kind, name = parts
        expected_colons = 1 if kind == "true-premise" else 0
        if (
            name.count(":") != expected_colons
            or name.startswith(":")
            or name.endswith(":")
        ):
            raise ValueError(f"{path}:{line}: malformed declaration/binder: {text}")
        if text != f"{kind} {name}":
            raise ValueError(f"{path}:{line}: noncanonical inventory row: {text}")
        if text in rows:
            raise ValueError(f"{path}:{line}: duplicate inventory row: {text}")
        rows.append(text)
    return sorted(rows)


def compare_inventory(actual: list[str], expected: list[str]) -> None:
    """Removed gaps must retire stale exceptions, just as new gaps need review."""
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected, actual, fromfile="allowlist", tofile="compiled", lineterm=""
            )
        )
        raise ValueError(
            "named inventory drift; review new and stale entries:\n" + diff
        )


def collect(modules: dict[str, Path], lean: Path = LEAN) -> list[dict[str, Any]]:
    """Rebuild every discovered source module before inspecting its imported environment."""
    subprocess.run(
        ["lake", "build", *sorted(modules)], cwd=lean, check=True, stdout=sys.stderr
    )
    # Dependencies may also provide Main.olean. Put freshly built first-party
    # products first so the census imports the intended local module.
    lake_path = subprocess.check_output(
        ["lake", "env", "printenv", "LEAN_PATH"], cwd=lean, text=True
    ).strip()
    local_paths = [lean / ".lake/build/lib/lean"] + [
        p / ".lake/build/lib/lean"
        for p in sorted((lean / "packages").iterdir())
        if p.is_dir()
    ]
    lean_path = os.pathsep.join([*(str(p) for p in local_paths), lake_path])
    with tempfile.TemporaryDirectory(prefix="lean-sorry-audit-") as temporary:
        snippet = Path(temporary) / "Census.lean"
        imports = "".join(f"import {module}\n" for module in sorted(modules))
        names = ", ".join("`" + module for module in sorted(modules))
        snippet.write_text(
            imports
            + (TOOL / "Audit.lean").read_text()
            + f"\nset_option maxHeartbeats 0 in\nrun_cmd liftTermElabM do SorryAudit.run #[{names}]\n"
        )
        result = subprocess.run(
            ["lake", "env", "env", f"LEAN_PATH={lean_path}", "lean", str(snippet)],
            cwd=lean,
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise ValueError("compiled census failed:\n" + result.stdout + result.stderr)
    return parse_records(result.stdout)


def main() -> None:
    """Use the same analyzer for the shell gate, tests and inventory review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="print current rows instead of comparing the allowlist",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check source debug options before a separate build",
    )
    parser.add_argument("--allowlist", type=Path, default=LEAN / "sorry-allowlist.txt")
    parser.add_argument(
        "--output", type=Path, help="write compiled JSON evidence to a fresh path"
    )
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise ValueError("evidence output exists; select a fresh path")
        modules, sources = discover()
        check_debug_options(sources)
        if args.preflight:
            print(
                f"sorry_gate: debug-option preflight passed for {len(sources)} source files."
            )
            return
        snapshot = {p: hashlib.sha256(p.read_bytes()).digest() for p in sources}
        records = collect(modules)
        current_modules, current_sources = discover()
        if (
            current_modules != modules
            or {p: hashlib.sha256(p.read_bytes()).digest() for p in current_sources}
            != snapshot
        ):
            raise ValueError(
                "source files changed during the compiled audit; rerun on a stable tree"
            )
        rows = inventory_for(records, modules)
        if args.output:
            with args.output.open("x") as stream:
                json.dump(
                    {
                        "modules": sorted(modules),
                        "declarations": records,
                        "inventory": rows,
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
        if args.inventory:
            print("\n".join(rows))
        else:
            compare_inventory(rows, read_allowlist(args.allowlist))
            print(
                f"sorry_gate: {len(records)} declarations in {len(modules)} modules; {len(rows)} reviewed gaps/True exceptions."
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"sorry_gate: ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
