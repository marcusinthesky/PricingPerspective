#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lean claim manifest extractor.

Reads `mapping.toml` (human-authored, see t24.1 blocker B1), runs
`lake env lean` on a generated snippet per entry to check the declaration and
its axiom set, and emits a schema-conformant `generated/manifest.json`.

Run from anywhere; it locates `src/lean` relative to this file:

    uv run src/lean/tools/claim-manifest/extract.py

Each entry records the declaration's elaborated `#check` type as
`statement_type`, which is hashed into `content_hash`. Pinning the *statement*
and not merely the declaration *name* is what makes silent statement drift
visible: weakening a mapped theorem in place otherwise leaves the name
resolvable and the axiom set clean, so nothing downstream would move.

Exit code is nonzero if any mapping entry fails validation (missing
declaration, `sorryAx`, an axiom outside the allowlist, or an unparsable
statement type). A manifest is
still written in that case with `build_result: "failed"` for the offending
claims, so callers can inspect *why* — but no renderer binding may treat a
"failed" claim as verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
LEAN_DIR = TOOL_DIR.parents[1]  # src/lean
SCHEMA_VERSION = "1.0"

# Mathlib's standard trust base. Anything else blocks generation.
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

AXIOMS_LINE_RE = re.compile(r"depends on axioms:\s*\[(.*?)\]")
NO_AXIOMS_RE = re.compile(r"does not depend on any axioms")


@dataclass
class CheckResult:
    build_result: str  # "checked" | "failed"
    axioms: list[str] = field(default_factory=list)
    statement_type: str = ""  # normalized `#check @decl` type; "" when unrecoverable
    reason: str = ""  # populated when build_result == "failed"
    raw_stdout: str = ""
    raw_stderr: str = ""


def module_of(file_rel: str) -> str:
    """Derive a Lean import path from a `packages/<Pkg>/<Pkg>/A/B.lean` path."""
    parts = Path(file_rel).parts
    if len(parts) < 3 or parts[0] != "packages":
        raise ValueError(
            f"cannot derive an import module from file path {file_rel!r}; "
            "expected 'packages/<Package>/<Package>/...'"
        )
    module_parts = parts[2:]
    if not module_parts or not module_parts[-1].endswith(".lean"):
        raise ValueError(f"file path {file_rel!r} does not end in .lean")
    module_parts = list(module_parts[:-1]) + [module_parts[-1][: -len(".lean")]]
    return ".".join(module_parts)


def run_lean_snippet(snippet: str, *, workdir: Path = LEAN_DIR, label: str = "snippet") -> tuple[int, str, str]:
    """Write `snippet` to a scratch file under src/lean and run `lake env lean` on it."""
    scratch = workdir / f"_claim_manifest_{label}.lean"
    scratch.write_text(snippet)
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", scratch.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            # Dominated by reading mathlib's .oleans, not by elaboration, so it is
            # cold-page-cache I/O that sets the ceiling: measured 2026-08-02, one
            # claim took 352 s cold and 10 s warm, while a sibling claim in the same
            # module spent 44 s in import alone. The old 180 s budget therefore
            # false-failed the first claims of a cold run — a real hazard now that
            # this gates CI, where a timeout is indistinguishable from a broken proof.
            timeout=600,
        )
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        scratch.unlink(missing_ok=True)


def parse_axioms(stdout: str) -> list[str] | None:
    """Extract the axiom list from `#print axioms` output, or None if absent."""
    # `#print axioms <decl>` wraps its `[...]` list across several lines when the
    # fully-qualified declaration name is long, so the single-line patterns below
    # would miss it. Flatten runs of whitespace to a single space first.
    flat = re.sub(r"\s+", " ", stdout)
    m = AXIOMS_LINE_RE.search(flat)
    if m:
        raw = m.group(1).strip()
        return [a.strip() for a in raw.split(",")] if raw else []
    if NO_AXIOMS_RE.search(flat):
        return []
    return None


def parse_statement_type(stdout: str, declaration: str) -> str:
    """Recover the `#check @<declaration>` type from the snippet's stdout.

    Without this the manifest pins only a declaration *name*, so weakening a
    theorem's statement in place leaves every gate green: the name still
    resolves, the axiom set is still clean, and `content_hash` never moves
    because it is computed over fields that all stayed the same. Recording the
    elaborated type makes a statement edit a visible manifest diff, which is
    what forces the paired disclosure to be re-reviewed (AGENTS.md 2.6,
    "statements are the API").

    The snippet prints `@<decl> : <type>` — wrapped across lines by the
    pretty-printer, at a width that depends on the declaration name's length —
    followed by the `#print axioms` verdict, which opens with `'<decl>'`. Slice
    between the two and collapse whitespace, so the stored type is stable under
    line breaking and changes only when the statement really changes.
    """
    # `#check @f` echoes `@f : ...` only when the `@` is meaningful; for a
    # declaration with no implicit or instance-implicit binders the elaborator
    # normalizes it away and the echo is a bare `f : ...`. Every claim mapped
    # today happens to carry implicit binders, so matching only the `@` form
    # would fail silently on the first fully-explicit theorem someone maps.
    # Try the `@` form first: it is unambiguous, and the bare form would
    # otherwise match one character into it.
    for marker in (f"@{declaration} :", f"{declaration} :"):
        start = stdout.find(marker)
        # Anchor to line start so a same-suffixed name inside the printed type
        # cannot be mistaken for the echo.
        if start == -1 or (start != 0 and stdout[start - 1] != "\n"):
            continue
        rest = stdout[start + len(marker) :]
        end = rest.find(f"'{declaration}'")
        if end != -1:
            rest = rest[:end]
        return re.sub(r"\s+", " ", rest).strip()
    return ""


def check_declaration(declaration: str, import_module: str, *, label: str) -> CheckResult:
    snippet = f"import {import_module}\n#check @{declaration}\n#print axioms {declaration}\n"
    code, out, err = run_lean_snippet(snippet, label=label)
    return classify(code, out, err, declaration=declaration)


def check_declaration_from_snippet_fixture(decl_source: str, declaration: str, *, label: str) -> CheckResult:
    """Test-only entry point: compile an inline declaration (no package import)
    and immediately `#check`/`#print axioms` it. Used to exercise the same
    classification logic against throwaway fixtures (e.g. a `sorry` proof or
    a fixture using a disallowed axiom) without needing them to be part of an
    importable Lake package.
    """
    snippet = f"{decl_source}\n#check @{declaration}\n#print axioms {declaration}\n"
    code, out, err = run_lean_snippet(snippet, label=label)
    return classify(code, out, err, declaration=declaration)


def classify(code: int, out: str, err: str, *, declaration: str) -> CheckResult:
    statement_type = parse_statement_type(out, declaration)
    if code != 0:
        return CheckResult(
            build_result="failed",
            statement_type=statement_type,
            reason="declaration not found or failed to elaborate (unknown identifier / changed namespace)",
            raw_stdout=out,
            raw_stderr=err,
        )
    axioms = parse_axioms(out)
    if axioms is None:
        return CheckResult(
            build_result="failed",
            statement_type=statement_type,
            reason="could not parse `#print axioms` output",
            raw_stdout=out,
            raw_stderr=err,
        )
    if "sorryAx" in axioms:
        return CheckResult(
            build_result="failed",
            axioms=axioms,
            statement_type=statement_type,
            reason="declaration depends on sorryAx (contains a `sorry`)",
            raw_stdout=out,
            raw_stderr=err,
        )
    unexpected = sorted(set(axioms) - ALLOWED_AXIOMS)
    if unexpected:
        return CheckResult(
            build_result="failed",
            axioms=axioms,
            statement_type=statement_type,
            reason=f"unexpected axiom(s) outside allowlist: {unexpected}",
            raw_stdout=out,
            raw_stderr=err,
        )
    if not statement_type:
        return CheckResult(
            build_result="failed",
            axioms=axioms,
            reason="could not parse the `#check` type; refusing to record a claim whose statement is unpinned",
            raw_stdout=out,
            raw_stderr=err,
        )
    return CheckResult(
        build_result="checked",
        axioms=axioms,
        statement_type=statement_type,
        raw_stdout=out,
        raw_stderr=err,
    )


def canonical_claims_json(claims: list[dict]) -> bytes:
    # Sort keys and use compact separators so the hash is stable across runs
    # and independent of dict insertion order.
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_manifest(mapping_path: Path, *, verbose: bool = False) -> tuple[dict, bool]:
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    with mapping_path.open("rb") as fh:
        mapping = tomllib.load(fh)
    entries = mapping.get("claims", [])

    claim_ids = [entry["claim_id"] for entry in entries]
    duplicate_ids = sorted({claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate claim_id value(s): {duplicate_ids}")

    claims: list[dict] = []
    all_ok = True
    for entry in entries:
        declaration = entry["declaration"]
        file_rel = entry["file"]
        label = re.sub(r"[^A-Za-z0-9_]", "_", declaration)
        try:
            import_module = module_of(file_rel)
            result = check_declaration(declaration, import_module, label=label)
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed claim, not a crash
            result = CheckResult(build_result="failed", reason=str(exc))

        if result.build_result != "checked":
            all_ok = False
            if verbose:
                print(f"[FAIL] {declaration}: {result.reason}", file=sys.stderr)
                if result.raw_stderr:
                    print(result.raw_stderr, file=sys.stderr)
        elif verbose:
            print(f"[OK]   {declaration}: axioms={result.axioms}", file=sys.stderr)

        claims.append(
            {
                "paper": entry["paper"],
                "claim_id": entry["claim_id"],
                "statement_summary": entry["statement_summary"],
                "declaration": declaration,
                "package": entry["package"],
                "file": file_rel,
                "build_result": result.build_result,
                "assumptions": entry.get("assumptions", []),
                "axioms": result.axioms,
                "statement_type": result.statement_type,
                "disclosure": entry["disclosure"] if result.build_result == "checked" else (
                    f"NOT verified — {result.reason}"
                ),
            }
        )

    content_hash = hashlib.sha256(canonical_claims_json(claims)).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claims": claims,
        "content_hash": content_hash,
    }
    return manifest, all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=TOOL_DIR / "mapping.toml")
    parser.add_argument("--out", type=Path, default=TOOL_DIR / "generated" / "manifest.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    manifest, ok = build_manifest(args.mapping, verbose=args.verbose)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.verbose:
        print(f"wrote {args.out} ({len(manifest['claims'])} claims, hash={manifest['content_hash'][:12]}…)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
