"""Validate the authored/generated Blueprint boundary before rendering."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

BLUEPRINT_DIR = Path(__file__).resolve().parent
LEAN_DIR = BLUEPRINT_DIR.parent
REPO_ROOT = LEAN_DIR.parents[1]
MANIFEST_PATH = LEAN_DIR / "tools" / "claim-manifest" / "generated" / "manifest.json"
SOURCE_DIR = BLUEPRINT_DIR / "src"
GENERATED_BINDINGS = SOURCE_DIR / "generated" / "claim_bindings.tex"
CHAPTERS_DIR = REPO_ROOT / "src" / "latex" / "projects"

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
LEAN_RE = re.compile(r"\\lean\s*\{([^}]+)\}")
LEANOK_RE = re.compile(r"\\leanok\b")
CLAIM_BINDING_RE = re.compile(r"\\ppClaim(?:Proof)?Binding\s*\{([^}]+)\}")
PRINT_STUB_RE = re.compile(
    r"\\newcommand\{\\(?:lean|leanok|discussion|mathlibok|notready)\}(?:\[1\])?\{\}"
)
BLUEPRINT_MACRO_RE = re.compile(r"\\(?:lean|leanok|uses|notready)\b")
LEAN_SOURCE_RE = re.compile(r"\.lean\b|(?:^|[/{\s])(?:src/lean|packages)/[^}\s]+")
STATUS_VALUE_RE = re.compile(r"\\ppDeclareValue\{lean-verified-([^}]+)\}")
CAMEL_TEXTTT_RE = re.compile(r"\\texttt\{[^}]*[a-z][A-Z][^}]*\}")


def _renderer_module() -> ModuleType:
    path = LEAN_DIR / "tools" / "claim-manifest" / "render.py"
    spec = importlib.util.spec_from_file_location("claim_manifest_render", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authored_tex_files(source_dir: Path) -> list[Path]:
    return sorted(
        path for path in source_dir.rglob("*.tex") if "generated" not in path.parts
    )


def validate_unique_labels(paths: list[Path]) -> list[str]:
    owners: dict[str, Path] = {}
    errors: list[str] = []
    for path in paths:
        for label in LABEL_RE.findall(path.read_text()):
            if label in owners:
                errors.append(
                    f"duplicate Blueprint label {label!r}: {owners[label]} and {path}"
                )
            else:
                owners[label] = path
    return errors


def validate_authored_markers(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        # The printable target must define inert stubs for web-only commands;
        # these are command definitions, not authored verification assertions.
        text = path.read_text()
        if path.name == "print.tex" and path.parent.name == "macros":
            text = PRINT_STUB_RE.sub("", text)
        if LEAN_RE.search(text):
            errors.append(f"manually authored \\lean declaration link in {path}")
        if LEANOK_RE.search(text):
            errors.append(f"manually authored \\leanok verified marker in {path}")
    return errors


def validate_manifest_projection(manifest: dict, generated_path: Path) -> list[str]:
    errors: list[str] = []
    claim_ids = [claim["claim_id"] for claim in manifest["claims"]]
    duplicates = sorted(
        {claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1}
    )
    if duplicates:
        errors.append(f"duplicate manifest claim IDs: {duplicates}")

    expected = _renderer_module().render_blueprint(manifest)
    actual = generated_path.read_text() if generated_path.exists() else ""
    if actual != expected:
        errors.append(
            f"stale or hand-edited generated Blueprint bindings: {generated_path}"
        )

    expected_declarations = sorted(
        claim["declaration"]
        for claim in manifest["claims"]
        if claim["build_result"] == "checked"
    )
    actual_declarations = sorted(LEAN_RE.findall(actual))
    if actual_declarations != expected_declarations:
        errors.append(
            "generated \\lean links differ from checked manifest declarations: "
            f"expected={expected_declarations}, actual={actual_declarations}"
        )
    # Two slots per checked claim: the statement marker beside \lean, and the
    # proof marker LeanBlueprint colours independently (see render_blueprint).
    if len(LEANOK_RE.findall(actual)) != 2 * len(expected_declarations):
        errors.append(
            "generated \\leanok count differs from twice the checked manifest claim count"
        )
    return errors


def validate_claim_binding_references(paths: list[Path], manifest: dict) -> list[str]:
    known = {claim["claim_id"] for claim in manifest["claims"]}
    errors: list[str] = []
    for path in paths:
        for claim_id in CLAIM_BINDING_RE.findall(path.read_text()):
            if claim_id not in known:
                errors.append(
                    f"Blueprint claim binding {claim_id!r} is absent from manifest: {path}"
                )
    return errors


def validate_claim_labels_exist(chapters_dir: Path, manifest: dict) -> list[str]:
    r"""Assert every manifest claim still labels something in its own paper.

    The generated ``\ppLeanStatusList`` renders one ``\Cref{<claim-id>}`` per
    checked claim, so a claim whose manuscript label has been deleted or moved to
    another paper leaves a dangling cross-reference in the published verification
    appendix. LaTeX degrades that to a warning and a ``??``, which is exactly the
    kind of failure a release audit misses, so it is enforced here instead.

    This is the reverse direction from ``validate_publication_boundary``: that one
    stops Lean identifiers leaking *into* prose, this one stops the manifest
    outliving the prose it describes. Both are needed while claims are being
    withdrawn or retargeted between papers.
    """
    errors: list[str] = []
    labels_by_paper: dict[str, set[str]] = {}
    for claim in manifest["claims"]:
        paper = claim["paper"]
        if paper not in labels_by_paper:
            paper_src = chapters_dir / paper / "src"
            labels: set[str] = set()
            for path in sorted(paper_src.rglob("*.tex")):
                if "generated" in path.parts:
                    continue
                labels.update(LABEL_RE.findall(path.read_text()))
            labels_by_paper[paper] = labels
        if claim["claim_id"] not in labels_by_paper[paper]:
            errors.append(
                f"manifest claim {claim['claim_id']!r} no longer labels anything in "
                f"{paper}: withdraw or retarget the mapping.toml entry in the same "
                "change as the prose edit"
            )
    return errors


def validate_status_projections_are_current(
    chapters_dir: Path, manifest: dict
) -> list[str]:
    r"""Assert no paper's status file asserts a claim the manifest dropped.

    The reverse direction of :func:`validate_claim_labels_exist`. ``render.py``
    writes one ``lean_status.tex`` per paper, and a paper whose last claim is
    withdrawn would otherwise keep a stale projection asserting machine-checked
    backing forever -- the file is generated, so nobody reads it, and no other
    gate looks at it. That is the worst shape a verification claim can take.
    """
    errors: list[str] = []
    declared: dict[str, set[str]] = {}
    for claim in manifest["claims"]:
        declared.setdefault(claim["paper"], set()).add(claim["claim_id"])
    for path in sorted(chapters_dir.glob("*/src/generated/lean_status.tex")):
        paper = path.parents[2].name
        rendered = set(STATUS_VALUE_RE.findall(path.read_text()))
        orphaned = sorted(rendered - declared.get(paper, set()))
        if orphaned:
            errors.append(
                f"stale generated status projection for {paper}: asserts "
                f"{orphaned} which the manifest no longer declares; re-run "
                "src/lean/tools/claim-manifest/render.py"
            )
    return errors


def validate_publication_boundary(
    chapters_dir: Path, declarations: list[str]
) -> list[str]:
    errors: list[str] = []
    for path in sorted(chapters_dir.glob("*/src/chapters/*.tex")):
        text = path.read_text()
        if BLUEPRINT_MACRO_RE.search(text):
            errors.append(f"Blueprint macro leaked into publication chapter: {path}")
        if LEAN_SOURCE_RE.search(text):
            errors.append(f"Lean source path leaked into publication chapter: {path}")
        if CAMEL_TEXTTT_RE.search(text):
            errors.append(
                f"Lean-shaped declaration identifier leaked into publication chapter: {path}"
            )
        for declaration in declarations:
            if declaration and declaration in text:
                errors.append(
                    f"manifest declaration {declaration!r} leaked into publication chapter: {path}"
                )
    return errors


def validate(
    manifest_path: Path, source_dir: Path, generated_path: Path, chapters_dir: Path
) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    authored = authored_tex_files(source_dir)
    errors = validate_unique_labels(authored)
    errors.extend(validate_authored_markers(authored))
    errors.extend(validate_claim_binding_references(authored, manifest))
    errors.extend(validate_manifest_projection(manifest, generated_path))
    errors.extend(validate_claim_labels_exist(chapters_dir, manifest))
    errors.extend(validate_status_projections_are_current(chapters_dir, manifest))
    errors.extend(
        validate_publication_boundary(
            chapters_dir,
            [claim["declaration"] for claim in manifest["claims"]],
        )
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--generated", type=Path, default=GENERATED_BINDINGS)
    parser.add_argument("--chapters-dir", type=Path, default=CHAPTERS_DIR)
    args = parser.parse_args()

    errors = validate(args.manifest, args.source_dir, args.generated, args.chapters_dir)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        "[PASS] Blueprint labels, generated bindings, authored markers, "
        "manifest claim labels, and publication boundary"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
