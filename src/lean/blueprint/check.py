"""Validate the authored/generated Blueprint boundary before rendering."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

BLUEPRINT_DIR = Path(__file__).resolve().parent
LEAN_DIR = BLUEPRINT_DIR.parent
REPO_ROOT = LEAN_DIR.parents[1]
MANIFEST_PATH = LEAN_DIR / "tools" / "claim-manifest" / "generated" / "manifest.json"
SOURCE_DIR = BLUEPRINT_DIR / "src"
NATIVE_DIR = LEAN_DIR / ".lake" / "build" / "blueprint"
CHAPTERS_DIR = REPO_ROOT / "src" / "latex" / "projects"

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
LEAN_RE = re.compile(r"\\lean\s*\{([^}]+)\}")
LEANOK_RE = re.compile(r"\\leanok\b")
NATIVE_NODE_RE = re.compile(r"\\inputleannode\s*\{([^}]+)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
REFERENCE_RE = re.compile(r"\\(?:uses|proves|[Cc]ref|eqref|ref)\s*\{([^}]+)\}")
LEGACY_BINDING_RE = re.compile(r"\\ppClaim(?:Proof)?Binding\b")
PRINT_STUB_RE = re.compile(
    r"\\newcommand\{\\(?:lean|leanok|discussion|mathlibok|notready)\}(?:\[1\])?\{\}"
)
BLUEPRINT_MACRO_RE = re.compile(
    r"\\(?:lean|leanok|uses|notready|inputleannode|inputleanmodule|ppClaimBinding|ppClaimProofBinding)\b"
)
LEAN_SOURCE_RE = re.compile(r"\.lean\b|(?:^|[/{\s])(?:src/lean|packages)/[^}\s]+")
CAMEL_TEXTTT_RE = re.compile(r"\\texttt\{[^}]*[a-z][A-Z][^}]*\}")


def _tool_module(name: str) -> ModuleType:
    path = LEAN_DIR / "tools" / "claim-manifest" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"claim_manifest_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load claim tool at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _renderer_module() -> ModuleType:
    return _tool_module("render")


def tex_text(path: Path) -> str:
    """Ignore TeX comments while retaining escaped percent signs."""
    return re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())


def authored_tex_files(source_dir: Path) -> list[Path]:
    return sorted(
        path for path in source_dir.rglob("*.tex") if "generated" not in path.parts
    )


def reachable_tex_files(source_dir: Path, entry: Path) -> tuple[list[Path], list[str]]:
    """Follow authored inputs; external compiled native libraries stay optional."""
    root = source_dir.resolve()
    seen: set[Path] = set()
    errors: list[str] = []
    pending = [entry.resolve()]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            errors.append(f"missing authored Blueprint input: {path}")
            continue
        seen.add(path)
        for name in INPUT_RE.findall(tex_text(path)):
            target = (root / name).resolve()
            if target.suffix != ".tex":
                target = target.with_suffix(".tex")
            if target.is_relative_to(root):
                pending.append(target)
    return sorted(seen), errors


def validate_unique_labels(paths: list[Path]) -> list[str]:
    owners: dict[str, Path] = {}
    errors: list[str] = []
    for path in paths:
        for label in LABEL_RE.findall(tex_text(path)):
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
        text = tex_text(path)
        if path.name == "print.tex" and path.parent.name == "macros":
            text = PRINT_STUB_RE.sub("", text)
        if LEGACY_BINDING_RE.search(text):
            errors.append(f"retired custom Blueprint claim binding in {path}")
        if LEAN_RE.search(text):
            errors.append(f"manually authored \\lean declaration link in {path}")
        if LEANOK_RE.search(text):
            errors.append(f"manually authored \\leanok verified marker in {path}")
    return errors


def validate_native_projection(manifest: dict, native_dir: Path) -> list[str]:
    """Check compiler exports on the proof-certification path, without building."""
    try:
        extractor = _tool_module("extract")
        native = extractor.load_native_claims(rebuild=False, native_dir=native_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"cannot read native LeanArchitect exports: {exc}"]
    expected = {claim["claim_id"]: claim for claim in manifest["claims"]}
    errors: list[str] = []
    if native.keys() != expected.keys():
        errors.append(
            "native publication claim mismatch: "
            f"missing={sorted(expected.keys() - native.keys())}; "
            f"unclaimed={sorted(native.keys() - expected.keys())}"
        )
    for label in sorted(native.keys() & expected.keys()):
        node, claim = native[label], expected[label]
        if node["name"] != claim["declaration"]:
            errors.append(f"native declaration differs from manifest for {label!r}")
        summary = " ".join(node["statement"]["text"].split())
        if summary != " ".join(claim["statement_summary"].split()):
            errors.append(f"native statement differs from manifest for {label!r}")
        for part in (node.get("statement"), node.get("proof")):
            for dependency in (part or {}).get("usesLabels", []):
                if dependency not in native:
                    errors.append(
                        f"unknown native Blueprint dependency {dependency!r} in {label!r}"
                    )
    # LeanArchitect writes its inferred graph to the LaTeX artifacts. The JSON
    # `uses` fields describe authored overrides, not those inferred edges.
    try:
        modules: set[str] = set()
        for library in extractor.NATIVE_LIBRARIES:
            index = native_dir / "library" / f"{library}.json"
            modules.update(json.loads(index.read_text())["modules"])
            if not index.with_suffix(".tex").is_file():
                errors.append(
                    f"missing native LaTeX library: {index.with_suffix('.tex')}"
                )
        for module in sorted(modules):
            module_path = native_dir / module
            for item in json.loads(module_path.read_text()):
                if item.get("type") != "node":
                    continue
                node = item["data"]
                label = node["latexLabel"]
                artifact = module_path.with_suffix(".artifacts") / f"{label}.tex"
                text = tex_text(artifact)
                if LABEL_RE.findall(text) != [label]:
                    errors.append(
                        f"native LaTeX label differs from metadata: {artifact}"
                    )
                if LEAN_RE.findall(text) != [node["name"]]:
                    errors.append(
                        f"native LaTeX declaration differs from metadata: {artifact}"
                    )
                for group in REFERENCE_RE.findall(text):
                    for dependency in (part.strip() for part in group.split(",")):
                        if dependency and dependency not in native:
                            errors.append(
                                f"unknown inferred native Blueprint reference {dependency!r}: {artifact}"
                            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"cannot read native LeanArchitect LaTeX: {exc}")
    return errors


def validate_node_references(paths: list[Path], manifest: dict) -> list[str]:
    """Every approved node is included once; every informal edge resolves."""
    ids = [claim["claim_id"] for claim in manifest["claims"]]
    known = set(ids)
    errors: list[str] = []
    if len(known) != len(ids):
        errors.append("duplicate manifest claim IDs")
    includes: dict[str, list[Path]] = {}
    labels: set[str] = set()
    for path in paths:
        text = tex_text(path)
        labels.update(LABEL_RE.findall(text))
        for label in NATIVE_NODE_RE.findall(text):
            includes.setdefault(label, []).append(path)
            if label not in known:
                errors.append(f"unknown native Blueprint input {label!r}: {path}")
    for label in sorted(known):
        count = len(includes.get(label, []))
        if count != 1:
            errors.append(
                f"native Blueprint node {label!r} must appear once; found {count}"
            )
        if label in labels:
            errors.append(f"native Blueprint label {label!r} is also authored by hand")
    available = labels | (known & includes.keys())
    for path in paths:
        for group in REFERENCE_RE.findall(tex_text(path)):
            for label in (item.strip() for item in group.split(",")):
                # Printable command definitions refer to their arguments.
                if label and "#" not in label and label not in available:
                    errors.append(f"unknown Blueprint reference {label!r}: {path}")
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
                labels.update(LABEL_RE.findall(tex_text(path)))
            labels_by_paper[paper] = labels
        if claim["claim_id"] not in labels_by_paper[paper]:
            errors.append(
                f"manifest claim {claim['claim_id']!r} no longer labels anything in "
                f"{paper}: withdraw or retarget the mapping.toml entry in the same "
                "change as the prose edit"
            )
    return errors


def validate_status_projections_are_current(
    chapters_dir: Path,
    by_paper: dict[str, dict],
    fallback_manifest: dict,
    projects: set[str] | None = None,
) -> list[str]:
    """Require exact status, disclosure, list and own-manifest hash projections."""
    existing = {
        path.parents[2].name
        for path in chapters_dir.glob("*/src/generated/lean_status.tex")
        if not path.parents[2].name.startswith("_")
    }
    papers = set(by_paper) | existing
    if projects is not None:
        papers &= projects
    renderer = _renderer_module()
    errors: list[str] = []
    for paper in sorted(papers):
        path = chapters_dir / paper / "src/generated/lean_status.tex"
        if not path.is_file():
            errors.append(f"missing generated status projection for {paper}: {path}")
            continue
        expected = renderer.render_paper_status(
            by_paper.get(paper, fallback_manifest), paper
        )
        if path.read_text() != expected:
            errors.append(
                f"stale or hand-edited generated status projection for {paper}: {path}; "
                "regenerate from that paper's own manifest"
            )
    return errors


def validate_publication_boundary(
    chapters_dir: Path, declarations: list[str], projects: set[str] | None = None
) -> list[str]:
    errors: list[str] = []
    for path in sorted(chapters_dir.glob("*/src/chapters/*.tex")):
        if projects is not None and path.parents[2].name not in projects:
            continue
        text = tex_text(path)
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
    manifest_path: Path,
    source_dir: Path,
    native_dir: Path | None,
    chapters_dir: Path,
    projects: set[str] | None = None,
) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text())
        by_paper = _renderer_module().paper_manifests(manifest, manifest_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"cannot select publication manifests: {exc}"]
    all_claims = {
        "claims": [
            claim
            for paper, record in by_paper.items()
            for claim in record["claims"]
            if claim["paper"] == paper
        ]
    }
    publication_claims = {
        "claims": [
            claim
            for paper, record in by_paper.items()
            if projects is None or paper in projects
            for claim in record["claims"]
            if claim["paper"] == paper
        ]
    }
    authored = authored_tex_files(source_dir)
    errors = validate_unique_labels(authored)
    if projects is not None:
        for paper in sorted(projects):
            if not (chapters_dir / paper / "src" / "chapters").is_dir():
                errors.append(f"missing publication chapter directory: {paper}")
    errors.extend(validate_authored_markers(authored))
    entries = [source_dir / name for name in ("web.tex", "print.tex")]
    entries = [path for path in entries if path.is_file()]
    if not entries:
        entries = [source_dir / "content.tex"]
    for entry in entries:
        reachable, input_errors = reachable_tex_files(source_dir, entry)
        errors.extend(input_errors)
        errors.extend(
            f"{entry.name}: {error}"
            for error in validate_node_references(reachable, all_claims)
        )
    if native_dir is not None:
        errors.extend(validate_native_projection(all_claims, native_dir))
    errors.extend(validate_claim_labels_exist(chapters_dir, publication_claims))
    errors.extend(
        validate_status_projections_are_current(
            chapters_dir, by_paper, manifest, projects
        )
    )
    errors.extend(
        validate_publication_boundary(
            chapters_dir,
            [claim["declaration"] for claim in publication_claims["claims"]],
            projects,
        )
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument(
        "--native-dir",
        type=Path,
        help="Also validate already-built native LeanArchitect exports (no build).",
    )
    parser.add_argument("--chapters-dir", type=Path, default=CHAPTERS_DIR)
    parser.add_argument(
        "--project", action="append", help="Limit publication checks to these projects."
    )
    args = parser.parse_args()

    projects = set(args.project) if args.project else None
    errors = validate(
        args.manifest, args.source_dir, args.native_dir, args.chapters_dir, projects
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        "[PASS] Blueprint native inputs, references, publication labels, exact status "
        "projections, and publication boundary"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
