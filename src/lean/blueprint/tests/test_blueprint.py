"""Behavioral regressions for native Blueprint and publication boundaries."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

CHECK_PATH = Path(__file__).resolve().parents[1] / "check.py"
SPEC = importlib.util.spec_from_file_location("blueprint_check", CHECK_PATH)
assert SPEC is not None and SPEC.loader is not None
blueprint_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blueprint_check)


def _manifest(claims: list[dict] | None = None, digest: str = "0") -> dict:
    return {"content_hash": digest * 64, "claims": claims or []}


def _claim(paper: str = "01_paper", label: str = "thm:kept") -> dict:
    return {
        "paper": paper,
        "claim_id": label,
        "declaration": "Example.kept" if paper == "01_paper" else "Example.dedicated",
        "statement_summary": "A conditional mathematical statement.",
        "build_result": "checked",
        "disclosure": "Checked under the stated assumptions.",
    }


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _native(root: Path, claims: list[dict]) -> Path:
    base = root / "native"
    for name in ("PricingPerspective", "WassersteinGeometry"):
        _write(base / f"library/{name}.json", json.dumps({"modules": []}))
        _write(base / f"library/{name}.tex", "% Native library\n")
    _write(base / "library/Publication.tex", "% Native library\n")
    _write(base / "library/Publication.json", json.dumps({"modules": ["module/Example.json"]}))
    _write(
        base / "module/Example.json",
        json.dumps(
            [
                {
                    "type": "node",
                    "data": {
                        "name": claim["declaration"],
                        "latexLabel": claim["claim_id"],
                        "hasLean": True,
                        "notReady": False,
                        "location": {"module": "Example"},
                        "statement": {
                            "text": claim["statement_summary"],
                            "usesLabels": [],
                        },
                        "proof": {"usesLabels": []},
                    },
                }
                for claim in claims
            ]
        ),
    )
    for claim in claims:
        _write(
            base / "module/Example.artifacts" / f"{claim['claim_id']}.tex",
            f"\\label{{{claim['claim_id']}}}\\lean{{{claim['declaration']}}}\n",
        )
    return base


def _fixture(root: Path, *, dedicated: bool = False) -> dict:
    tool = root / "claim-manifest"
    first = _claim()
    records = {"01_paper": _manifest([first])}
    _write(tool / "mapping.toml", '[[claims]]\npaper="01_paper"\nclaim_id="thm:kept"\n')
    manifest_path = _write(
        tool / "generated/manifest.json", json.dumps(records["01_paper"])
    )
    if dedicated:
        second = _claim("06_paper", "thm:dedicated")
        records["06_paper"] = _manifest([second], "1")
        _write(
            tool / "paper6-publication.toml",
            '[[claims]]\npaper="06_paper"\nclaim_id="thm:dedicated"\n',
        )
        _write(
            tool / "generated/paper6-publication.json", json.dumps(records["06_paper"])
        )
    source, chapters = root / "blueprint", root / "projects"
    claims = [claim for record in records.values() for claim in record["claims"]]
    _write(
        source / "content.tex",
        "".join(f"\\inputleannode{{{claim['claim_id']}}}\n" for claim in claims),
    )
    renderer = blueprint_check._renderer_module()
    for paper, record in records.items():
        _write(
            chapters / paper / "src/chapters/model.tex",
            "".join(
                f"\\label{{{claim['claim_id']}}} A claim.\n"
                for claim in record["claims"]
            ),
        )
        _write(
            chapters / paper / "src/generated/lean_status.tex",
            renderer.render_paper_status(record, paper),
        )
    return {
        "manifest": manifest_path,
        "source": source,
        "chapters": chapters,
        "native": _native(root, claims),
        "records": records,
    }


def _validate(fixture: dict, *, native: bool = False) -> list[str]:
    return blueprint_check.validate(
        fixture["manifest"],
        fixture["source"],
        fixture["native"] if native else None,
        fixture["chapters"],
    )


def test_duplicate_and_shadowed_labels_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = _write(root / "a.tex", "\\label{thm:same}")
        second = _write(root / "b.tex", "\\label{thm:same}")
        assert blueprint_check.validate_unique_labels([first, second])
        first.write_text("\\label{thm:kept}\\inputleannode{thm:kept}")
        errors = blueprint_check.validate_node_references(
            [first], _manifest([_claim()])
        )
        assert any("also authored" in error for error in errors)


def test_authored_verification_and_retired_bindings_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td) / "content.tex", "\\lean{Example.kept}\\leanok")
        errors = blueprint_check.validate_authored_markers([path])
        assert len(errors) == 2
        path.write_text("\\ppClaimBinding{thm:kept}")
        assert blueprint_check.validate_authored_markers([path])
        path.write_text("% A comment mentioning \\leanok is harmless.\n")
        assert blueprint_check.validate_authored_markers([path]) == []


def test_native_inputs_are_complete_unique_and_approved() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td) / "content.tex", "\\inputleannode{thm:unknown}")
        manifest = _manifest([_claim()])
        errors = blueprint_check.validate_node_references([path], manifest)
        assert any("unknown native" in error for error in errors)
        assert any("found 0" in error for error in errors)
        path.write_text("\\inputleannode{thm:kept}" * 2)
        assert any(
            "found 2" in error
            for error in blueprint_check.validate_node_references([path], manifest)
        )


def test_unknown_informal_references_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write(
            Path(td) / "content.tex",
            "\\label{def:informal}\\inputleannode{thm:kept}"
            "\\uses{def:informal, thm:kept}\\Cref{def:missing}",
        )
        errors = blueprint_check.validate_node_references([path], _manifest([_claim()]))
        assert len(errors) == 1 and "def:missing" in errors[0]
        path.write_text(
            path.read_text().replace("\\Cref{def:missing}", "% \\Cref{def:missing}")
        )
        assert (
            blueprint_check.validate_node_references([path], _manifest([_claim()]))
            == []
        )


def test_unreachable_or_missing_authored_inputs_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td))
        source = fixture["source"]
        content = source / "content.tex"
        _write(source / "unused.tex", content.read_text())
        content.write_text("An orphan file must not supply the rendered nodes.\n")
        assert any("found 0" in error for error in _validate(fixture))
        content.write_text("\\input{missing_chapter}\n")
        assert any(
            "missing authored Blueprint input" in error for error in _validate(fixture)
        )


def test_native_exports_match_declarations_and_statement_text() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture = _fixture(root)
        assert _validate(fixture, native=True) == []
        path = fixture["native"] / "module/Example.json"
        original = json.loads(path.read_text())
        for field, value, message in (
            ("name", "Example.retargeted", "native declaration differs"),
            ("statement", {"text": "A weakened claim."}, "native statement differs"),
            ("latexLabel", "thm:unclaimed", "native publication claim mismatch"),
        ):
            mutated = json.loads(json.dumps(original))
            mutated[0]["data"][field] = value
            path.write_text(json.dumps(mutated))
            assert any(message in error for error in _validate(fixture, native=True))
        path.write_text(json.dumps(original * 2))
        assert any(
            "duplicate native claim label" in error
            for error in _validate(fixture, native=True)
        )
        path.unlink()
        assert any(
            "cannot read native" in error for error in _validate(fixture, native=True)
        )
        # A cached publication-boundary check needs neither native exports nor Lake.
        assert _validate(fixture) == []


def test_unknown_native_dependency_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td))
        path = fixture["native"] / "module/Example.json"
        nodes = json.loads(path.read_text())
        nodes[0]["data"]["proof"]["usesLabels"] = ["thm:not-exported"]
        path.write_text(json.dumps(nodes))
        assert any(
            "unknown native Blueprint dependency" in error
            for error in _validate(fixture, native=True)
        )
        nodes[0]["data"]["proof"]["usesLabels"] = []
        path.write_text(json.dumps(nodes))
        artifact = fixture["native"] / "module/Example.artifacts/thm:kept.tex"
        original = artifact.read_text()
        artifact.write_text(original + "\\uses{thm:not-exported}\n")
        assert any(
            "unknown inferred native Blueprint reference" in error
            for error in _validate(fixture, native=True)
        )
        artifact.write_text(original.replace("Example.kept", "Example.retargeted"))
        assert any(
            "native LaTeX declaration differs" in error
            for error in _validate(fixture, native=True)
        )
        artifact.unlink()
        assert any(
            "cannot read native LeanArchitect LaTeX" in error
            for error in _validate(fixture, native=True)
        )


def test_withdrawn_or_retargeted_manuscript_label_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td), dedicated=True)
        assert _validate(fixture) == []
        prose = fixture["chapters"] / "06_paper/src/chapters/model.tex"
        prose.write_text("The label was withdrawn.\n")
        assert any(
            "thm:dedicated" in error and "no longer labels" in error
            for error in _validate(fixture)
        )
        record = fixture["records"]["06_paper"]
        record["claims"][0]["paper"] = "03_other"
        errors = blueprint_check.validate_claim_labels_exist(
            fixture["chapters"], record
        )
        assert any("03_other" in error for error in errors)


def test_missing_and_mutated_status_projections_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td))
        path = fixture["chapters"] / "01_paper/src/generated/lean_status.tex"
        original = path.read_text()
        mutations = (
            original.replace("0" * 64, "9" * 64),
            original.replace("{checked}", "{failed}"),
            original.replace(
                "Checked under the stated assumptions.", "Unconditional proof."
            ),
            original.replace("\\item \\Cref{thm:kept}", "\\item \\Cref{thm:wrong}"),
            original.replace(
                "\\newcommand{\\ppLeanStatusList}", "\\newcommand{\\hiddenStatus}"
            ),
        )
        for text in mutations:
            assert text != original
            path.write_text(text)
            assert any("stale or hand-edited" in error for error in _validate(fixture))
        path.unlink()
        assert any("missing generated status" in error for error in _validate(fixture))


def test_dedicated_manifests_keep_individual_status_hashes() -> None:
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td), dedicated=True)
        assert _validate(fixture, native=True) == []
        path = fixture["chapters"] / "06_paper/src/generated/lean_status.tex"
        path.write_text(path.read_text().replace("1" * 64, "0" * 64))
        assert any(
            "06_paper" in error and "stale or hand-edited" in error
            for error in _validate(fixture)
        )
        dedicated = fixture["manifest"].parent / "paper6-publication.json"
        dedicated.unlink()
        assert any(
            "cannot select publication manifests" in error
            for error in _validate(fixture)
        )
        fixture["manifest"].unlink()
        assert any(
            "cannot select publication manifests" in error
            for error in _validate(fixture)
        )


def test_retired_status_and_explicit_publication_scope() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for paper in ("01_paper", "99_unpublished"):
            _write(
                root / paper / "src/chapters/model.tex", "\\inputleannode{thm:leak}\n"
            )
            _write(root / paper / "src/generated/lean_status.tex", "withdrawn claim\n")
        scope = {"01_paper"}
        errors = blueprint_check.validate_publication_boundary(root, [], scope)
        assert len(errors) == 1 and "01_paper" in errors[0]
        errors = blueprint_check.validate_status_projections_are_current(
            root, {}, _manifest(), scope
        )
        assert len(errors) == 1 and "01_paper" in errors[0]
        assert len(blueprint_check.validate_publication_boundary(root, [])) == 2


def test_publication_declaration_and_source_leaks_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td) / "01_paper/src/chapters/results.tex", "Example.kept")
        errors = blueprint_check.validate_publication_boundary(
            Path(td), ["Example.kept"]
        )
        assert any("manifest declaration" in error for error in errors)
        path.write_text("See src/lean/packages/Example.lean")
        assert any(
            "Lean source path" in error
            for error in blueprint_check.validate_publication_boundary(Path(td), [])
        )


def test_current_tree_passes() -> None:
    errors = blueprint_check.validate(
        blueprint_check.MANIFEST_PATH,
        blueprint_check.SOURCE_DIR,
        None,
        blueprint_check.CHAPTERS_DIR,
    )
    assert errors == [], "\n".join(errors)


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("\nALL BLUEPRINT TESTS PASSED")
