"""Paper 1 publication-surface guard for the theory/methods lane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import yaml

# Token-level mean pooling is part of article encoding, independent of the
# retired firm-level mean-embedding comparator.
_FORBIDDEN = re.compile(
    r"linear[_ -]?mmd|mean[- ]embedding|centroid|"
    r"energy-beyond-mean|confound-spec8|w2_vs_return_and_mean_metric|model_grid|"
    r"mantel",
    flags=re.IGNORECASE,
)


def _repository_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").exists()
    )


def test_paper1_publication_surface_excludes_retired_mean_branch() -> None:
    """Keep retired comparator code, artifacts, and prose out of Paper 1."""
    root = _repository_root()
    source_roots = (
        root / "src/latex/projects/01_continuous_bounds/src/chapters",
        root / "src/latex/projects/01_continuous_bounds/src/generated",
    )
    violations: list[str] = []
    for source_root in source_roots:
        for path in source_root.rglob("*"):
            if path.suffix not in {".py", ".tex", ".json", ".yaml"}:
                continue
            match = _FORBIDDEN.search(path.read_text(encoding="utf-8"))
            if match is not None:
                violations.append(f"{path.relative_to(root)}: {match.group(0)}")
    assert violations == []

    stages = cast(
        "dict[str, dict[str, object]]",
        yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))["stages"],
    )
    paper1_dvc = "\n".join(
        yaml.safe_dump(stage, sort_keys=True)
        for name, stage in stages.items()
        if name.startswith("p1_")
    )
    assert _FORBIDDEN.search(paper1_dvc) is None
    assert not (
        root / "src/latex/projects/01_continuous_bounds/src/images/"
        "w2_vs_return_and_mean_metric.pgf"
    ).exists()
    assert not (
        root / "src/latex/projects/01_continuous_bounds/src/generated/model_grid.tex"
    ).exists()


def test_shared_linear_mmd_surface_remains_available() -> None:
    """The Paper 1 cut must not remove the shared linear-MMD comparator."""
    root = _repository_root()
    registry = yaml.safe_load((root / "params.yaml").read_text(encoding="utf-8"))[
        "typed_analysis"
    ]
    assert registry["distances"]["linear_mmd"]["family"] == "mmd"
