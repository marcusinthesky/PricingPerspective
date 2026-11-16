"""Generated Paper 5 Matryoshka W1/W2 mechanism table."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_DIMENSIONS = (4096, 1024, 256, 64)


class Paper5TransportDimensionTableError(ValueError):
    """Raised when the dimension-ladder artifact cannot support the table."""


def _number(cell: dict[str, object], key: str) -> float:
    """Read one finite numeric cell."""
    value = cell.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Paper 5 transport-dimension table requires numeric {key!r}"
        raise Paper5TransportDimensionTableError(message)
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        message = f"Paper 5 transport-dimension table requires finite {key!r}"
        raise Paper5TransportDimensionTableError(message)
    return result


def _validated_cells(payload: object) -> list[dict[str, object]]:
    """Validate the native-scale, truncate-then-normalize table contract."""
    if not isinstance(payload, dict):
        message = "Paper 5 transport-dimension result must be a mapping"
        raise Paper5TransportDimensionTableError(message)
    result = cast("dict[str, object]", payload)
    if result.get("analysis_id") != "paper5-matryoshka-w1-w2-mechanism-v1":
        message = "Paper 5 transport-dimension result has an unknown analysis ID"
        raise Paper5TransportDimensionTableError(message)
    design = result.get("design")
    if not isinstance(design, dict) or design.get("distance_rescaling") != "none":
        message = "Paper 5 transport-dimension table forbids distance rescaling"
        raise Paper5TransportDimensionTableError(message)
    raw_cells = result.get("cells")
    if not isinstance(raw_cells, list) or not all(
        isinstance(cell, dict) for cell in raw_cells
    ):
        message = "Paper 5 transport-dimension result must contain mapping cells"
        raise Paper5TransportDimensionTableError(message)
    cells = cast("list[dict[str, object]]", raw_cells)
    dimensions = tuple(int(_number(cell, "effective_dimension")) for cell in cells)
    if dimensions != _EXPECTED_DIMENSIONS:
        message = "Paper 5 transport-dimension cells are not 4096/1024/256/64"
        raise Paper5TransportDimensionTableError(message)
    for cell in cells:
        if (
            cell.get("representation_transform") != "truncate_then_l2_normalize_rows"
            or cell.get("ground_metric") != "euclidean_chord"
            or cell.get("distance_rescaling") != "none"
        ):
            message = "Paper 5 transport-dimension cell violates its geometry contract"
            raise Paper5TransportDimensionTableError(message)
    return cells


def _table_row(cell: dict[str, object]) -> str:
    """Render one dimension's stage-by-stage W1/W2 comparison."""
    dimension = int(_number(cell, "effective_dimension"))
    return (
        " & ".join(
            [
                f"{dimension:,}",
                f"{100.0 * _number(cell, 'chord_distance_cv'):.2f}\\%",
                f"{100.0 * _number(cell, 'assignment_overlap'):.1f}\\%",
                f"{_number(cell, 'off_diagonal_weight_correlation'):.4f}",
                f"{_number(cell, 'spatial_lag_correlation'):.4f}",
                f"{_number(cell, 'held_out_score_w2_minus_w1'):.6f}",
            ]
        )
        + r" \\"
    )


def render_paper5_transport_dimension_table(
    results_path: Path,
    output_path: Path,
) -> None:
    """Render the native-scale Matryoshka mechanism ladder as LaTeX."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cells = _validated_cells(payload)
    distance_count = int(_number(cells[0], "article_distance_count"))
    pair_count = int(_number(cells[0], "firm_pair_count"))
    atom_count = int(_number(cells[0], "assignment_atom_count"))
    rows = [_table_row(cell) for cell in cells]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.2pt}",
                r"\begin{tabular}{rrrrrr}",
                r"\toprule",
                (
                    r"Dim. & Chord-distance CV & Assignment overlap & Weight corr. "
                    r"& Spatial-lag corr. & $\Delta$ held-out score \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Matryoshka dimension ladder for target-anchored W1 "
                    r"and W2 transport. At each dimension, leading coordinates are "
                    r"truncated and then row-wise L2-normalized; the ground cost is "
                    r"unscaled Euclidean chord distance. Chord-distance CV uses all "
                    f"{distance_count:,} article-pair costs across {pair_count:,} "
                    r"unordered firm pairs. Assignment overlap is the share of the "
                    f"{atom_count:,} matched target atoms receiving the same peer atom "
                    r"under minimum-sum $d$ and $d^2$ assignments. Weight "
                    r"correlation uses off-diagonal target-operator entries; "
                    r"spatial-lag correlation "
                    r"uses the common 2023--2026 excess-return panel. The final column "
                    r"is W2 minus W1 mean Gaussian quasi-log score per firm-day after "
                    r"fitting the FF5-plus-momentum model on 2018--2022; positive "
                    r"values favour W2.}"
                ),
                r"\label{tab:transport-dimension-ladder}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = [
    "Paper5TransportDimensionTableError",
    "render_paper5_transport_dimension_table",
]
