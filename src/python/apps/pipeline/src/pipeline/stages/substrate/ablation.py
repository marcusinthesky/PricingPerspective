"""Self-contained W2 ablation harness over (embedding model × Matryoshka dim).

Iterates the ``ablation.grid`` defined in ``params.yaml``. For each present
(model, dim) cell it:

1. reuses the typed exact W2 artifact for each ``matryoshka_dim``;
2. computes the Mantel statistic ``r_M`` (and a bootstrap CI over the pairwise
   correlation) between the W2 distance matrix and the return-distance matrix
   derived from the governed covariance artifact.

It additionally emits, once per model at full width, a ``baselines`` block: the
Mantel statistic for the **first-moment baseline** distance
``||mu_i - mu_j||`` between L2-normalised mean-pooled embeddings, together with
a *paired* firm-bootstrap interval for its gap to the W2 statistic.
This answers the standing question of what the measure-valued treatment buys
over mean-pooling, and it uses the Euclidean (metric, strong-negative-type)
form rather than the ``1 - cos`` dissimilarity, which fails the triangle
inequality and would be an inadmissible competitor.

Cells whose embeddings are not yet on disk are SKIPPED and logged as pending.
When the covariance matrix is unavailable the W2 cells are still recorded,
but the Mantel columns are left null and the cell is flagged as
``mantel_unavailable`` so the summary re-populates once covariance is restored.

This harness never reads or writes the canonical ``data/shared/energy_tests/`` or
``data/shared/mantel_tests/`` paths — everything lives under ``data/shared/ablations/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import duckdb
import numpy as np
import pandas as pd
import typer
import yaml
from jcor.association.mantel import (
    mantel_bootstrap_ci,
    mantel_test,
    paired_mantel_bootstrap,
)

from pipeline._kernels.arrays import (
    InvalidNormalizationRowsError,
    l2_normalize_rows_then_mean,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact
from pipeline.stages.substrate.mantel import _align_matrices, _load_covariance_matrix

if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)
MIN_EMBEDDING_FILES = 2
ABLATION_DISTANCE_ID = "wasserstein_w2"


@dataclass(frozen=True)
class ResamplingOptions:
    """Mantel and bootstrap controls shared by every ablation comparison."""

    method: Literal["pearson", "spearman"] = "pearson"
    permutations: int = 10_000
    bootstrap_iters: int = 1_000
    seed: int = 42


@dataclass(frozen=True)
class AblationOptions:
    """Numerical and reuse policy for an ablation-grid execution."""

    resampling: ResamplingOptions = ResamplingOptions()
    typed_distance_root: Path = Path("data/shared/typed_distances")


DEFAULT_ABLATION_OPTIONS = AblationOptions()


def _dim_label(dim: int | None) -> str:
    """Directory-safe label for a Matryoshka dim (``None`` -> ``full``)."""
    return "full" if dim is None else str(dim)


def _bootstrap_node_indices(n_nodes: int, options: ResamplingOptions) -> np.ndarray:
    """Preserve the historical NumPy node-resample schedule at the artifact edge."""
    rng = np.random.default_rng(options.seed)
    indices = np.empty((options.bootstrap_iters, n_nodes), dtype=np.intp)
    for bootstrap_index in range(options.bootstrap_iters):
        indices[bootstrap_index] = rng.integers(0, n_nodes, size=n_nodes)
    return indices


def _load_typed_metric(
    distance_artifact: Path,
    *,
    distance_id: str = ABLATION_DISTANCE_ID,
) -> tuple[list[str], np.ndarray]:
    """Load one validated typed distance matrix in its native metric scale."""
    frame, summary = read_typed_distance_artifact(
        distance_artifact,
        expected_identity={"distance_id": distance_id},
    )
    item_ids = summary["item_ids"]
    if not isinstance(item_ids, list):
        message = "typed distance summary item_ids must be a list"
        raise TypeError(message)
    tickers = [str(item) for item in item_ids]
    values = frame["value"].to_numpy(dtype=np.float64)
    return tickers, values.reshape(len(tickers), len(tickers))


def _mantel_for_cell(
    w2_artifact: Path,
    covariance_matrix: Path,
    options: ResamplingOptions,
) -> dict[str, Any]:
    """Compute Mantel r_M, p-value, bootstrap CI and n_pairs for one cell."""
    w2_tickers, w2_dist = _load_typed_metric(w2_artifact)
    cov_tickers, cov_corr = _load_covariance_matrix(covariance_matrix, "correlation")
    _common, aligned_w2, aligned_corr = _align_matrices(
        w2_tickers, w2_dist, cov_tickers, cov_corr
    )
    aligned_cov_dist = np.sqrt(2 * (1 - aligned_corr))

    result = mantel_test(
        aligned_w2,
        aligned_cov_dist,
        method=options.method,
        num_permutations=options.permutations,
    )
    r_m, p_value = result.correlation, result.pvalue
    bootstrap_indices = _bootstrap_node_indices(
        aligned_w2.shape[0],
        options,
    )
    ci_low, ci_high = mantel_bootstrap_ci(
        aligned_w2,
        aligned_cov_dist,
        bootstrap_indices,
        method=options.method,
    )
    n = aligned_w2.shape[0]
    n_pairs = n * (n - 1) // 2
    return {
        "r_M": float(r_m),
        "mantel_p": float(p_value),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "n_pairs": int(n_pairs),
        "distance_id": ABLATION_DISTANCE_ID,
    }


def _mean_embedding_vectors(
    embeddings_dir: Path, tickers: list[str]
) -> dict[str, np.ndarray] | None:
    """Per-firm L2-normalised mean-pooled embedding, the first-moment summary.

    Mirrors the normalisation of ``_embedding_moments`` in the paper-1 dyadic
    confound stage exactly — row-wise L2 normalisation first, then mean-pool,
    then renormalise the mean — so the baseline compares against the same
    first-moment object the published model ladder already controls for. The
    computation is duplicated rather than imported because that module sits in
    the papers layer and this one is substrate; importing upward would invert
    the dependency direction the import fences enforce.

    Returns ``None`` if any ticker's embedding file is missing or empty, so the
    caller can omit the baseline rather than silently comparing against zeros.
    """
    if not embeddings_dir.exists():
        return None
    con = duckdb.connect(":memory:")
    vectors: dict[str, np.ndarray] = {}
    for ticker in tickers:
        path = embeddings_dir / f"{ticker}.parquet"
        if not path.exists():
            con.close()
            return None
        emb_df = con.execute("SELECT embedding FROM read_parquet(?)", [str(path)]).df()
        if emb_df.empty:
            con.close()
            return None
        stacked = np.stack(list(emb_df["embedding"])).astype(np.float64, copy=False)
        try:
            _, prototype = l2_normalize_rows_then_mean(stacked)
        except InvalidNormalizationRowsError:
            con.close()
            return None
        vectors[ticker] = prototype
    con.close()
    return vectors


def _baseline_for_model(
    embeddings_dir: Path,
    w2_artifact: Path,
    covariance_matrix: Path,
    options: ResamplingOptions,
) -> dict[str, Any] | None:
    """Mantel statistic for the mean-embedding distance, and its gap to W2.

    The baseline distance is ``||mu_i - mu_j||`` between L2-normalised mean
    embeddings — a **true metric on the mean-vector summaries**, and of strong
    negative type because it is Euclidean. It is an empirical comparator, not
    a separating divergence on the underlying article distributions: distinct
    distributions can share a mean vector. This is deliberately not the
    ``1 - cos`` dissimilarity that the dyadic ladder uses as a regressor:
    ``1 - cos`` fails the triangle inequality and is not even a pseudometric,
    so reporting it as *the* baseline would compare against an inadmissible
    competitor. The two are monotone transforms of one another, so any rank
    statistic is identical between them and only the Pearson form differs.
    """
    w2_tickers, w2_dist = _load_typed_metric(w2_artifact)
    cov_tickers, cov_corr = _load_covariance_matrix(covariance_matrix, "correlation")
    common, aligned_w2, aligned_corr = _align_matrices(
        w2_tickers, w2_dist, cov_tickers, cov_corr
    )
    aligned_cov_dist = np.sqrt(2 * (1 - aligned_corr))

    vectors = _mean_embedding_vectors(embeddings_dir, list(common))
    if vectors is None:
        return None
    matrix = np.stack([vectors[t] for t in common])
    gram = matrix @ matrix.T
    baseline = np.sqrt(np.maximum(2.0 - 2.0 * gram, 0.0))
    np.fill_diagonal(baseline, 0.0)

    result = mantel_test(
        baseline,
        aligned_cov_dist,
        method=options.method,
        num_permutations=options.permutations,
    )
    bootstrap_indices = _bootstrap_node_indices(baseline.shape[0], options)
    ci_low, ci_high = mantel_bootstrap_ci(
        baseline,
        aligned_cov_dist,
        bootstrap_indices,
        method=options.method,
    )
    gap = paired_mantel_bootstrap(
        aligned_w2,
        baseline,
        aligned_cov_dist,
        bootstrap_indices,
        method=options.method,
    )
    n = baseline.shape[0]
    return {
        "distance": "mean_embedding_euclidean",
        "distance_id": ABLATION_DISTANCE_ID,
        "r_M": float(result.correlation),
        "mantel_p": float(result.pvalue),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "n_pairs": int(n * (n - 1) // 2),
        **gap._asdict(),
    }


def run_ablation(
    embeddings_root: Path,
    covariance_matrix: Path,
    output_dir: Path,
    params_file: Path,
    options: AblationOptions = DEFAULT_ABLATION_OPTIONS,
) -> None:
    """Run the (model × Matryoshka dim) ablation grid from ``params_file``.

    Reuses per-cell typed W2 artifacts under ``typed_distance_root`` and writes a
    consolidated ``summary.parquet`` + ``summary.yaml`` with columns
    ``[model, dim, native_dim, r_M, mantel_p, bootstrap_ci_low,
    bootstrap_ci_high, n_pairs]``. Cells with no on-disk embeddings are marked
    ``pending``; cells with no covariance matrix get null Mantel columns.
    """
    with params_file.open(encoding="utf-8") as fh:
        params = yaml.safe_load(fh)

    grid: dict[str, list[int | None]] = params["ablation"]["grid"]
    model_meta: dict[str, Any] = params.get("embedding", {}).get("models", {})
    representations: dict[str, dict[str, Any]] = params["typed_analysis"][
        "representations"
    ]
    cov_available = covariance_matrix.exists()
    if not cov_available:
        typer.echo(
            f"Warning: covariance matrix not found at {covariance_matrix}; "
            "Mantel columns will be null (W2 cells still recorded).",
            err=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []

    for model, dims in grid.items():
        model_dir = embeddings_root / model
        native_dim = model_meta.get(model, {}).get("dims")
        for dim in dims:
            label = _dim_label(dim)
            cell = {
                "model": model,
                "dim": label,
                "native_dim": native_dim,
                "distance_id": ABLATION_DISTANCE_ID,
                "r_M": None,
                "mantel_p": None,
                "bootstrap_ci_low": None,
                "bootstrap_ci_high": None,
                "n_pairs": None,
                "status": "pending",
            }

            parquet_files: list[Path] = (
                list(model_dir.glob("*.parquet")) if model_dir.exists() else []
            )
            if len(parquet_files) < MIN_EMBEDDING_FILES:
                typer.echo(f"[pending] {model} dim={label}: embeddings not on disk")
                rows.append(cell)
                continue

            matching = [
                spec["representation_id"]
                for spec in representations.values()
                if spec["provider_id"] == model and spec.get("dimension") == dim
            ]
            if len(matching) != 1:
                msg = (
                    f"expected one representation for {model} dim={label}, "
                    f"got {matching}"
                )
                raise ValueError(msg)
            cell_out = (
                options.typed_distance_root / model / matching[0] / ABLATION_DISTANCE_ID
            )
            if not cell_out.exists():
                typer.echo(
                    f"[pending] {model} dim={label}: {cell_out} not materialized"
                )
                rows.append(cell)
                continue

            if cov_available:
                mantel = _mantel_for_cell(
                    cell_out,
                    covariance_matrix,
                    options.resampling,
                )
                cell.update(mantel)
                cell["status"] = "complete"
                typer.echo(f"[reuse] {model} dim={label}: r_M={mantel['r_M']:.6f}")

                # The first-moment baseline is a property of the model, not of a
                # truncation width, so compute it once per model at full width.
                if dim is None:
                    baseline = _baseline_for_model(
                        model_dir,
                        cell_out,
                        covariance_matrix,
                        options.resampling,
                    )
                    if baseline is None:
                        typer.echo(
                            f"[baseline] {model}: embeddings unavailable, skipped"
                        )
                    else:
                        baseline["model"] = model
                        baseline["native_dim"] = native_dim
                        baselines.append(baseline)
                        typer.echo(
                            f"[baseline] {model}: r_M={baseline['r_M']:.6f} "
                            f"vs W2 {mantel['r_M']:.6f} "
                            f"(gap {baseline['gap_mean']:+.6f}, "
                            f"95% CI [{baseline['gap_ci_low']:.6f}, "
                            f"{baseline['gap_ci_high']:.6f}])"
                        )
            else:
                cell["status"] = "mantel_unavailable"

            rows.append(cell)

    _write_ablation_summary(output_dir, rows, baselines)


def _write_ablation_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
) -> None:
    """Persist the consolidated tabular and YAML ablation summaries."""
    columns = [
        "model",
        "dim",
        "native_dim",
        "distance_id",
        "r_M",
        "mantel_p",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "n_pairs",
        "status",
    ]
    summary_df = cast("pd.DataFrame", pd.DataFrame(rows)[columns])
    summary_df.to_parquet(output_dir / "summary.parquet", index=False)

    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "distance_id": ABLATION_DISTANCE_ID,
                "cells": summary_df.to_dict(orient="records"),
                "baselines": baselines,
            },
            fh,
            sort_keys=False,
        )

    n_complete = int((summary_df["status"] == "complete").sum())
    n_pending = int((summary_df["status"] == "pending").sum())
    typer.echo(
        f"\nAblation complete: {len(rows)} cells "
        f"({n_complete} complete, {n_pending} pending). "
        f"Summary -> {output_dir / 'summary.yaml'}"
    )
