"""Saved lexical and pretraining comparisons, independent of TeX publication."""

from __future__ import annotations

import json
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sklearn
import typer
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from statadic.geometry.discrepancy.transport.exact import (
    balanced_wasserstein_from_cost_jax,
)
from statadic.inference.dyadic import DyadicFitRequest, DyadicInference
from statadic.law.dyadic import DyadicModel, DyadicSample

from pipeline._kernels.typed_distances import _exact_wasserstein_pair_jax
from pipeline.io.contracts import write_manifest
from pipeline.io.params import load_params
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.stages.papers.paper1._dyadic.adapters import legacy_node_count_schedule
from pipeline.stages.papers.paper1._dyadic.estimation import _fit_spec
from pipeline.stages.papers.paper1.representation_sensitivity import _node_counts
from pipeline.stages.substrate.chronobert import (
    _file_identity,
    _json_identity,
    _read_article_sample,
)

_cost_w2 = jax.jit(balanced_wasserstein_from_cost_jax)
_SAMPLE_ROWS = 128

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(
    output_dir: Path, command: str, inputs: list[Path], metadata: dict[str, Any]
) -> None:
    identities = [_file_identity(path) for path in inputs]
    write_manifest(
        output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": command.replace("paper1-", "p1_").replace("-", "_"),
            "lane": "pipeline",
            "phase": "compute",
            "protocol_id": "paper1.text_controls.v1",
            "code_identity": {
                "kind": "sha256",
                "value": entrypoint_code_identity(command),
            },
            "parameter_identity": {"kind": "sha256", "value": _json_identity(metadata)},
            "seed_policy": "saved-common-node-schedule"
            if command == "paper1-text-controls"
            else "deterministic",
            "upstream_artifacts": [
                {"kind": "sha256", "value": item["sha256"], "path": item["path"]}
                for item in identities
            ],
            "environment": {
                "sklearn_version": sklearn.__version__,
                "jax_version": jax.__version__,
            },
            "outputs": [
                {**_file_identity(path), "kind": path.suffix.removeprefix(".")}
                for path in sorted(output_dir.iterdir())
                if path.name != "provenance.manifest.json"
            ],
            "risk_class": "moderate",
            "metadata": metadata,
        },
    )


def tfidf_pair_distances(
    firm_vectors: csr_matrix,
    article_vectors: csr_matrix,
    boundaries: np.ndarray,
    tickers: list[str],
    *,
    max_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the original distances from the two published sparse TF-IDF fits."""
    if (
        firm_vectors.shape[0] != len(tickers)
        or len(boundaries) != len(tickers) + 1
        or boundaries[0] != 0
        or boundaries[-1] != article_vectors.shape[0]
        or np.any(np.diff(boundaries) <= 0)
        or not np.isfinite(firm_vectors.data).all()
        or not np.isfinite(article_vectors.data).all()
    ):
        message = "numeric TF-IDF inputs do not match the declared cohort"
        raise ValueError(message)
    cosine = cosine_distances(firm_vectors)
    records = []
    for left, right in combinations(range(len(tickers)), 2):
        a = article_vectors[boundaries[left] : boundaries[left + 1]]
        b = article_vectors[boundaries[right] : boundaries[right + 1]]
        costs = euclidean_distances(csr_matrix(a), csr_matrix(b))
        w2 = float(_cost_w2(jnp.asarray(costs, dtype=jnp.float64), 2.0))
        for arm, value in (
            ("tfidf_firm_cosine", cosine[left, right]),
            ("tfidf_document_w2", w2),
        ):
            records.append(
                {
                    "arm": arm,
                    "ticker_i": tickers[left],
                    "ticker_j": tickers[right],
                    "distance": float(value),
                }
            )
    return pd.DataFrame(records), {
        "vectorizer": "TfidfVectorizer(max_features=max_features)",
        "max_features": max_features,
        "fit_units": ["concatenated_firm", "article"],
        "firm_vocabulary_size": firm_vectors.shape[1],
        "article_vocabulary_size": article_vectors.shape[1],
        "zero_firm_rows": int(np.count_nonzero(firm_vectors.getnnz(axis=1) == 0)),
        "zero_article_rows": int(np.count_nonzero(article_vectors.getnnz(axis=1) == 0)),
        "w2": (
            "sqrt(minimum mean squared Euclidean assignment cost), "
            "equal article weights"
        ),
    }


def run_tfidf_controls(
    *, numeric_inputs: Path, params_file: Path, output_dir: Path
) -> None:
    """Use the published, vocabulary-free sparse vectors for the frozen cohort."""
    params = load_params(params_file)
    tickers = sorted(params["symbols"])
    design_path = numeric_inputs / "design.json"
    design = json.loads(design_path.read_text())
    settings = params["paper1_text_controls"]
    if design["ticker_order"] != tickers or any(
        design[key] != settings[key] for key in ("max_chars", "max_features")
    ):
        message = "numeric TF-IDF design differs from the configured study"
        raise ValueError(message)
    inputs = [
        design_path,
        numeric_inputs / "firm_vectors.npz",
        numeric_inputs / "document_vectors.npz",
    ]
    pairs, metadata = tfidf_pair_distances(
        csr_matrix(load_npz(inputs[1])),
        csr_matrix(load_npz(inputs[2])),
        np.asarray(design["boundaries"], dtype=np.int64),
        tickers,
        max_features=settings["max_features"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    _manifest(
        output_dir,
        "paper1-tfidf-controls",
        inputs,
        {
            **metadata,
            "max_chars": settings["max_chars"],
            "ticker_order": tickers,
            "selection": "first 128 ascending url_hash",
            "article_hash_inventory_sha256": _json_identity(design["cohorts"]),
        },
    )


def run_pretraining_distances(
    *,
    embeddings_dir: Path,
    articles_dir: Path,
    model_key: str,
    params_file: Path,
    output_dir: Path,
) -> None:
    """Reuse the canonical normalized-embedding W2 kernel on each pinned model."""
    params = load_params(params_file)
    tickers = sorted(params["symbols"])
    manifest_file = embeddings_dir / "provenance.manifest.json"
    manifest = json.loads(manifest_file.read_text())
    checkpoint = params["paper1_text_controls"]["models"][model_key]
    if manifest["metadata"]["model"] != checkpoint:
        message = "embedding checkpoint identity differs from the requested control"
        raise ValueError(message)
    clouds: dict[str, jax.Array] = {}
    inputs = [manifest_file]
    for ticker in tickers:
        source = articles_dir / f"{ticker}.parquet"
        sample = _read_article_sample(source, ticker)
        path = embeddings_dir / f"{ticker}.parquet"
        frame = pd.read_parquet(path).sort_values("url_hash")
        if frame["url_hash"].tolist() != sample["url_hash"].to_pylist():
            message = (
                f"{ticker}: embeddings do not match the frozen 128-article selection"
            )
            raise ValueError(message)
        values = np.asarray(frame["embedding"].tolist(), dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (
            values.shape != (128, 1152)
            or not np.isfinite(norms).all()
            or (norms == 0).any()
        ):
            message = f"{ticker}: invalid encoder vectors"
            raise ValueError(message)
        clouds[ticker] = jnp.asarray(values / norms)
        inputs.extend([source, path])
    records = []
    for left, right in combinations(tickers, 2):
        value = float(_exact_wasserstein_pair_jax(clouds[left], clouds[right], 2.0))
        records.append(
            {
                "arm": checkpoint["id"],
                "ticker_i": left,
                "ticker_j": right,
                "distance": value,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_dir / "pairs.parquet", index=False)
    _manifest(
        output_dir,
        "paper1-pretraining-distances",
        inputs,
        {
            "model": checkpoint,
            "normalization": "article L2 once",
            "distance": "wasserstein_w2",
            "article_hash_inventory_sha256": manifest["metadata"][
                "article_hash_inventory_sha256"
            ],
        },
    )


def run_text_controls(
    *,
    pairs_file: Path,
    node_counts_file: Path,
    control_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Apply the existing estimator and supplied counts to supplementary controls."""
    pairs = pd.read_parquet(pairs_file)
    canonical = pairs.loc[
        (pairs["model"] == "qwen3-embedding-8b") & (pairs["m"] == _SAMPLE_ROWS),
        ["ticker_i", "ticker_j", "return_chord", "wasserstein_w2"],
    ].copy()
    canonical = canonical.sort_values(["ticker_i", "ticker_j"], ignore_index=True)
    keys = ["ticker_i", "ticker_j"]
    if canonical.duplicated(keys).any():
        message = "canonical dyads must be unique"
        raise ValueError(message)
    tickers = sorted(set(canonical["ticker_i"]) | set(canonical["ticker_j"]))
    counts = _node_counts(node_counts_file, tickers)
    controls = [
        canonical[[*keys, "wasserstein_w2"]]
        .rename(columns={"wasserstein_w2": "distance"})
        .assign(arm="canonical_qwen8b_w2")
    ]
    inputs = [pairs_file, node_counts_file]
    for directory in control_dirs:
        path = directory / "pairs.parquet"
        controls.append(pd.read_parquet(path))
        inputs.extend([path, directory / "provenance.manifest.json"])
    all_pairs = pd.concat(controls, ignore_index=True)
    cells: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []
    for index, (arm, raw_group) in enumerate(
        all_pairs.groupby("arm", sort=False), start=1
    ):
        group = raw_group.sort_values(keys, ignore_index=True)
        if (
            not group[keys].equals(canonical[keys])
            or not np.isfinite(group["distance"]).all()
        ):
            message = (
                f"{arm}: incomplete, duplicated or invalid canonical dyad coverage"
            )
            raise ValueError(message)
        x = group["distance"].to_numpy(dtype=np.float64)
        sample = DyadicSample[str](
            canonical["return_chord"].to_numpy(dtype=np.float64),
            canonical["ticker_i"].to_numpy(dtype=str).tolist(),
            canonical["ticker_j"].to_numpy(dtype=str).tolist(),
        )
        fit = _fit_spec(
            DyadicFitRequest(
                model=DyadicModel(
                    ["w2_distance"],
                    {"w2_distance": x},
                    sample,
                    node_effects=True,
                    effect_name_prefix="firm_fe",
                ),
                inference=DyadicInference[str](
                    bootstrap_iters=counts.shape[0],
                    bootstrap_seed=42,
                    node_schedule=legacy_node_count_schedule(sample, counts),
                ),
            ),
            spec_index=index,
            include_draws=True,
        )
        frame = cast("pd.DataFrame", fit["_bootstrap_draws"]).copy()
        sd = float(np.std(x, ddof=1))
        frame["standardized_effect"] = frame["beta_w2"] * sd
        frame["arm"] = arm
        valid = frame.loc[
            frame["solve_ok"] & np.isfinite(frame["standardized_effect"]),
            "standardized_effect",
        ]
        if valid.empty:
            message = f"{arm}: no valid node-bootstrap fits"
            raise ValueError(message)
        cells.append(
            {
                "arm": arm,
                "coefficient": fit["coef_w2_distance"],
                "standard_error": fit["se_w2_distance"],
                "ci_low": fit["ci_low_w2_distance"],
                "ci_high": fit["ci_high_w2_distance"],
                "pvalue": fit["pvalue_w2_distance"],
                "distance_sd": sd,
                "effect_one_sd": float(cast("float", fit["coef_w2_distance"])) * sd,
                "effect_one_sd_ci_low": float(np.percentile(valid, 2.5)),
                "effect_one_sd_ci_high": float(np.percentile(valid, 97.5)),
                "partial_r2": fit["within_r2"],
                "geometry_spearman_vs_canonical_w2": float(
                    spearmanr(x, canonical["wasserstein_w2"]).statistic
                ),
                "bootstrap_valid_draws": len(valid),
            }
        )
        draws.append(frame.rename(columns={"beta_w2": "coefficient"}))
        typer.echo(f"Fitted comparison arm: {arm}")
    result = {
        "schema_version": 1,
        "publication_status": "supplementary",
        "return_window": "2018-2022",
        "n_firms": len(tickers),
        "n_dyads": len(canonical),
        "bootstrap_requested_draws": counts.shape[0],
        "common_node_schedule": True,
        "model": "return_chord ~ distance + symmetric additive firm effects",
        "cells": cells,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    all_pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    pd.concat(draws, ignore_index=True).to_parquet(
        output_dir / "bootstrap_draws.parquet", index=False
    )
    lines = [
        "# Text controls — supplementary comparisons",
        "",
        "2018-2022 returns; common node-bootstrap schedule; symmetric firm effects.",
        "",
        (
            "| Arm | Coefficient [95% CI] | One-SD effect [95% CI] | "
            "Partial R² | Rank vs W₂ |"
        ),
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['arm']} | {row['coefficient']:.4f} "
        f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
        f"{row['effect_one_sd']:.4f} [{row['effect_one_sd_ci_low']:.4f}, "
        f"{row['effect_one_sd_ci_high']:.4f}] | "
        f"{row['partial_r2']:.4f} | "
        f"{row['geometry_spearman_vs_canonical_w2']:.4f} |"
        for row in cells
    )
    (output_dir / "review.md").write_text("\n".join(lines) + "\n")
    _manifest(
        output_dir,
        "paper1-text-controls",
        inputs,
        {"publication_status": "supplementary", "common_node_schedule": True},
    )
    return result
