"""Governed anatomy of Paper 3's news-only allocation.

The stage turns the canonical zero-slack allocation into additive firm, pair,
sector, and expanding-information-vintage diagnostics.  It deliberately keeps
returns out of the computation: every output explains the decision induced by
the information geometry rather than re-evaluating its conventional variance.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Never

import numpy as np
import pandas as pd
import yaml

from pipeline._kernels.paper3_validation import solve_certificate
from pipeline.io.contracts import read_manifest, write_manifest
from pipeline.io.paper3_certificate import sha256_path
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from collections.abc import Mapping

_SIMPLEX_TOLERANCE = 1e-8
_IDENTITY_TOLERANCE = 5e-6
_FINAL_VINTAGE_TOLERANCE = 1e-8
_SUPPORT_TOLERANCE = 1e-8
_NONNEGATIVITY_TOLERANCE = 1e-12


class PortfolioAnatomyError(ValueError):
    """Raised when an anatomy input or additive identity fails closed."""


@dataclass(frozen=True, slots=True)
class Paper3PortfolioAnatomyPaths:
    """Declared inputs and output directory for the anatomy stage."""

    weights_path: Path
    distance_artifact_dir: Path
    pit_w2_path: Path
    pit_manifest_path: Path
    universe_csv: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class PortfolioAnatomyArtifacts:
    """Pure computation result before persistence."""

    firms: pd.DataFrame
    pairs: pd.DataFrame
    sectors: pd.DataFrame
    vintage_weights: pd.DataFrame
    vintage_summary: pd.DataFrame
    vintage_changes: pd.DataFrame
    summary: dict[str, object]


def _error(message: str) -> Never:
    raise PortfolioAnatomyError(message)


def _check_matrix(matrix: np.ndarray, n_firms: int, label: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (n_firms, n_firms) or not np.isfinite(values).all():
        _error(f"{label} must be a finite {n_firms} by {n_firms} matrix")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0.0):
        _error(f"{label} must be symmetric")
    if not np.allclose(np.diag(values), 0.0, atol=1e-12, rtol=0.0):
        _error(f"{label} must have a zero diagonal")
    if np.any(values < -_NONNEGATIVITY_TOLERANCE):
        _error(f"{label} must be nonnegative")
    return values


def _check_weights(weights: np.ndarray, n_firms: int, label: str) -> np.ndarray:
    q = np.asarray(weights, dtype=np.float64)
    if (
        q.shape != (n_firms,)
        or not np.isfinite(q).all()
        or np.any(q < -_NONNEGATIVITY_TOLERANCE)
    ):
        _error(f"{label} must be a finite nonnegative vector of length {n_firms}")
    if abs(float(q.sum()) - 1.0) > _SIMPLEX_TOLERANCE:
        _error(f"{label} must sum to one")
    return np.maximum(q, 0.0)


def _canonical_frames(
    metadata: pd.DataFrame, weights: np.ndarray, squared_w2: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build firm, pair, and sector decompositions for one optimum."""
    tickers = metadata["ticker"].astype(str).to_numpy()
    names = metadata["name"].astype(str).to_numpy()
    sectors = metadata["sector"].astype(str).to_numpy()
    n_firms = len(metadata)
    q = _check_weights(weights, n_firms, "canonical weights")
    matrix = _check_matrix(squared_w2, n_firms, "canonical squared W2")

    marginal = matrix @ q
    firm_credit = 0.5 * q * marginal
    credit = float(firm_credit.sum())
    if credit <= 0.0:
        _error("canonical certificate credit must be positive")
    firm_share = firm_credit / credit
    firm_share_gap = float(np.max(np.abs(firm_share - q)))
    if firm_share_gap > _IDENTITY_TOLERANCE:
        _error(
            "canonical weights fail the interior KKT firm-share identity: "
            f"maximum gap {firm_share_gap:.3e}"
        )

    pair_matrix = np.multiply.outer(q, q) * matrix
    upper_i, upper_j = np.triu_indices(n_firms, k=1)
    pair_credit = pair_matrix[upper_i, upper_j]
    pair_total = float(pair_credit.sum())
    if not np.isclose(pair_total, credit, atol=1e-10, rtol=1e-10):
        _error("unordered pair credits do not add to the certificate")
    pairs = pd.DataFrame(
        {
            "ticker_i": tickers[upper_i],
            "name_i": names[upper_i],
            "sector_i": sectors[upper_i],
            "ticker_j": tickers[upper_j],
            "name_j": names[upper_j],
            "sector_j": sectors[upper_j],
            "weight_i": q[upper_i],
            "weight_j": q[upper_j],
            "squared_w2": matrix[upper_i, upper_j],
            "certificate_credit": pair_credit,
            "certificate_share": pair_credit / credit,
            "same_sector": sectors[upper_i] == sectors[upper_j],
        }
    ).sort_values(
        ["certificate_credit", "ticker_i", "ticker_j"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    pairs.insert(0, "rank", np.arange(1, len(pairs) + 1, dtype=np.int64))

    strongest_index = np.argmax(pair_matrix, axis=1)
    active = q > _SUPPORT_TOLERANCE
    strongest_partner = np.where(active, tickers[strongest_index], "")
    strongest_share = np.where(
        active,
        pair_matrix[np.arange(n_firms), strongest_index] / credit,
        0.0,
    )
    firms = metadata.copy()
    firms["weight"] = q
    firms["active"] = active
    firms["marginal_squared_w2"] = marginal
    firms["certificate_credit"] = firm_credit
    firms["certificate_share"] = firm_share
    firms["kkt_share_gap"] = firm_share - q
    firms["strongest_partner"] = strongest_partner
    firms["strongest_pair_share"] = strongest_share

    sector_rows: list[dict[str, object]] = []
    for sector in sorted(set(sectors)):
        in_sector = sectors == sector
        within = float(
            pairs.loc[
                (pairs["sector_i"] == sector) & (pairs["sector_j"] == sector),
                "certificate_credit",
            ].sum()
        )
        cross_attribution = 0.5 * float(
            pairs.loc[
                (pairs["sector_i"] == sector) ^ (pairs["sector_j"] == sector),
                "certificate_credit",
            ].sum()
        )
        attribution = within + cross_attribution
        sector_rows.append(
            {
                "sector": sector,
                "n_firms": int(in_sector.sum()),
                "portfolio_weight": float(q[in_sector].sum()),
                "equal_weight_share": float(in_sector.mean()),
                "within_certificate_credit": within,
                "cross_certificate_attribution": cross_attribution,
                "certificate_credit_attribution": attribution,
                "certificate_share_attribution": attribution / credit,
            }
        )
    sector_frame = pd.DataFrame(sector_rows).sort_values(
        ["portfolio_weight", "sector"], ascending=[False, True], ignore_index=True
    )
    within_credit = float(pairs.loc[pairs["same_sector"], "certificate_credit"].sum())
    cross_credit = pair_total - within_credit
    sector_total = float(sector_frame["certificate_credit_attribution"].sum())
    if not np.isclose(sector_total, credit, atol=1e-10, rtol=1e-10):
        _error("sector-attributed credits do not add to the certificate")
    identities = {
        "certificate_credit": credit,
        "firm_credit_sum": float(firm_credit.sum()),
        "pair_credit_sum": pair_total,
        "sector_credit_sum": sector_total,
        "within_sector_credit": within_credit,
        "cross_sector_credit": cross_credit,
        "firm_share_max_abs_gap": firm_share_gap,
    }
    return firms, pairs, sector_frame, identities


def _vintage_frames(
    metadata: pd.DataFrame,
    canonical_weights: np.ndarray,
    vintages: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Solve the same news-only allocation at each expanding cutoff."""
    tickers = metadata["ticker"].astype(str).to_numpy()
    n_firms = len(tickers)
    weight_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    previous: np.ndarray | None = None
    previous_vintage: str | None = None
    final_weights: np.ndarray | None = None
    for vintage in sorted(vintages):
        matrix = _check_matrix(vintages[vintage], n_firms, f"PIT matrix {vintage}")
        solution = solve_certificate(matrix, cap=1.0)
        q = _check_weights(solution.weights, n_firms, f"PIT weights {vintage}")
        final_weights = q
        year = int(vintage[:4])
        turnover = None if previous is None else 0.5 * float(np.abs(q - previous).sum())
        maximum = int(np.argmax(q))
        summary_rows.append(
            {
                "vintage": vintage,
                "year": year,
                "certificate_credit": float(solution.credit),
                "effective_number_assets": float(1.0 / (q @ q)),
                "maximum_weight": float(q[maximum]),
                "maximum_weight_ticker": str(tickers[maximum]),
                "active_firms": int(np.count_nonzero(q > _SUPPORT_TOLERANCE)),
                "one_way_turnover": turnover,
                "kkt_residual": float(solution.kkt_residual),
            }
        )
        vintage_frame = metadata[["ticker", "name", "sector"]].copy()
        vintage_frame.insert(0, "year", year)
        vintage_frame.insert(0, "vintage", vintage)
        vintage_frame["weight"] = q
        weight_rows.append(vintage_frame)
        if previous is not None and previous_vintage is not None:
            changes = q - previous
            order = np.lexsort((tickers, -np.abs(changes)))
            for rank, index in enumerate(order, start=1):
                change_rows.append(
                    {
                        "from_vintage": previous_vintage,
                        "to_vintage": vintage,
                        "from_year": int(previous_vintage[:4]),
                        "to_year": year,
                        "rank": rank,
                        "ticker": str(tickers[index]),
                        "weight_change": float(changes[index]),
                        "absolute_weight_change": float(abs(changes[index])),
                    }
                )
        previous = q
        previous_vintage = vintage
    if final_weights is None:
        _error("PIT archive contains no vintages")
    final_gap = float(np.max(np.abs(final_weights - canonical_weights)))
    if final_gap > _FINAL_VINTAGE_TOLERANCE:
        _error(
            "final expanding-cutoff allocation does not reproduce the canonical "
            f"allocation: maximum gap {final_gap:.3e}"
        )
    return (
        pd.concat(weight_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(change_rows),
        final_gap,
    )


def build_portfolio_anatomy(
    metadata: pd.DataFrame,
    weights: np.ndarray,
    squared_w2: np.ndarray,
    vintages: Mapping[str, np.ndarray],
) -> PortfolioAnatomyArtifacts:
    """Compute all additive anatomy outputs from already aligned inputs."""
    required = {"ticker", "name", "sector"}
    if not required.issubset(metadata.columns) or metadata["ticker"].duplicated().any():
        _error("firm metadata must contain unique ticker, name, and sector columns")
    if metadata[list(required)].isna().any(axis=None):
        _error("firm metadata must not contain missing ticker, name, or sector values")
    q = _check_weights(weights, len(metadata), "canonical weights")
    firms, pairs, sectors, identities = _canonical_frames(metadata, q, squared_w2)
    vintage_weights, vintage_summary, vintage_changes, final_gap = _vintage_frames(
        metadata, q, vintages
    )
    top = firms.sort_values(["weight", "ticker"], ascending=[False, True]).head(10)
    top_weights = top["weight"].to_numpy(dtype=np.float64)
    top_pair_shares = top["strongest_pair_share"].to_numpy(dtype=np.float64)
    top_holdings = [
        {
            "ticker": str(top.iloc[index]["ticker"]),
            "name": str(top.iloc[index]["name"]),
            "sector": str(top.iloc[index]["sector"]),
            "weight": float(top_weights[index]),
            "strongest_partner": str(top.iloc[index]["strongest_partner"]),
            "strongest_pair_share": float(top_pair_shares[index]),
        }
        for index in range(len(top))
    ]
    leading_pairs = pairs.head(12)
    pair_credits = leading_pairs["certificate_credit"].to_numpy(dtype=np.float64)
    pair_shares = leading_pairs["certificate_share"].to_numpy(dtype=np.float64)
    top_pairs = [
        {
            "ticker_i": str(leading_pairs.iloc[index]["ticker_i"]),
            "ticker_j": str(leading_pairs.iloc[index]["ticker_j"]),
            "same_sector": bool(leading_pairs.iloc[index]["same_sector"]),
            "certificate_credit": float(pair_credits[index]),
            "certificate_share": float(pair_shares[index]),
        }
        for index in range(len(leading_pairs))
    ]
    vintage_credit = vintage_summary["certificate_credit"].to_numpy(dtype=np.float64)
    vintage_effective_n = vintage_summary["effective_number_assets"].to_numpy(
        dtype=np.float64
    )
    vintage_maximum = vintage_summary["maximum_weight"].to_numpy(dtype=np.float64)
    vintage_active = vintage_summary["active_firms"].to_numpy(dtype=np.int64)
    vintage_kkt = vintage_summary["kkt_residual"].to_numpy(dtype=np.float64)
    vintage_turnover = vintage_summary["one_way_turnover"].to_numpy(dtype=np.float64)
    vintage_years = vintage_summary["year"].to_numpy(dtype=np.int64)
    vintage_records = [
        {
            "vintage": str(vintage_summary.iloc[index]["vintage"]),
            "year": int(vintage_years[index]),
            "certificate_credit": float(vintage_credit[index]),
            "effective_number_assets": float(vintage_effective_n[index]),
            "maximum_weight": float(vintage_maximum[index]),
            "maximum_weight_ticker": str(
                vintage_summary.iloc[index]["maximum_weight_ticker"]
            ),
            "active_firms": int(vintage_active[index]),
            "one_way_turnover": (
                None
                if np.isnan(vintage_turnover[index])
                else float(vintage_turnover[index])
            ),
            "kkt_residual": float(vintage_kkt[index]),
        }
        for index in range(len(vintage_summary))
    ]
    summary: dict[str, object] = {
        "schema_version": "paper3_portfolio_anatomy.v1",
        "evidence_status": "full_sample_descriptive",
        "interpretation": (
            "Anatomy of the canonical news-only allocation; "
            "not a causal sector decomposition or prospective performance test."
        ),
        "portfolio": {
            "n_firms": len(firms),
            "active_firms": int(firms["active"].sum()),
            "effective_number_assets": float(1.0 / (q @ q)),
            "maximum_weight": float(q.max()),
            "maximum_weight_ticker": str(metadata.iloc[int(np.argmax(q))]["ticker"]),
        },
        "certificate": {
            **identities,
            "within_sector_share": identities["within_sector_credit"]
            / identities["certificate_credit"],
            "cross_sector_share": identities["cross_sector_credit"]
            / identities["certificate_credit"],
        },
        "identity_checks": {
            "firm_share_max_abs_gap": identities["firm_share_max_abs_gap"],
            "pair_credit_abs_gap": abs(
                identities["pair_credit_sum"] - identities["certificate_credit"]
            ),
            "sector_credit_abs_gap": abs(
                identities["sector_credit_sum"] - identities["certificate_credit"]
            ),
            "final_vintage_weight_max_abs_gap": final_gap,
            "support_tolerance": _SUPPORT_TOLERANCE,
        },
        "top_holdings": top_holdings,
        "top_pairs": top_pairs,
        "vintages": vintage_records,
        "vintage_design": {
            "kind": "expanding_information_cutoffs",
            "causal_scope": (
                "Calendar transitions describe allocation sensitivity as the "
                "information set expands; they do not identify event effects."
            ),
        },
    }
    return PortfolioAnatomyArtifacts(
        firms=firms,
        pairs=pairs,
        sectors=sectors,
        vintage_weights=vintage_weights,
        vintage_summary=vintage_summary,
        vintage_changes=vintage_changes,
        summary=summary,
    )


def _load_typed_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    frame, summary = read_typed_distance_artifact(
        path,
        expected_identity={
            "distance_id": "wasserstein_w2",
            "value_semantics": "statistical_distance",
        },
    )
    item_ids = summary.get("item_ids")
    metadata = summary.get("metadata")
    if not isinstance(item_ids, list) or not isinstance(metadata, dict):
        _error("typed distance artifact has malformed roster metadata")
    if metadata.get("normalization") != "rooted":
        _error("Paper 3 anatomy requires rooted W2 and performs the single square")
    tickers = [str(item) for item in item_ids]
    n_firms = len(tickers)
    expected_i = np.repeat(tickers, n_firms)
    expected_j = np.tile(tickers, n_firms)
    if (
        frame.shape[0] != n_firms * n_firms
        or not np.array_equal(frame["item_i"].astype(str).to_numpy(), expected_i)
        or not np.array_equal(frame["item_j"].astype(str).to_numpy(), expected_j)
    ):
        _error("typed distance rows do not follow the declared full-matrix roster")
    rooted = frame["value"].to_numpy(dtype=np.float64).reshape(n_firms, n_firms)
    return tickers, np.square(rooted, dtype=np.float64)


def _load_aligned_inputs(
    paths: Paper3PortfolioAnatomyPaths,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    tickers, squared_w2 = _load_typed_matrix(paths.distance_artifact_dir)
    weights_frame = pd.read_parquet(paths.weights_path)
    if set(weights_frame.columns) != {"ticker", "weight"}:
        _error("canonical weight artifact must contain only ticker and weight")
    duplicate_weights = weights_frame["ticker"].duplicated().any()
    if duplicate_weights or set(weights_frame["ticker"]) != set(tickers):
        _error("canonical weight roster must match the typed distance roster")
    weights = (
        weights_frame.set_index("ticker")
        .loc[tickers, "weight"]
        .to_numpy(dtype=np.float64)
    )

    universe = pd.read_csv(paths.universe_csv)
    required = {"Symbol", "Name", "Sector"}
    if not required.issubset(universe.columns) or universe["Symbol"].duplicated().any():
        _error("universe CSV must contain unique Symbol, Name, and Sector columns")
    indexed = universe.set_index("Symbol")
    missing = sorted(set(tickers) - set(indexed.index.astype(str)))
    if missing:
        _error("universe CSV is missing priced tickers: " + ", ".join(missing))
    metadata = indexed.loc[tickers, ["Name", "Sector"]].reset_index()
    metadata.columns = ["ticker", "name", "sector"]

    manifest = read_manifest(paths.pit_manifest_path)
    pit_metadata = manifest.get("metadata")
    if not isinstance(pit_metadata, dict):
        _error("PIT manifest is missing metadata")
    if pit_metadata.get("archive_scale") != "squared_wasserstein_2_transport_cost":
        _error("PIT archive must already contain squared W2 transport costs")
    if [str(item) for item in pit_metadata.get("item_ids", [])] != tickers:
        _error("PIT roster must exactly match the canonical distance roster")
    windows = pit_metadata.get("vintage_windows")
    if not isinstance(windows, dict) or not windows:
        _error("PIT manifest declares no vintage windows")
    with np.load(paths.pit_w2_path, allow_pickle=False) as archive:
        if set(archive.files) != set(windows):
            _error("PIT archive keys do not match its manifest windows")
        vintages = {
            vintage: np.asarray(archive[vintage], dtype=np.float64)
            for vintage in sorted(windows)
        }
    return metadata, weights, squared_w2, vintages


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix
    )
    os.close(descriptor)
    return Path(name)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_yaml(path: Path, value: object) -> None:
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _code_identity() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_paper3_portfolio_anatomy(paths: Paper3PortfolioAnatomyPaths) -> None:
    """Load governed inputs, compute anatomy outputs, and persist a manifest."""
    metadata, weights, squared_w2, vintages = _load_aligned_inputs(paths)
    artifacts = build_portfolio_anatomy(metadata, weights, squared_w2, vintages)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "firm_anatomy.parquet": artifacts.firms,
        "pair_contributions.parquet": artifacts.pairs,
        "sector_summary.parquet": artifacts.sectors,
        "vintage_weights.parquet": artifacts.vintage_weights,
        "vintage_summary.parquet": artifacts.vintage_summary,
        "vintage_changes.parquet": artifacts.vintage_changes,
    }
    for name, frame in frames.items():
        _write_parquet(paths.output_dir / name, frame)
    summary_path = paths.output_dir / "summary.yaml"
    _write_yaml(summary_path, artifacts.summary)
    output_paths = [paths.output_dir / name for name in frames] + [summary_path]
    upstreams = (
        paths.weights_path,
        paths.distance_artifact_dir,
        paths.pit_w2_path,
        paths.pit_manifest_path,
        paths.universe_csv,
    )
    parameter_payload = {
        "support_tolerance": _SUPPORT_TOLERANCE,
        "identity_tolerance": _IDENTITY_TOLERANCE,
        "vintages": sorted(vintages),
    }
    write_manifest(
        paths.output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "p3_portfolio_anatomy",
            "lane": "p3",
            "phase": "estimate",
            "protocol_id": "paper3.portfolio-anatomy.v1",
            "code_identity": {"kind": "sha256", "value": _code_identity()},
            "parameter_identity": {
                "kind": "sha256",
                "value": hashlib.sha256(
                    json.dumps(parameter_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "seed_policy": "deterministic-no-randomness",
            "upstream_artifacts": [
                {"kind": "path-sha256", "value": f"{path}:{sha256_path(path)}"}
                for path in upstreams
            ],
            "environment": {"python": platform.python_version(), "dtype": "float64"},
            "outputs": [
                {
                    "path": str(path),
                    "kind": path.suffix.removeprefix("."),
                    "sha256": sha256_path(path),
                    "bytes": path.stat().st_size,
                }
                for path in output_paths
            ],
            "risk_class": "moderate",
            "metadata": {
                "schema_version": artifacts.summary["schema_version"],
                "identity_checks": artifacts.summary["identity_checks"],
            },
        },
    )


__all__ = [
    "Paper3PortfolioAnatomyPaths",
    "PortfolioAnatomyArtifacts",
    "PortfolioAnatomyError",
    "build_portfolio_anatomy",
    "run_paper3_portfolio_anatomy",
]
