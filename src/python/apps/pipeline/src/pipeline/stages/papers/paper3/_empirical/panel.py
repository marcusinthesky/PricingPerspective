"""W2 loading, transmission-scale calibration, and covariance target construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pandas as pd
from jcor.model.covariance import (
    plug_in_lambda_dist,
    sample_covariance,
    shrink_to_dist,
)
from jcor.operators.covariance import sigma_dist
from jcor.optimize.psd import nearest_covariance, repair_diagnostics

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.papers.paper3._empirical.contracts import Float


def load_panel(
    returns_dir: Path,
    distance_artifact_dir: Path,
    *,
    provider_id: str = "qwen3-embedding-8b",
    representation_id: str = "qwen3-embedding-8b-unit",
    distance_id: str = "wasserstein_w2",
) -> tuple[list[str], Float, Float, pd.DatetimeIndex]:
    """Load returns and convert the typed rooted-W2 matrix to ``W2^2``."""
    frame, summary = read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
            "value_semantics": "statistical_distance",
        },
    )
    metadata = summary.get("metadata")
    item_ids = summary.get("item_ids")
    if not isinstance(metadata, dict) or not isinstance(item_ids, list):
        msg = "typed W2 artifact has malformed summary metadata"
        raise TypeError(msg)
    if (
        metadata.get("family") != "wasserstein"
        or metadata.get("estimator") != "balanced_wasserstein_2"
        or metadata.get("normalization") != "rooted"
    ):
        msg = (
            "Paper 3 requires rooted balanced W2, got "
            f"{metadata.get('family')}/{metadata.get('estimator')}/"
            f"{metadata.get('normalization')}"
        )
        raise ValueError(msg)
    distance_tickers = [str(item) for item in item_ids]
    rooted_w2 = (
        frame["value"]
        .to_numpy(dtype=np.float64)
        .reshape(len(distance_tickers), len(distance_tickers))
    )
    ret_syms = {p.stem for p in returns_dir.glob("*.parquet")}
    tickers = sorted(set(distance_tickers) & ret_syms)
    distance_indices = [distance_tickers.index(ticker) for ticker in tickers]

    # Aligned return panel: inner-join on Date across all tickers.
    frames = []
    for t in tickers:
        df = pd.read_parquet(returns_dir / f"{t}.parquet")[["Date", "return"]]
        frames.append(df.rename(columns={"return": t}).set_index("Date"))
    panel = pd.concat(frames, axis=1, join="inner").dropna()
    returns = panel[tickers].to_numpy(dtype=np.float64)

    # Typed W2 artifacts persist rooted values. Polarization consumes the
    # quadratic transport cost, so this is the one intentional square.
    squared_distances = np.square(
        rooted_w2[np.ix_(distance_indices, distance_indices)], dtype=np.float64
    )
    np.fill_diagonal(squared_distances, 0.0)
    dates = pd.DatetimeIndex(panel.index)
    return tickers, returns, squared_distances, dates


def calibrate_kappa(returns: Float, squared_distances: Float) -> tuple[float, float]:
    """Calibrate the scalar transmission benchmark ``κ`` by least squares.

    The identity ``S_ij = ½(σ_i² + σ_j² − κ²·D²_ij)`` gives, off-diagonal,
    ``y_ij := ½(σ_i² + σ_j²) − S_ij = ½κ²·D²_ij``.  Regressing ``y`` on
    ``½·D²`` through the origin yields ``θ̂ = κ²`` and its standard error;
    ``κ̂ = √θ̂`` with delta-method SE ``se(θ̂)/(2κ̂)``.

    Args:
        returns: Return panel, shape ``(T, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.

    Returns:
        Tuple ``(kappa_hat, kappa_se)``.

    """
    sample_covariance_matrix = sample_covariance(returns, ddof=1)
    v = np.diag(sample_covariance_matrix)
    n = sample_covariance_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    y = 0.5 * (v[iu[0]] + v[iu[1]]) - sample_covariance_matrix[iu]
    x = 0.5 * squared_distances[iu]
    denom = float(np.sum(x * x))
    theta = float(np.sum(x * y) / denom) if denom > 0 else 0.0
    resid = y - theta * x
    dof = max(len(y) - 1, 1)
    s2 = float(np.sum(resid**2) / dof)
    theta_se = float(np.sqrt(s2 / denom)) if denom > 0 else float("inf")
    kappa_hat = float(np.sqrt(theta)) if theta > 0 else 0.0
    kappa_se = theta_se / (2.0 * kappa_hat) if kappa_hat > 0 else float("inf")
    return kappa_hat, kappa_se


# ---------------------------------------------------------------------------
# Step 1: Σ_dist, Σ(λ*), PSD repair
# ---------------------------------------------------------------------------


def build_sigma_dist(
    returns: Float,
    squared_distances: Float,
    kappa: float,
) -> dict[str, object]:
    """Build Σ_dist, its Higham repair, plug-in blend, and diagnostics.

    Args:
        returns: Return panel, shape ``(T, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.
        kappa: Calibrated κ.

    Returns:
        Dict with ``sigma_dist_raw``, ``sigma_dist`` (PSD-repaired),
        ``sigma_shrink``, ``lam``, ``diagnostics`` (a ``RepairDiagnostics``).

    """
    sigma_sq = np.diag(sample_covariance(returns, ddof=1))
    # Σ_dist, its repair, and the diagnostics are all traced; the host sees one
    # materialisation each of the raw and repaired matrices. ``converged`` is
    # now real evidence rather than ``None`` — the repair reports whether its
    # nearest-correlation claim actually holds for this window.
    with jax.enable_x64(new_val=True):
        raw_traced = sigma_dist(squared_distances, sigma_sq, kappa=kappa)
        result = nearest_covariance(raw_traced)
        diag = repair_diagnostics(raw_traced, result.matrix, converged=result.converged)
        raw = np.array(raw_traced, dtype=np.float64)
        repaired = np.array(result.matrix, dtype=np.float64)
    # The repaired target estimates its diagonal variances and κ from these
    # same returns.  It is therefore data-dependent: use the explicitly named
    # plug-in regularisation heuristic, not the fixed-target MSE-optimal API.
    lam, covariance_sum_variance, bias_sq_sum = plug_in_lambda_dist(returns, repaired)
    shrink = shrink_to_dist(returns, repaired, lam=lam)
    return {
        "sigma_dist_raw": raw,
        "sigma_dist": repaired,
        "sigma_shrink": shrink.sigma,
        "lam": lam,
        "var_S_sum": covariance_sum_variance,
        "adjusted_gap_sum": bias_sq_sum,
        # Historical key retained for artifact compatibility; for this
        # same-window target the quantity is not an estimate of target bias.
        "bias_sq_sum": bias_sq_sum,
        "diagnostics": diag,
        # The dual KKT residual and the Newton steps spent reaching it. Carried
        # separately because `RepairDiagnostics` reduces convergence to a
        # boolean, and the manuscript quotes the residual so a reader can see
        # *how far* from nearest the repair is rather than only whether a
        # threshold was cleared.
        #
        # This comment previously recorded the boolean as False and the solver
        # as stalling "four orders above its own stopping tolerance". That was a
        # solver defect, fixed 2026-08-08 in `jcor.optimize.psd` (floored CG
        # ridge; Armijo tested below its objective's float64 granularity), not a
        # property of this panel — see that module's docstring. On this panel
        # the repair now converges in 7 Newton steps to 6.7420e-14 against the
        # 6.0041e-13 tolerance, which is the case `results.tex` and
        # `appendices.tex` were already written for ("clearing it by a factor of
        # about eight").
        "psd_residual": float(result.residual),
        "psd_iterations": int(result.iterations),
    }
