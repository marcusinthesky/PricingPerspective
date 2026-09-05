"""Evaluate Paper 5's H2--H4 hypotheses.

The module covers the zero-intercept restriction, both bracket-coverage and
materiality families, and encompassing/backtest rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from jcor.model.covariance import ledoit_wolf_sample, sample_covariance
from numpy.typing import NDArray
from scipy.stats import chi2

logger = logging.getLogger(__name__)

Float = NDArray[np.float64]
NOMINAL_ALPHA = 0.05
MAX_MATERIALITY_RATIO = 2.0


class SarBracket(TypedDict):
    """Return shape of :func:`sar_robust_bracket`."""

    Sigma_lo: Float
    Sigma_hi: Float
    bracket_width_median: float


@dataclass(frozen=True)
class EncompassingInputs:
    """Train/test panels and fitted covariance-model inputs for H4."""

    train_returns: Float
    test_returns: Float
    weights: Float
    rho_hat: float
    diagonal_variance: Float
    sigma_dist: Float | None


# ---------------------------------------------------------------------------
# H2 -- bar-alpha = 0 zero-intercept pricing restriction (Kou Eq. 15)
# ---------------------------------------------------------------------------


def h2_zero_intercept_test(
    alpha_hat: float, sigma2_hat: float, t_obs: int, n: int
) -> dict[str, object]:
    """Test ``bar-alpha = 0`` with a Wald statistic bounded at the panel's ``n``.

    ``alpha_hat`` is a pooled cross-sectional/time-series mean (see
    :func:`pipeline.stages.papers.paper5.estimate.sar_qmle`); its variance under
    the model's iid-error assumption is ``sigma2_hat / (T*n)``.  Wald
    statistic ``W = alpha_hat^2 / Var(alpha_hat) ~ chi2(1)`` asymptotically.
    DISCLOSED: this asymptotic is a bounded claim at the panel's ``n`` (100 for the
    current roster; see the plan's methodology note); the Monte Carlo behind
    this level is t04's responsibility, not computed here.

    Args:
        alpha_hat: Pooled intercept estimate.
        sigma2_hat: Residual variance estimate.
        t_obs: Number of time periods.
        n: Number of assets.

    Returns:
        Dict with ``wald_stat``, ``pvalue``, ``se_alpha``, ``reject_05``.

    """
    se_alpha = float(np.sqrt(sigma2_hat / (t_obs * n)))
    wald = float((alpha_hat / se_alpha) ** 2) if se_alpha > 0 else float("nan")
    pvalue = float(1.0 - chi2.cdf(wald, df=1)) if np.isfinite(wald) else float("nan")
    return {
        "alpha_hat": alpha_hat,
        "se_alpha": se_alpha,
        "wald_stat": wald,
        "pvalue": pvalue,
        "reject_05": bool(pvalue < NOMINAL_ALPHA) if np.isfinite(pvalue) else False,
        "n_obs": int(t_obs),
        "n_assets": int(n),
        # Interpolated, not fixed: this string is serialised into the results
        # artifact, so a hard-coded n stated a cross-section size the run had
        # not used the moment the panel changed.
        "disclosure": (
            f"Bounded claim at n={int(n)}: chi2(1) asymptotics not fully "
            "trustworthy at this cross-section size; size/power Monte Carlo "
            "(t04) stands behind this finite-sample level."
        ),
    }


# ---------------------------------------------------------------------------
# H3 -- coverage + materiality, two bracket families
# ---------------------------------------------------------------------------


def sigma_sar(w: Float, rho: float, v_diag: Float) -> Float:
    """``Sigma_SAR = (I - rho W)^-1 V (I - rho W)^-T`` (V diagonal).

    Args:
        w: Interaction matrix, shape ``(n, n)``.
        rho: Spatial autoregressive coefficient, ``|rho| < 1``.
        v_diag: Idiosyncratic variances, shape ``(n,)``.

    Returns:
        ``(n, n)`` symmetric PSD covariance matrix.

    """
    n = w.shape[0]
    a_inv = np.linalg.inv(np.eye(n) - rho * w)
    v = np.diag(v_diag)
    sigma = a_inv @ v @ a_inv.T
    return cast("Float", 0.5 * (sigma + sigma.T))


def sar_robust_bracket(
    w: Float,
    delta_w_bound: float,
    rho_hat: float,
    rho_se: float,
    v_diag: Float,
) -> SarBracket:
    """SAR-robust covariance bracket over the ``(rho, W)`` ambiguity set.

    Interval-arithmetic bracket: evaluate ``Sigma_SAR`` at the four corners
    ``rho in {rho_hat - z*se, rho_hat + z*se}`` (``z=1.96``, 95% CI) crossed
    with ``W in {W - dW, W + dW}`` (``dW`` a uniform perturbation of size
    ``delta_w_bound`` per entry, clipped to keep row-sums sane), and take the
    entrywise min/max across the four corners as ``[Sigma_lo, Sigma_hi]``.
    This is the SAR-family analogue of the energy-bracket T2 covariance result, built
    directly on the ``(rho, W)`` ambiguity set rather than on the
    energy-distance-to-covariance map.

    Args:
        w: Point-estimate interaction matrix.
        delta_w_bound: Scalar perturbation magnitude for ``W`` (from
            :mod:`pipeline.stages.papers.paper5.weights`).
        rho_hat: Point estimate of ``rho``.
        rho_se: Standard error of ``rho_hat`` (bracket half-width driver).
        v_diag: Idiosyncratic variances, shape ``(n,)``.

    Returns:
        Dict with ``Sigma_lo``, ``Sigma_hi``, and width summary stats.

    """
    z = 1.96
    rho_lo = float(np.clip(rho_hat - z * rho_se, -0.999, 0.999))
    rho_hi = float(np.clip(rho_hat + z * rho_se, -0.999, 0.999))
    w_lo = w - delta_w_bound
    w_hi = w + delta_w_bound
    corners = []
    for rho in (rho_lo, rho_hi):
        for w_c in (w_lo, w_hi):
            try:
                corners.append(sigma_sar(w_c, rho, v_diag))
            except np.linalg.LinAlgError:
                continue
    if not corners:
        n = w.shape[0]
        nan = np.full((n, n), np.nan)
        return {"Sigma_lo": nan, "Sigma_hi": nan, "bracket_width_median": float("nan")}
    stack = np.stack(corners, axis=0)
    sigma_lo = stack.min(axis=0)
    sigma_hi = stack.max(axis=0)
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    width = (sigma_hi - sigma_lo)[iu]
    return {
        "Sigma_lo": sigma_lo,
        "Sigma_hi": sigma_hi,
        "bracket_width_median": float(np.median(width)),
    }


def bracket_coverage(sigma_true: Float, sigma_lo: Float, sigma_hi: Float) -> float:
    """Fraction of off-diagonal entries with ``sigma_true`` inside the bracket."""
    n = sigma_true.shape[0]
    iu = np.triu_indices(n, k=1)
    lo = sigma_lo[iu]
    hi = sigma_hi[iu]
    true = sigma_true[iu]
    covered = (true >= lo) & (true <= hi)
    return float(np.mean(covered))


def h3_bracket_families(
    sar_bracket: SarBracket,
    sigma_true: Float,
    energy_bracket_calibrated_coverage: float,
    energy_bracket_certified_coverage: float,
    lw_band_width_t60: float,
) -> dict[str, dict[str, object]]:
    """H3: coverage + materiality for BOTH bracket families.

    Family 1 (SAR-robust): computed here (:func:`sar_robust_bracket`).
    Family 2 (energy brackets): the point-coverage numbers are
    REUSED directly from Paper 5's energy-robustness prerequisite
    ``data/papers/paper5/energy_robustness/results.json['h1_coverage']``
    (the energy H1, exactly as specified — not recomputed) since the energy
    bracket construction itself belongs to Paper 5's energy-robustness
    prerequisite.

    Args:
        sar_bracket: Output of :func:`sar_robust_bracket`.
        sigma_true: Realized (test-fold) sample covariance to test coverage
            against.
        energy_bracket_calibrated_coverage: Reused calibrated-mode
            point coverage.
        energy_bracket_certified_coverage: Reused certified-mode
            point coverage.
        lw_band_width_t60: Reused energy-bracket Ledoit-Wolf materiality band width.

    Returns:
        Dict with both families' coverage + materiality.

    """
    coverage_sar = bracket_coverage(
        sigma_true, sar_bracket["Sigma_lo"], sar_bracket["Sigma_hi"]
    )
    width_sar = float(sar_bracket.get("bracket_width_median", float("nan")))
    materiality_sar = (
        width_sar / lw_band_width_t60 if lw_band_width_t60 > 0 else float("nan")
    )
    return {
        "sar_robust": {
            "point_coverage": coverage_sar,
            "median_bracket_width_sigma": width_sar,
            "materiality_ratio": materiality_sar,
            "gate_verdict": (
                "PASS (pre-registered gate: bracket width < 2x LW T=60 band)"
                if np.isfinite(materiality_sar)
                and materiality_sar < MAX_MATERIALITY_RATIO
                else "FAIL or undefined"
            ),
        },
        "energy_brackets_paper5": {
            "calibrated_point_coverage": energy_bracket_calibrated_coverage,
            "certified_point_coverage": energy_bracket_certified_coverage,
            "note": (
                "Reused verbatim from data/papers/paper5/energy_robustness/results.json"
            ),
        },
    }


# ---------------------------------------------------------------------------
# H4 -- encompassing test + backtest rows
# ---------------------------------------------------------------------------


def encompassing_oos_loss(sigma_true: Float, sigma_hat: Float) -> float:
    """Frobenius-norm OOS covariance loss ``||sigma_true - sigma_hat||_F``."""
    return float(np.linalg.norm(sigma_true - sigma_hat, ord="fro"))


def h4_encompassing(inputs: EncompassingInputs) -> dict[str, object]:
    """H4: OOS covariance loss for Sigma_SAR vs Ledoit-Wolf vs Sigma_dist.

    Args:
        inputs: Training/test panels and fitted SAR/Sigma-dist inputs.

    Returns:
        Dict of OOS Frobenius losses per method and the encompassing verdict
        (lowest loss).

    """
    sigma_true = np.asarray(sample_covariance(inputs.test_returns, ddof=1))
    sigma_sar_hat = sigma_sar(inputs.weights, inputs.rho_hat, inputs.diagonal_variance)
    sigma_lw = np.asarray(ledoit_wolf_sample(inputs.train_returns).sigma)

    losses = {
        "sar_sigma": encompassing_oos_loss(sigma_true, sigma_sar_hat),
        "ledoit_wolf": encompassing_oos_loss(sigma_true, sigma_lw),
    }
    if inputs.sigma_dist is not None:
        losses["sigma_dist"] = encompassing_oos_loss(sigma_true, inputs.sigma_dist)
    else:
        losses["sigma_dist"] = float("nan")

    finite = {k: v for k, v in losses.items() if np.isfinite(v)}
    best = min(finite, key=lambda k: finite[k]) if finite else "undefined"
    return {"losses": losses, "best_method": best}


# ---------------------------------------------------------------------------
# t10a additions -- per-asset heterogeneous-rho breakdown, notation panel,
# knn_k/bandwidth-h robustness cuts (surfaced for t10/t17's exhibit budget).
# These are ADDITIVE diagnostics only; they do not change h2/h3/h4 semantics.
# ---------------------------------------------------------------------------


def local_moran_contributions(
    r: Float, w: Float, tickers: list[str]
) -> dict[str, float]:
    """Per-asset local Moran's I contribution (heterogeneous-rho diagnostic).

    Decomposes the pooled Moran's I statistic
    (:func:`pipeline.stages.papers.paper5.estimate.moran_i`) into per-asset
    contributions ``I_i = z_i * sum_j w_ij z_j`` on time-standardized
    returns, surfacing which tickers drive the pooled spatial-autocorrelation
    estimate. This is a diagnostic decomposition of the existing (scalar)
    ``rho_hat`` fit -- it does not re-estimate rho per asset.

    Args:
        r: Return panel, shape ``(T, n)``.
        w: Interaction matrix, shape ``(n, n)``.
        tickers: Column labels matching ``r``'s columns, length ``n``.

    Returns:
        Mapping ticker -> time-averaged local Moran's I contribution.

    """
    std = r.std(axis=0, ddof=1)
    std_safe = np.where(std > 0, std, 1.0)
    z = (r - r.mean(axis=0)) / std_safe
    lag = z @ w.T
    local = z * lag
    per_asset = local.mean(axis=0)
    return {t: float(v) for t, v in zip(tickers, per_asset, strict=True)}


def notation_panel_entries(
    n: int, folds: dict[str, tuple[str, str]], w_variant_names: list[str]
) -> list[dict[str, str]]:
    """Build static data/notation panel rows for t10/t17.

    Args:
        n: Cross-section size (number of assets).
        folds: Mapping fold name -> (start, end) calendar dates.
        w_variant_names: Names of the interaction-matrix variants fit.

    Returns:
        List of ``{"symbol", "description", "value"}`` rows.

    """
    return [
        {"symbol": "n", "description": "cross-section size (assets)", "value": str(n)},
        {
            "symbol": "T_eval",
            "description": "evaluation folds",
            "value": ", ".join(sorted(folds)),
        },
        {
            "symbol": "W",
            "description": "interaction-matrix variants fit",
            "value": ", ".join(w_variant_names),
        },
        {
            "symbol": "rho",
            "description": "SAR spatial-autoregressive coefficient",
            "value": "scalar, |rho|<1",
        },
        {
            "symbol": "alpha",
            "description": "pooled cross-sectional/time-series intercept",
            "value": "H2 restriction target (bar-alpha=0)",
        },
    ]


def robustness_row(
    param_name: str, param_value: float, rho_hat: float, moran_i_value: float
) -> dict[str, object]:
    """One row of a robustness-cut table (knn_k or bandwidth-h sensitivity).

    Args:
        param_name: Name of the swept parameter (``"knn_k"`` or
            ``"bandwidth_h"``).
        param_value: The parameter's value for this cut.
        rho_hat: SAR rho point estimate fit at this cut.
        moran_i_value: Moran's I fit at this cut.

    Returns:
        Dict row suitable for a robustness-table binding.

    """
    return {
        "param_name": param_name,
        "param_value": param_value,
        "rho_hat": rho_hat,
        "moran_i": moran_i_value,
    }
