"""Per-window covariance estimators, the weights they induce, and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import numpy as np
from jcor.model.covariance import ledoit_wolf_sample, sample_covariance
from jcor.operators.covariance import sigma_dist
from jcor.optimize.psd import nearest_covariance

from pipeline.stages.papers.paper5._energy_robust.contracts import (
    RobustOptimizationRequest,
)
from pipeline.stages.papers.paper5._energy_robust.optimize import (
    _nearest_psd_host,
    _pgd_weights,
)
from pipeline.stages.papers.paper5._energy_robust.weights import (
    certified_caps,
    robust_weights,
)
from pipeline.stages.papers.paper5.energy_radius import (
    CovarianceBracketInputs,
    assemble_brackets,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        Float,
        RobustCovarianceInputs,
    )


def _cov_and_weights(
    train: Float,
    r_t: float,
    inputs: RobustCovarianceInputs,
) -> dict[str, object]:
    """Build all estimators + strategy weights for one training window.

    The distance-to-covariance scale and radius inputs are fixed on the
    calibration epoch. The per-window return covariance diagonal remains
    causal: it is recomputed from ``train`` only. Both bracket modes are built
    per window: CALIBRATED (``kappa_lo = kappa_hi`` = calibration kappa, GMM
    ``D^2`` correction) drives ``robust_corner`` / ``higham_corner`` /
    ``robust_select``; CERTIFIED (envelope ``[ell_lo, L_hat]``, no GMM -- see
    the no-double-counting note in
    :func:`pipeline.stages.papers.paper5.energy_radius.assemble_brackets`) drives
    ``robust_corner_cert``.

    ``cert_stale_vol`` ("price-light"): the CERTIFIED corner rebuilt with
    ``calib_sigma_sq`` -- the CALIBRATION-epoch per-asset variance vector --
    in place of ``train``'s window variance, so EVERY input to
    ``assemble_brackets`` (``d2``, ``eps_stat_matrix``, ``envelope``, AND
    now the diagonal) is a calibration-epoch constant, held FIXED across all
    backtest windows. Its weights are therefore window-invariant, unlike
    every other strategy here, which recomputes sigma^2 from ``train`` per
    window. HONESTY SCOPING: ``train`` (and, in :func:`backtest`, the
    evaluation-epoch return panel generally) is used ONLY to EVALUATE
    realized performance of ``cert_stale_vol``'s weights -- never to FORM
    them; the strategy needs NO evaluation-epoch price data at all to
    produce a portfolio. The resulting vol staleness (calibration-epoch
    variance standing in for the true window variance) is exactly the
    delta-radius covered by the newly proven Lean theorem
    ``covariance_bracket_of_variance_radius`` (``RobustEnergy.lean``), which
    brackets the true covariance using an observable variance proxy known
    only up to a slack ``delta_i`` -- here realized as the calibration/
    evaluation variance gap.
    """
    n = train.shape[1]
    kappa = inputs.calibration_kappa
    kappa_abs = abs(kappa)
    sigma_sq = np.diag(sample_covariance(train, ddof=1))
    # The PSD repair is traced, so Σ_dist stays on device through it; x64 is
    # required for its nearest-matrix contract and is what this stage runs at.
    with jax.enable_x64(new_val=True):
        sd = np.array(
            nearest_covariance(
                sigma_dist(inputs.squared_distances, sigma_sq, kappa=kappa)
            ).matrix,
            dtype=np.float64,
        )

    brackets_cal = assemble_brackets(
        CovarianceBracketInputs(
            inputs.squared_distances,
            inputs.eps_stat_matrix,
            inputs.eps_gmm_d2,
            sigma_sq,
            kappa_abs,
            kappa_abs,
        )
    )
    brackets_cert = assemble_brackets(
        CovarianceBracketInputs(
            inputs.squared_distances,
            inputs.eps_stat_matrix,
            np.zeros_like(inputs.squared_distances),
            sigma_sq,
            inputs.envelope["ell_lo"],
            inputs.envelope["L_hat"],
        )
    )
    sigma_hi = cast("Float", brackets_cal["Sigma_hi"])
    sigma_hi_cert = cast("Float", brackets_cert["Sigma_hi"])

    # price-light: CERTIFIED corner rebuilt on the FIXED calibration-epoch
    # variance vector -- cheap (one more assemble_brackets call) and
    # entirely calibration-epoch, hence window-invariant (see docstring).
    brackets_stale = assemble_brackets(
        CovarianceBracketInputs(
            inputs.squared_distances,
            inputs.eps_stat_matrix,
            np.zeros_like(inputs.squared_distances),
            inputs.calibration_sigma_sq,
            inputs.envelope["ell_lo"],
            inputs.envelope["L_hat"],
        )
    )
    sigma_hi_stale = cast("Float", brackets_stale["Sigma_hi"])

    w_sample = _pgd_weights(np.asarray(sample_covariance(train, ddof=1)))
    w_lw = _pgd_weights(np.asarray(ledoit_wolf_sample(train).sigma))
    w_dist = _pgd_weights(sd)
    w_corner = robust_weights(RobustOptimizationRequest("corner", sigma_hi=sigma_hi))
    w_higham = robust_weights(
        RobustOptimizationRequest("higham_corner", sigma_hi=sigma_hi)
    )
    w_corner_cert = robust_weights(
        RobustOptimizationRequest("corner", sigma_hi=sigma_hi_cert)
    )
    w_stale = robust_weights(
        RobustOptimizationRequest("corner", sigma_hi=sigma_hi_stale)
    )
    w_ridge = robust_weights(
        RobustOptimizationRequest("ridge", sigma_hat=sd, radius=r_t)
    )
    w_equal = np.ones(n) / n

    cap_corner = certified_caps(w_corner, sigma_hi, sd, r_t)
    cap_corner_cert = certified_caps(w_corner_cert, sigma_hi_cert, sd, r_t)
    cap_stale = certified_caps(w_stale, sigma_hi_stale, sd, r_t)
    cap_ridge = certified_caps(w_ridge, sigma_hi, sd, r_t)
    w_select = (
        w_corner
        if cap_corner["certified_cap"] <= cap_ridge["certified_cap"]
        else w_ridge
    )

    weights = {
        "sample": w_sample,
        "ledoit_wolf": w_lw,
        "sigma_dist": w_dist,
        "robust_corner": w_corner,
        "higham_corner": w_higham,
        "robust_corner_cert": w_corner_cert,
        "cert_stale_vol": w_stale,
        "robust_ridge": w_ridge,
        "robust_select": w_select,
        "equal_weight": w_equal,
    }
    covs = {
        "sample": sample_covariance(train, ddof=1),
        "ledoit_wolf": ledoit_wolf_sample(train).sigma,
        "sigma_dist": sd,
        "robust_corner": sigma_hi,
        "higham_corner": _nearest_psd_host(sigma_hi),
        "robust_corner_cert": sigma_hi_cert,
        "cert_stale_vol": sigma_hi_stale,
        "robust_ridge": sd,
        "robust_select": sigma_hi if w_select is w_corner else sd,
        "equal_weight": sample_covariance(train, ddof=1),
    }
    _diag_keys = (
        "bracket_width_D_median",
        "bracket_width_D_mean",
        "bracket_width_Sigma_median",
        "bracket_width_Sigma_mean",
    )
    return {
        "weights": weights,
        "covs": covs,
        "kappa": kappa,
        "bracket_diag": {
            "calibrated": {k: brackets_cal[k] for k in _diag_keys},
            "certified": {k: brackets_cert[k] for k in _diag_keys},
        },
        "cap_corner": cap_corner,
        "cap_corner_cert": cap_corner_cert,
        "cap_stale": cap_stale,
        "cap_ridge": cap_ridge,
    }


def _portfolio_diagnostics(sigma: Float, w: Float) -> dict[str, float]:
    ev = np.abs(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    hhi = float(np.sum(w**2))
    eff_n = float(1.0 / hhi) if hhi > 0 else 0.0
    return {"cond": cond, "eff_n": eff_n, "hhi": hhi}
