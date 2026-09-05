"""One-off numerical validation of ``sar_qmle``/``sar_gmm`` against pysal/spreg.

Compares this codebase's net-new SAR estimators (Ord-Jacobian concentrated
QMLE and Kelejian-Prucha 2SLS/GMM, both in
:mod:`pipeline.stages.papers.paper5.estimate`)
entry-for-entry against the reference implementations ``spreg.ML_Lag`` (QMLE)
and ``spreg.GM_Lag`` (2SLS/GMM), on both the synthetic ``mc.py`` SAR DGP and
(non-smoke path) the empirical panel.

pysal/``spreg`` is a validation-only dependency (added to the pipeline
project's dev/validation dependency group by a prior task); it is never a
runtime dependency of the pipeline stage graph. Import it as ``spreg`` /
``libpysal`` directly -- the legacy ``pysal.spreg`` namespace no longer
resolves in the pinned pysal release.

This module is importable and side-effect-free at import time; run it via
``main(smoke=True)`` for a fast single-tiny-case check, or ``main()`` for the
full grid (deferred -- not run by this task; see t14 README).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import jax
import numpy as np
from jcor.core.random import cell_key
from libpysal.weights import full2W
from numpy.typing import NDArray
from simulation.dgp.spatial import SarDGP, make_synthetic_w, simulate_sar
from spreg import GM_Lag, ML_Lag
from threadpoolctl import threadpool_limits

from pipeline.io.settings import runtime_settings
from pipeline.stages.papers.paper5.estimate import sar_gmm, sar_qmle
from pipeline.stages.papers.paper5.mc import (
    RHO_GRID,
    T_SMALL,
)
from pipeline.stages.papers.paper5.weights import build_w_kernel
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

logger = logging.getLogger(__name__)

Float = NDArray[np.float64]

_GMM_BOUNDARY_EPS = 1e-3
_SPREG_SINGULAR_INSTRUMENT_ERROR = (
    "Singular matrix Z'H(H'H)^-1H'Z - endogenous variable(s) may be part of X"
)


class _SpregEstimates(TypedDict):
    """Store reference ``spreg`` estimates and GM availability diagnostics."""

    ml_lag_rho: float
    gm_lag_rho: float | None
    gm_lag_singular_instruments: bool


def _spreg_estimates(r: Float, w: Float) -> _SpregEstimates:
    """Run ``spreg`` estimators on the panel's mean cross-section.

    ``spreg`` expects a single cross-section ``(n, 1)`` ``y``, a fixed-effects
    ``x`` (intercept-only here), and a ``libpysal`` weights object built from
    the same fixed ``w`` used by ``sar_qmle``/``sar_gmm``. The comparison uses
    the cross-sectional (time-averaged) collapse of the panel, since
    ``ML_Lag``/``GM_Lag`` are single-cross-section estimators, not pooled
    panel estimators.
    """
    n = w.shape[0]
    y = r.mean(axis=0).reshape(-1, 1)
    # spreg adds the constant internally; `x` excludes it (intercept-only
    # model here, matching sar_qmle/sar_gmm's single common-alpha term).
    x = np.zeros((n, 0))
    weights = full2W(w)
    weights.transform = "O"  # keep the raw (already row-stochastic) W as-is

    ml = ML_Lag(y, x, w=weights, method="full")

    try:
        gm = GM_Lag(y, x, w=weights, w_lags=1)
    except Exception as exc:
        # spreg 1.8.5 converts the underlying numpy.linalg.LinAlgError to a
        # generic Exception, so the exact pinned message is the narrowest
        # available discriminator. Propagate every other GM_Lag failure.
        if str(exc) != _SPREG_SINGULAR_INSTRUMENT_ERROR:
            raise
        logger.exception(
            "spreg.GM_Lag encountered the expected singular intercept-only "
            "instrument design (n=%d, x_columns=%d, w_lags=1); recording the "
            "reference estimate as unavailable",
            n,
            x.shape[1],
        )
        # Row-stochastic W means W @ 1 = 1: with an intercept-only x, spreg's
        # default WX-order instrument set collapses to constants, so
        # Z'H(H'H)^-1H'Z is exactly singular. This is the same degeneracy
        # this project's own instrument set {1, W1, W^2 1} is exposed to
        # (see estimate.py sar_gmm docstring) -- disclosed here rather than
        # masked; not fixed by this setup task (see t14 README, deferred D1).
        gm_lag_rho = None
        gm_lag_singular = True
    else:
        # Keep result extraction outside the recovery boundary: malformed
        # estimator output is not the known singular-instrument fallback.
        gm_lag_rho: float | None = float(gm.betas[-1, 0])
        gm_lag_singular = False

    return {
        "ml_lag_rho": float(ml.rho),
        "gm_lag_rho": gm_lag_rho,
        "gm_lag_singular_instruments": gm_lag_singular,
    }


def _compare_one(rho_true: float, n: int, t_obs: int, seed: int) -> dict[str, Any]:
    key = cell_key(
        seed,
        "paper5_validate_estimators",
        "rho",
        rho_true,
        "n",
        n,
        "T",
        t_obs,
    )
    weight_key, returns_key = jax.random.split(key)
    w_jax = make_synthetic_w(weight_key, n)
    r = np.asarray(
        simulate_sar(
            returns_key,
            SarDGP(w_jax, rho_true, alpha=0.01, sigma=0.05, n_observations=t_obs),
        )
    )
    # Validation crosses into libpysal/spreg, whose ``full2W`` adapter uses
    # NumPy-style list indexing that JAX arrays intentionally reject. Keep the
    # DGP key-only, then materialize one validation-boundary copy for both the
    # local and reference estimators.
    w = np.asarray(w_jax, dtype=np.float64)

    qmle = sar_qmle(r, w)
    gmm = sar_gmm(r, w)
    ref = _spreg_estimates(r, w)

    gmm_clip_flag = abs(abs(gmm["rho_hat"]) - (1.0 - 1e-3)) < _GMM_BOUNDARY_EPS

    return {
        "rho_true": rho_true,
        "n": n,
        "t_obs": t_obs,
        "qmle_rho_hat": qmle["rho_hat"],
        "gmm_rho_hat": gmm["rho_hat"],
        "ml_lag_rho": ref["ml_lag_rho"],
        "gm_lag_rho": ref["gm_lag_rho"],
        "gm_lag_singular_instruments": ref["gm_lag_singular_instruments"],
        "qmle_vs_ml_lag_diff": qmle["rho_hat"] - ref["ml_lag_rho"],
        "gmm_vs_gm_lag_diff": (
            gmm["rho_hat"] - ref["gm_lag_rho"]
            if ref["gm_lag_rho"] is not None
            else None
        ),
        "gmm_boundary_clip_flag": gmm_clip_flag,
    }


def run_synthetic_grid(
    seed: int = 0, n: int = 20, t_obs: int = T_SMALL, *, smoke: bool = False
) -> list[dict[str, Any]]:
    """Compare estimators across the rho grid or one smoke case."""
    grid = (RHO_GRID[0],) if smoke else RHO_GRID
    return [_compare_one(rho_true, n, t_obs, seed) for rho_true in grid]


def run_empirical_comparison() -> dict[str, Any]:
    """Compare estimators on the empirical panel (deferred -- not run in smoke mode).

    Uses the 100-ticker priced universe (``pipeline.stages.substrate.panel
    .load_universe`` over ``data/shared/returns`` and the qwen3-embedding-8b
    ``energy_tests.parquet``, the ``${baseline_model}`` default in
    ``dvc.yaml``) and the diffusion-kernel interaction matrix ``W^h``
    (:func:`pipeline.stages.papers.paper5.weights.build_w_kernel`) rather than
    the shared typed replication artifact ``W^flat``. ``W^flat`` is now
    materialized by the shared ``typed-barycentre`` stage under
    ``data/shared/barycentres/<model>/<model>-unit/energy_simplex`` and consumed
    directly by ``paper5_empirical``; this validation-only comparison continues
    to exercise ``W^h`` because it is exactly reproducible from the energy
    distances already loaded here. Both operators are row-stochastic with a
    zero diagonal, so this path exercises the same ``{1, W1, W^2 1}``
    GMM-instrument degeneracy documented in ``estimate.py``'s ``sar_gmm``
    docstring.

    Missing, malformed, or mismatched inputs raise explicitly; validation must
    never turn an absent empirical comparison into a successful skipped run.
    """
    returns_dir = Path("data/shared/returns")
    distance_artifact_dir = Path(
        "data/shared/typed_distances/qwen3-embedding-8b/"
        "qwen3-embedding-8b-unit/energy_v"
    )
    if not returns_dir.is_dir():
        message = f"Empirical returns directory not found: {returns_dir}"
        raise FileNotFoundError(message)
    read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity={
            "provider_id": "qwen3-embedding-8b",
            "representation_id": "qwen3-embedding-8b-unit",
            "distance_id": "energy_v",
        },
    )
    tickers, _d, d2 = load_priced_distance_universe(distance_artifact_dir, returns_dir)
    w_h, _bandwidth_h = build_w_kernel(d2, tickers)
    _dates, panel = load_aligned_return_panel(
        returns_dir, tickers, "2018-01-01", "2022-12-31"
    )

    return _compare_one_from_panel(panel, w_h)


def _compare_one_from_panel(r: Float, w: Float) -> dict[str, Any]:
    qmle = sar_qmle(r, w)
    gmm = sar_gmm(r, w)
    ref = _spreg_estimates(r, w)
    return {
        "qmle_rho_hat": qmle["rho_hat"],
        "gmm_rho_hat": gmm["rho_hat"],
        "ml_lag_rho": ref["ml_lag_rho"],
        "gm_lag_rho": ref["gm_lag_rho"],
        "gm_lag_singular_instruments": ref["gm_lag_singular_instruments"],
        "qmle_vs_ml_lag_diff": qmle["rho_hat"] - ref["ml_lag_rho"],
        "gmm_vs_gm_lag_diff": (
            gmm["rho_hat"] - ref["gm_lag_rho"]
            if ref["gm_lag_rho"] is not None
            else None
        ),
    }


def main(
    *, smoke: bool = False, output_dir: Path | str = Path("data/papers/paper5")
) -> dict[str, Any]:
    """Run the synthetic-grid (and, if not ``smoke``, empirical) comparison.

    Args:
        smoke: If ``True``, run only a single tiny synthetic case (no
            empirical comparison, no full grid) to confirm imports resolve
            and the comparison pipeline executes.
        output_dir: Directory for the JSON summary (``data/papers/paper5``).

    Returns:
        Summary dict written to ``<output_dir>/validate_estimators_summary.json``.

    """
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(output_dir)

    with threadpool_limits(limits=runtime_settings().blas_limit):
        synthetic = run_synthetic_grid(smoke=smoke)
        empirical = None if smoke else run_empirical_comparison()

    summary = {
        "smoke": smoke,
        "synthetic": synthetic,
        "empirical": empirical,
        "gmm_boundary_clip_events": [
            row["rho_true"] for row in synthetic if row["gmm_boundary_clip_flag"]
        ],
    }

    if not smoke:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "validate_estimators_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Wrote %s", output_dir / "validate_estimators_summary.json")
    else:
        logger.info("Smoke run complete: %s", summary)

    return summary


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "run_empirical_comparison",
    "run_synthetic_grid",
]
