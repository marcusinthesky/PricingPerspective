"""Paper 5 energy-robustness empirical driver.

Executes the empirics lane of
Paper 5's H3 energy-robustness design uses Sigma_dist
as the CENTER of a statistically calibrated ambiguity set
``Box[Sigma_lo, Sigma_hi] cap PSD cap {||Sigma - Sigma_hat_T||_F <= r(T)}``.
Reuses Paper 3's data loading, kappa calibration, Sigma_dist construction,
Higham repair and PGD min-variance solver
(:mod:`pipeline.stages.papers.paper3.empirical`, :mod:`simulation.optimize`,
:mod:`jcor.optimize.psd`); the new radius calibration and bracket assembly live
in :mod:`pipeline.stages.papers.paper5.energy_radius`.

Two bracket modes per window (see
:func:`pipeline.stages.papers.paper5.energy_radius.assemble_brackets`):
CALIBRATED (``kappa_lo = kappa_hi = kappa_hat``, GMM ``D^2`` correction) and
CERTIFIED (Lean-T2-faithful envelope ``[ell_lo, L_hat]``, no GMM). Strategies
(backtest matrix): ``sample``, ``ledoit_wolf``, ``sigma_dist`` (point),
``robust_corner`` (min_w w'Sigma_hi w on the CALIBRATED bracket),
``higham_corner`` (same, but Sigma_hi Higham-repaired to PSD first -- the
"heuristic" per the plan's optimizer ladder), ``robust_corner_cert`` (corner
on the CERTIFIED bracket), ``cert_stale_vol`` (the "price-light" strategy:
CERTIFIED corner built from calibration-epoch diagonal variance ONLY --
window-invariant, needs no evaluation-epoch price data to form weights; see
:func:`_cov_and_weights`), ``robust_ridge`` (min_w w'Sigma_hat w +
r(T)||w||^2), ``robust_select`` (whichever of calibrated-corner/ridge has the
smaller certified cap), ``equal_weight``.

SCALE / HONESTY. All structural inputs (``kappa_hat``, eps^stat, eps^GMM,
``r(T)``, and the certified envelope) are calibrated once on the 2018--2022
epoch and held fixed across every backtest window. Window return covariances
remain causal: they use only returns preceding each rebalance. Evaluation uses
the separate governed post-2022 return panel. Exact run scales (bootstrap B,
``n_boot``, ``n_clusters``) are recorded in ``RUN_MANIFEST.md`` by
:func:`run_paper5_energy_robustness`.

The step implementations live in the private :mod:`._energy_robust` package;
this module is the addressed surface -- the ``run_paper5_energy_robustness``
entry point plus the contracts, optimizers, and helpers used by the Paper 5 CLI
and test suite.
"""

from __future__ import annotations

from pipeline.jax_cache import configure_persistent_cache
from pipeline.stages.papers.paper5._energy_robust.backtest import backtest
from pipeline.stages.papers.paper5._energy_robust.contracts import (
    MAX_MATERIALITY_RATIO,
    STRATEGIES,
    Float,
    Paper5EnergyRobustnessConfig,
    Paper5EnergyRobustnessPaths,
    RobustBacktestConfig,
    RobustBacktestInputs,
    RobustCovarianceInputs,
    RobustOptimizationRequest,
    RobustPortfolioComputationError,
    RobustPortfolioError,
)
from pipeline.stages.papers.paper5._energy_robust.covariance import _cov_and_weights
from pipeline.stages.papers.paper5._energy_robust.coverage import (
    pairwise_covariance_coverage,
)
from pipeline.stages.papers.paper5._energy_robust.driver import (
    run_paper5_energy_robustness,
)
from pipeline.stages.papers.paper5._energy_robust.optimize import (
    _jit_pgd,
    _jit_wdro,
    _pgd_weights,
    higham_corner_weights,
    robust_corner_weights,
    robust_ridge_weights,
    sdp_corner_weights,
    wdro_weights,
)
from pipeline.stages.papers.paper5._energy_robust.weights import (
    certified_caps,
    robust_weights,
)

configure_persistent_cache()

# The facade must import every name it re-exports,
# so `Float`, `MAX_MATERIALITY_RATIO`, the two exception classes, and the four
# private helpers used by the energy-robustness tests appear below
# too (t44). Every one of them was reachable as a module attribute before the
# split; this widens `__all__`, not the namespace.
__all__ = [
    "MAX_MATERIALITY_RATIO",
    "STRATEGIES",
    "Float",
    "Paper5EnergyRobustnessConfig",
    "Paper5EnergyRobustnessPaths",
    "RobustBacktestConfig",
    "RobustBacktestInputs",
    "RobustCovarianceInputs",
    "RobustOptimizationRequest",
    "RobustPortfolioComputationError",
    "RobustPortfolioError",
    "_cov_and_weights",
    "_jit_pgd",
    "_jit_wdro",
    "_pgd_weights",
    "backtest",
    "certified_caps",
    "higham_corner_weights",
    "pairwise_covariance_coverage",
    "robust_corner_weights",
    "robust_ridge_weights",
    "robust_weights",
    "run_paper5_energy_robustness",
    "sdp_corner_weights",
    "wdro_weights",
]
