"""Paper 3 empirical driver: distance-implied covariance → minimum-variance.

Runs the validated Paper-3 machinery (:mod:`jcor.model.covariance`,
:mod:`jcor.operators.covariance`, :mod:`jcor.optimize.psd`,
:mod:`jcor.inference`, :mod:`jcor.operators.longrun`,
:mod:`simulation.optimize` / :mod:`simulation.portfolios`) on the real
100-ticker daily-return panel (2018–2022), the Qwen3-embedding Wasserstein-2
matrix, and the sample covariance.

Pipeline (mission steps 1–4):

1. Build ``Σ_dist`` from squared W2 + per-asset return variances under the
   scalar exact-transmission, zero-transport-excess benchmark; the explicitly
   heuristic same-window plug-in blend toward it; the nearest-covariance
   repair; and repair diagnostics.
2. Envelope-saturation tests on the panel — one internally profiled Hansen-J/GMM
   over-identification statistic (Hall-centered HAC), descriptive profiled GEL,
   the Wolak inequality test (pairs clustered to portfolios, MC χ̄² weights),
   uncorrected indicator-GMS diagnostics, and κ-inference (delta method plus a
   descriptive moment-Jacobian diagnostic).
3. Look-ahead-safe out-of-sample minimum-variance backtest comparing Σ_dist,
   Σ(λ_t), sample covariance and Ledoit–Wolf over a window sweep, with
   net-of-transaction-cost Sharpe, realized volatility, turnover, condition
   number, effective number of assets and HHI, plus block-bootstrap CIs.
4. Emit ``data/papers/paper3/empirical/{results.parquet, summary.yaml}``.

DISCLOSURE. Lean proves the Hilbert/W2 Fréchet ceiling and the nonnegative
transport-excess gap under their stated hypotheses. The empirical target
applies those results to observed article laws through a disclosed scalar
exact-transmission benchmark and sets transport excess to zero; those two
identifying restrictions are tested, not machine-checked facts about the data.
Universe: 100 tickers (no exclusions — every firm in the news universe is
series).

The step implementations live in the private :mod:`.._empirical` package (t42);
this module owns the ``run_paper3_empirical`` entry point and re-exports the
contract and helper surface that Paper 5, the ablation stage, the CLI, and the
test suite address by name.
"""

from __future__ import annotations

from threadpoolctl import threadpool_limits

from pipeline.io.settings import runtime_settings
from pipeline.jax_cache import configure_persistent_cache
from pipeline.stages.papers.paper3._empirical.backtest import backtest
from pipeline.stages.papers.paper3._empirical.backtest_cells import (
    SUB_PERIODS,
    _load_pit_matrices,
)
from pipeline.stages.papers.paper3._empirical.contracts import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    SIGNIFICANCE_LEVEL,
    BacktestConfig,
    BacktestInputs,
    Float,
    IdentityTestConfig,
    IdentityTestInputs,
    Paper3EmpiricalConfig,
    Paper3EmpiricalError,
    Paper3EmpiricalPaths,
)
from pipeline.stages.papers.paper3._empirical.driver import _run_paper3_empirical_impl
from pipeline.stages.papers.paper3._empirical.identity import (
    _pair_moment_process,
    identity_tests,
)
from pipeline.stages.papers.paper3._empirical.manifest import (
    _sha256_path,
    _write_paper3_manifest,
)
from pipeline.stages.papers.paper3._empirical.optimize import (
    _cov_estimators,
    _jit_pgd,
    _mv_weights,
    _portfolio_diagnostics,
)
from pipeline.stages.papers.paper3._empirical.panel import (
    build_sigma_dist,
    calibrate_kappa,
    load_panel,
)
from pipeline.stages.papers.paper3._empirical.robustness import robustness_gel_gms
from pipeline.stages.papers.paper3._empirical.sharpe import _bootstrap_sharpe_ci

configure_persistent_cache()

# Public contract plus the private helpers Paper 5, the model-robustness stage, the
# CLI, and the test suite import from this module path (t42). Dropping any of
# these silently breaks a cross-reference no single test covers.
__all__ = [
    "MIN_BOOTSTRAP_OBSERVATIONS",
    "SIGNIFICANCE_LEVEL",
    "SUB_PERIODS",
    "BacktestConfig",
    "BacktestInputs",
    "Float",
    "IdentityTestConfig",
    "IdentityTestInputs",
    "Paper3EmpiricalConfig",
    "Paper3EmpiricalError",
    "Paper3EmpiricalPaths",
    "_bootstrap_sharpe_ci",
    "_cov_estimators",
    "_jit_pgd",
    "_load_pit_matrices",
    "_mv_weights",
    "_pair_moment_process",
    "_portfolio_diagnostics",
    "_sha256_path",
    "_write_paper3_manifest",
    "backtest",
    "build_sigma_dist",
    "calibrate_kappa",
    "identity_tests",
    "load_panel",
    "robustness_gel_gms",
    "run_paper3_empirical",
]


def run_paper3_empirical(
    paths: Paper3EmpiricalPaths,
    config: Paper3EmpiricalConfig | None = None,
) -> None:
    """End-to-end Paper-3 empirical run; writes results.parquet + summary.yaml.

    Args:
        paths: Input and output artifact paths.
        config: Window, clustering, and simulation controls.

    """
    config = Paper3EmpiricalConfig() if config is None else config
    # 2026-07-18 profile: 89% of this stage's wall was
    # higham_psd_repair -> higham_correlation_repair (simulation/psd_repair.py),
    # called from `_empirical.panel.build_sigma_dist` and
    # `_empirical.optimize._cov_estimators` — 2537 calls x ~200
    # alternating-projection iterations (max_iter cap reached) x numpy eigh on
    # 52x52, i.e. ~512k tiny eigh calls.
    #
    # t65.4 REPLACED that repair with `jcor.optimize.psd.nearest_covariance`
    # (Qi-Sun semismooth Newton), so the "~200 iterations" term is now 8-11
    # Newton steps. Note what this profile independently established, and which
    # motivated the replacement: "max_iter cap reached" on all 2537 calls means
    # the 1e-10 tolerance was NEVER met in a production run, so every repaired
    # matrix the stage emitted was a nonconverged iterate rather than the
    # nearest correlation matrix its docstring described.
    #
    # BLAS thread-pool oversubscription on those tiny matrices was the
    # bottleneck: capping to PP_BLAS_LIMIT threads
    # (default 2) took the stage from 1149.6s (uncapped) to 284.5s (~4x),
    # with the capped run's outputs verified BIT-IDENTICAL to the canonical
    # repro (summary.yaml, robustness_gel_gms.yaml diff-empty; results.parquet
    # frame-equal). ``PipelineRuntimeSettings`` validates the limit; zero opts
    # out. The scope covers the whole call so the model-robustness driver's in-process
    # calls into this module benefit too.
    with threadpool_limits(limits=runtime_settings().blas_limit):
        _run_paper3_empirical_impl(paths, config)
