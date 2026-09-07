"""Return-block bootstrap and Wolak inference for the H1 frontier."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from jcor.inference import (
    chi_bar_squared_sf,
    chi_bar_squared_weights_mc,
    cluster_pairs,
    wolak_statistic,
)

from pipeline._kernels.bootstrap import bootstrap_indices
from pipeline.stages.substrate.energy_shared import _cluster_assignments
from pipeline.stages.substrate.w2_exposure_frontier.frontier import _weighted_quantile
from pipeline.stages.substrate.w2_exposure_frontier.returns_loadings import (
    estimate_loadings,
)

if TYPE_CHECKING:
    from pipeline.stages.substrate.w2_exposure_frontier.contracts import (
        FrontierBootstrapConfig,
    )

# One-sided z_{1-alpha} (alpha=0.05) and z_{power} (power=0.80), used for the
# MDE = (z_alpha + z_power) * SE(ell_hat) reported alongside wolak_stat.
_Z_ALPHA_ONE_SIDED_05 = 1.6448536269514722
_Z_POWER_80 = 0.8416212335729143


class BootstrapWolakResult(NamedTuple):
    """Block-bootstrap CI for ell_hat plus the externally-anchored Wolak test.

    Attributes:
        ell_hat: Point estimate of the lower Lipschitz constant.
        ell_lo: 2.5% bootstrap percentile of ell_hat (the conservative floor
            used to drive the kill and the cap).
        ell_hi: 97.5% bootstrap percentile of ell_hat.
        ell_se: Bootstrap standard deviation of ell_hat.
        wolak_stat: The pooled chi-bar-squared statistic testing the moment
            inequalities m_p = ||beta_i - beta_j|| - wolak_ell0 * D_ij >= 0
            against the EXTERNALLY ANCHORED ``wolak_ell0`` (F4 fix -- not the
            self-fitted ell_lo).
        wolak_p: The chi-bar-squared p-value (large p supports the frontier
            premise: the lower-Lipschitz inequalities at wolak_ell0 are not
            violated).
        wolak_ell0: The external anchor tested (see module-level
            ``DEFAULT_WOLAK_ELL0`` TODO).
        mde: Minimum detectable effect, ``(z_alpha + z_power) * ell_se``
            (one-sided alpha=0.05, power=0.80) -- the smallest true ell the
            bootstrap has power to distinguish from 0.
        n_clusters_eff: Number of non-empty pair-clusters used.
        block_length: Circular block length used for the return-block
            bootstrap (F6 fix -- resamples TIME blocks of the return panel,
            re-estimating beta_hat per replicate; not embedding rows).

    """

    ell_hat: float
    ell_lo: float
    ell_hi: float
    ell_se: float
    wolak_stat: float
    wolak_p: float
    wolak_ell0: float
    mde: float
    n_clusters_eff: int
    block_length: int


def frontier_bootstrap_and_wolak(
    returns_train: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    config: FrontierBootstrapConfig,
) -> BootstrapWolakResult:
    """Bootstrap ell_hat and run the externally anchored Wolak test.

    The resampling unit is TIME: a circular block bootstrap
    (``jcor.operators.longrun.circular_block_bootstrap_indices``,
    Politis-Romano) over the ``(T, N)`` train return panel. Each replicate
    resamples time blocks, re-estimates whitened-PCA loadings via
    ``pipeline.returns_loadings.estimate_loadings`` on the resampled panel,
    and recomputes ell_hat on the resulting beta_hat -- ``D`` (text distance)
    is held fixed throughout (it is the conditioning variable, plan
    Requirement 7).

    The Wolak test evaluates the moment inequalities
    m_p = ||beta_i - beta_j|| - wolak_ell0 * D_ij >= 0 pooled into
    pair-clusters (``pipeline.paper3_empirical._cluster_assignments`` +
    ``jcor.inference.cluster_pairs``) against the externally-anchored
    ``wolak_ell0`` -- NOT the self-fitted ell_lo (F4 fix: the old
    ``ell_tested = max(ell_lo, 0)`` tested the bootstrap distribution against
    its own boundary and trivially produced ~1e-28 statistics).

    Args:
        returns_train: ``(T, N)`` train-window daily return panel (no NaNs),
            column order matching ``d``/``d2``.
        d: (n, n) Hilbertian distance matrix.
        d2: (n, n) raw energy-distance matrix (for pair clustering).
        config: Factor count, quantile, resampling, inference, and RNG
            controls. ``block_length=None`` selects a block length via
            ``jcor.inference.block_length.optimal_block_length_numpy`` on the
            equal-weighted portfolio return series (Politis-White, matching
            the convention in ``pipeline.paper3_empirical._bootstrap_sharpe_ci``).

    Returns:
        A :class:`BootstrapWolakResult`.

    """
    _t_obs, n = returns_train.shape
    iu, ju = np.triu_indices(n, k=1)
    d_pairs = d[iu, ju]

    point_betas = estimate_loadings(returns_train, config.k).betas
    y_pt = np.linalg.norm(point_betas[iu] - point_betas[ju], axis=1)
    valid = d_pairs > 0.0
    ratios_pt = y_pt[valid] / d_pairs[valid]
    ell_hat = _weighted_quantile(ratios_pt, d_pairs[valid], config.tau_lo)

    assign = _cluster_assignments(d2, config.n_clusters, config.seed)

    boot_idx, block_length = bootstrap_indices(
        returns_train, config.n_boot, config.seed, config.block_length
    )

    ell_boot = np.empty(config.n_boot, dtype=np.float64)
    y_clustered_rows: list[np.ndarray] = []
    for b in range(config.n_boot):
        resampled_returns = returns_train[boot_idx[b]]
        resampled_betas = estimate_loadings(resampled_returns, config.k).betas
        y_b = np.linalg.norm(resampled_betas[iu] - resampled_betas[ju], axis=1)
        ell_boot[b] = _weighted_quantile(
            y_b[valid] / d_pairs[valid], d_pairs[valid], config.tau_lo
        )
        clustered = cluster_pairs(y_b[None, :], assign, config.n_clusters)
        # `cluster_pairs` returns a `jax.Array` as of t65.1. This is the host
        # boundary: everything below is NumPy (`vstack`, `cov`), so materialize
        # here rather than letting a device array propagate.
        y_clustered_rows.append(np.asarray(clustered[0]))

    y_clustered = np.vstack(y_clustered_rows)
    s_boot = np.atleast_2d(np.cov(y_clustered.T))
    n_clusters_eff = s_boot.shape[0]

    ell_lo, ell_hi = np.percentile(ell_boot, [2.5, 97.5])
    ell_se = float(np.std(ell_boot, ddof=1))
    mde = (_Z_ALPHA_ONE_SIDED_05 + _Z_POWER_80) * ell_se

    ybar = cluster_pairs(y_pt[None, :], assign, config.n_clusters)[0]
    dbar = cluster_pairs(d_pairs[None, :], assign, config.n_clusters)[0]
    gbar = ybar - config.wolak_ell0 * dbar

    stat, _ = wolak_statistic(gbar, s_boot, 1)
    weights = chi_bar_squared_weights_mc(s_boot, config.n_mc, config.seed)
    pval = chi_bar_squared_sf(stat, weights)

    return BootstrapWolakResult(
        ell_hat=ell_hat,
        ell_lo=float(ell_lo),
        ell_hi=float(ell_hi),
        ell_se=ell_se,
        wolak_stat=float(stat),
        wolak_p=float(pval),
        wolak_ell0=float(config.wolak_ell0),
        mde=float(mde),
        n_clusters_eff=n_clusters_eff,
        block_length=int(block_length),
    )
