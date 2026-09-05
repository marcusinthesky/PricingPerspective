"""Paired Sharpe-difference testing and block-bootstrap Sharpe CIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import numpy as np
from jcor.decision.sharpe import sharpe_difference_test
from jcor.inference.block_length import optimal_block_length_numpy
from jcor.operators.longrun import circular_block_bootstrap_indices

from pipeline.stages.papers.paper3._empirical.contracts import (
    MIN_BOOTSTRAP_OBSERVATIONS,
)
from pipeline.stages.papers.paper3._empirical.optimize import (
    _cov_estimators,
    _mv_weights,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper3._empirical.contracts import (
        Float,
    )


def _paired_sharpe_diff_summary(
    returns: Float,
    squared_distances: Float,
    n_boot: int = 500,
    seed: int = 0,
    windows: tuple[int, ...] = (10, 60),
) -> dict[str, object]:
    """Paired Sharpe-difference test (Σ_dist vs sample) at headline windows.

    Runs a light-weight standalone re-derivation (long-only, 10bps net) of the
    per-day net-return series for ``sample`` and ``sigma_dist`` at each window
    in ``windows`` and calls :func:`jcor.operators.longrun.sharpe_difference_test`.

    Args:
        returns: Return panel, shape ``(T, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.
        n_boot: Bootstrap resamples for the test.
        seed: RNG seed.
        windows: Estimation windows to test (default headline 10/60).

    Returns:
        Dict keyed ``"window_{T}"`` -> ``{"stat": float, "pvalue": float}``.

    """
    n_observations, _n = returns.shape
    out: dict[str, object] = {}
    for win in windows:
        if win >= n_observations:
            continue
        rebal = max(1, win // 4)
        net: dict[str, list[float]] = {"sample": [], "sigma_dist": []}
        held: dict[str, Float | None] = {"sample": None, "sigma_dist": None}
        for t in range(win, n_observations):
            r_next = returns[t]
            if (t - win) % rebal == 0:
                train = returns[t - win : t]
                estims = _cov_estimators(train, squared_distances)
                for m in ("sample", "sigma_dist"):
                    held[m] = _mv_weights(cast("Float", estims[m]), long_only=True)
            for m in ("sample", "sigma_dist"):
                w = held[m]
                ret = float(r_next @ w) - (10.0 / 1e4) * 0.0
                net[m].append(ret)
                grown = w * (1.0 + r_next)
                held[m] = grown / grown.sum()
        stat, pvalue = sharpe_difference_test(
            np.asarray(net["sigma_dist"]),
            np.asarray(net["sample"]),
            n_boot=n_boot,
            seed=seed,
        )
        out[f"window_{win}"] = {"stat": stat, "pvalue": pvalue}
    return out


def _bootstrap_sharpe_ci(
    net: Float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Circular-block-bootstrap 95% CI for the (non-annualized) Sharpe.

    Args:
        net: Net-of-cost return series, shape ``(T,)``.
        n_boot: Number of resamples.
        seed: RNG seed.

    Returns:
        Tuple ``(lo, hi)`` — 2.5%/97.5% Sharpe quantiles (non-annualized).

    """
    n_observations = net.size
    if n_observations < MIN_BOOTSTRAP_OBSERVATIONS:
        return 0.0, 0.0
    # The block-length estimator is a float64 NumPy host door; materialize the
    # JAX index result before using it to gather the NumPy return series.
    with jax.enable_x64(new_val=True):
        bl_float = float(optimal_block_length_numpy(net, scheme="circular"))
        bl = max(1, round(bl_float))
        idx = circular_block_bootstrap_indices(n_observations, bl, n_boot, seed=seed)
    boot = net[np.asarray(idx)]  # (n_boot, T)
    mu = boot.mean(axis=1)
    sd = boot.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(sd > 0, mu / sd, 0.0)
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))
