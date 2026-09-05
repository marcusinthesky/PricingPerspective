"""H1 bracket coverage on realized covariance, with a block-bootstrap CI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jcor.model.covariance import sample_covariance

from pipeline._kernels.bootstrap import bootstrap_indices

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import Float


def pairwise_covariance_coverage(
    sigma_lo: Float, sigma_hi: Float, s_realized: Float
) -> float:
    """Return off-diagonal realized-covariance bracket coverage."""
    n = sigma_lo.shape[0]
    iu = np.triu_indices(n, k=1)
    lo, hi, s = sigma_lo[iu], sigma_hi[iu], s_realized[iu]
    covered = (s >= lo) & (s <= hi)
    return float(np.mean(covered))


def _coverage_block_bootstrap_ci(
    sigma_lo: Float,
    sigma_hi: Float,
    r_eval: Float,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Point coverage + circular-block-bootstrap CI of pairwise coverage.

    Resamples the evaluation return panel with circular blocks
    (Politis-White automatic length on the equal-weight portfolio, matching
    ``h1_oos_coverage._resample_test_panel``), recomputes the realized
    covariance and the pairwise coverage per resample, and reports the
    2.5/97.5% quantiles. Folds in what was previously a one-off script
    (recorded manifest gap).
    """
    idx, block_length = bootstrap_indices(r_eval, n_boot, seed)
    covs = np.empty(n_boot)
    for b in range(n_boot):
        s_b = np.asarray(sample_covariance(r_eval[np.asarray(idx[b])], ddof=1))
        covs[b] = pairwise_covariance_coverage(sigma_lo, sigma_hi, s_b)
    point = pairwise_covariance_coverage(
        sigma_lo, sigma_hi, np.asarray(sample_covariance(r_eval, ddof=1))
    )
    return {
        "point_coverage": float(point),
        "block_length": int(block_length),
        "n_boot": int(n_boot),
        "ci_lo": float(np.quantile(covs, 0.025)),
        "ci_hi": float(np.quantile(covs, 0.975)),
    }
