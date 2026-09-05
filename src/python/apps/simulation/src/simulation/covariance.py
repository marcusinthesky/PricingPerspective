"""Compatibility facade for covariance construction and estimation in jcor.

All public objects are direct re-exports so historical ``simulation.covariance``
imports retain identity. New production code should import constructors from
:mod:`jcor.operators` and estimators from :mod:`jcor.model`.
"""

from jcor.model.covariance import (
    LedoitWolfResult,
    ShrinkageResult,
    cv_lambda_grid,
    ledoit_wolf_sample,
    optimal_lambda_dist,
    plug_in_lambda_dist,
    sample_cov_entry_variance,
    sample_covariance,
    shrink_to_dist,
    target_weight_dist,
)
from jcor.operators.covariance import sigma_dist, sigma_dist_hetero_kappa

__all__ = [
    "LedoitWolfResult",
    "ShrinkageResult",
    "cv_lambda_grid",
    "ledoit_wolf_sample",
    "optimal_lambda_dist",
    "plug_in_lambda_dist",
    "sample_cov_entry_variance",
    "sample_covariance",
    "shrink_to_dist",
    "sigma_dist",
    "sigma_dist_hetero_kappa",
    "target_weight_dist",
]
