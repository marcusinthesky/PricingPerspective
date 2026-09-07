"""Stable facade for covariance estimators, shrinkage, and cross-validation.

Implementation owners are split by concern under :mod:`jcor.model._covariance`.
All public objects below are direct re-exports, so existing project-facing
imports retain object identity.
"""

from jcor.model._covariance.cross_validation import cv_lambda_grid
from jcor.model._covariance.estimators import (
    LedoitWolfResult,
    ledoit_wolf_sample,
    ledoit_wolf_sample_kernel,
    sample_covariance,
    sample_covariance_kernel,
)
from jcor.model._covariance.shrinkage import (
    ShrinkageResult,
    optimal_lambda_dist,
    plug_in_lambda_dist,
    plug_in_lambda_dist_kernel,
    sample_cov_entry_variance,
    sample_cov_entry_variance_kernel,
    shrink_to_dist,
    shrink_to_dist_kernel,
    target_weight_dist,
)

# Preserve qualified names used by repr/pickle as well as import identity.
LedoitWolfResult.__module__ = __name__
ShrinkageResult.__module__ = __name__
cv_lambda_grid.__module__ = __name__
ledoit_wolf_sample.__module__ = __name__
ledoit_wolf_sample_kernel.__module__ = __name__
optimal_lambda_dist.__module__ = __name__
plug_in_lambda_dist.__module__ = __name__
plug_in_lambda_dist_kernel.__module__ = __name__
sample_cov_entry_variance.__module__ = __name__
sample_cov_entry_variance_kernel.__module__ = __name__
sample_covariance.__module__ = __name__
sample_covariance_kernel.__module__ = __name__
shrink_to_dist.__module__ = __name__
shrink_to_dist_kernel.__module__ = __name__
target_weight_dist.__module__ = __name__

__all__ = [
    "LedoitWolfResult",
    "ShrinkageResult",
    "cv_lambda_grid",
    "ledoit_wolf_sample",
    "ledoit_wolf_sample_kernel",
    "optimal_lambda_dist",
    "plug_in_lambda_dist",
    "plug_in_lambda_dist_kernel",
    "sample_cov_entry_variance",
    "sample_cov_entry_variance_kernel",
    "sample_covariance",
    "sample_covariance_kernel",
    "shrink_to_dist",
    "shrink_to_dist_kernel",
    "target_weight_dist",
]
