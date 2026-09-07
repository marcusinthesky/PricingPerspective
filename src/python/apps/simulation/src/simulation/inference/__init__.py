"""Compatibility facade for generic inference now owned by :mod:`jcor`.

All public objects are direct re-exports, so historical
``simulation.inference`` imports retain object identity. New production code
should import from :mod:`jcor.inference`.
"""

from __future__ import annotations

from jcor.inference import (
    GelResult,
    GmsResult,
    HansenJResult,
    KappaDeltaResult,
    LinearGelResult,
    LinearHansenJResult,
    MomentJacobianResult,
    WaldResult,
    WolakResult,
    andrews_barwick_gms,
    boundary_robust_pvalue,
    chi_bar_squared_sf,
    chi_bar_squared_weights_mc,
    cluster_pairs,
    gel_test,
    gmm_wald,
    hansen_j,
    indicator_gms,
    kappa_delta_method,
    linear_gel_test,
    linear_hansen_j,
    moment_jacobian_concentration,
    wolak_statistic,
    wolak_test,
)

__all__ = [
    "GelResult",
    "GmsResult",
    "HansenJResult",
    "KappaDeltaResult",
    "LinearGelResult",
    "LinearHansenJResult",
    "MomentJacobianResult",
    "WaldResult",
    "WolakResult",
    "andrews_barwick_gms",
    "boundary_robust_pvalue",
    "chi_bar_squared_sf",
    "chi_bar_squared_weights_mc",
    "cluster_pairs",
    "gel_test",
    "gmm_wald",
    "hansen_j",
    "indicator_gms",
    "kappa_delta_method",
    "linear_gel_test",
    "linear_hansen_j",
    "moment_jacobian_concentration",
    "wolak_statistic",
    "wolak_test",
]
