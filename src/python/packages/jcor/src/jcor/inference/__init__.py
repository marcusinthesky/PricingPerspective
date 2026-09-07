"""S7 — null distributions: statistic -> p-value.

Position
--------
rank 7

Consumes
--------
a statistic and its data-generating context

Produces
--------
a ``TestResult``

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

from jcor.inference.calibration import add_one_p
from jcor.inference.disco import (
    DiscoPermutationOrigin,
    DiscoPermutationResult,
    disco_permutation_null,
    disco_permutation_test,
)

__all__ += [
    "ChiBarWeightsKernelResult",
    "DiscoPermutationOrigin",
    "DiscoPermutationResult",
    "add_one_p",
    "disco_permutation_null",
    "disco_permutation_test",
]

# --- t46 migration re-exports; each task appends only to its own region ---

# --- t46.8 ---
from jcor.inference.block_length import (  # noqa: E402
    BlockLengthScheme,
    heuristic_block_length,
    optimal_block_length,
    optimal_block_length_numpy,
    stationary_bootstrap_ci,
)

__all__ += [
    "BlockLengthScheme",
    "heuristic_block_length",
    "optimal_block_length",
    "optimal_block_length_numpy",
    "stationary_bootstrap_ci",
]
# --- end t46.8 ---

# --- t56.2 shared JAX primitives ---
from jcor.inference.block_length import (  # noqa: E402
    heuristic_block_length_result,
    optimal_block_length_result,
    stationary_bootstrap_estimate,
)

__all__ += [
    "heuristic_block_length_result",
    "optimal_block_length_result",
    "stationary_bootstrap_estimate",
]
# --- end t56.2 ---

# --- t46.7 ---
# `HypothesisTest` / `PermutationTestResult` are deliberately NOT re-exported:
# they are superseded by `jcor.core.results.TestResult` and reachable only via
# `jcor.inference._results` and the `jcor.hypothesis` shim until t46.10 deletes
# them. `MixtureTestResult` IS exported — it is a live procedure-specific
# container, not a legacy one.
from jcor.inference.bootstrap import bootstrap_confidence_interval  # noqa: E402
from jcor.inference.mixture import (  # noqa: E402
    MixtureTestResult,
    corrected_mixture_test,
)
from jcor.inference.permutation import (  # noqa: E402
    EnergyPermutationOrigin,
    EnergyPermutationResult,
    EnergyStatisticOrigin,
    energy_permutation_test,
    energy_permutation_test_result,
    permutation_test,
    permutation_test_kernel,
    permutation_test_result,
)
from jcor.inference.studentized import studentized_mixture_statistic  # noqa: E402

__all__ += [
    "EnergyPermutationOrigin",
    "EnergyPermutationResult",
    "EnergyStatisticOrigin",
    "MixtureTestResult",
    "bootstrap_confidence_interval",
    "corrected_mixture_test",
    "energy_permutation_test",
    "energy_permutation_test_result",
    "permutation_test",
    "permutation_test_kernel",
    "permutation_test_result",
    "studentized_mixture_statistic",
]
# --- end t46.7 ---

# --- generic moment-condition inference ---
from jcor.inference.clustering import cluster_pairs  # noqa: E402
from jcor.inference.gel import (  # noqa: E402
    GelResult,
    LinearGelResult,
    gel_test,
    linear_gel_test,
)
from jcor.inference.gmm import (  # noqa: E402
    HansenJResult,
    LinearHansenJResult,
    WaldResult,
    gmm_wald,
    gmm_wald_kernel,
    hansen_j,
    hansen_j_kernel,
    linear_hansen_j,
    linear_hansen_j_kernel,
)
from jcor.inference.gms import (  # noqa: E402
    GmsResult,
    andrews_barwick_gms,
    indicator_gms,
)
from jcor.inference.kappa import (  # noqa: E402
    KappaDeltaResult,
    MomentJacobianResult,
    boundary_robust_pvalue,
    boundary_robust_pvalue_kernel,
    kappa_delta_kernel,
    kappa_delta_method,
    moment_jacobian_concentration,
    moment_jacobian_concentration_kernel,
)
from jcor.inference.wolak import (  # noqa: E402
    ChiBarWeightsKernelResult,
    WolakKernelResult,
    WolakResult,
    chi_bar_squared_sf,
    chi_bar_squared_sf_kernel,
    chi_bar_squared_weights_mc,
    chi_bar_squared_weights_mc_kernel,
    wolak_statistic,
    wolak_test,
    wolak_test_kernel,
)

__all__ += [
    "GelResult",
    "GmsResult",
    "HansenJResult",
    "KappaDeltaResult",
    "LinearGelResult",
    "LinearHansenJResult",
    "MomentJacobianResult",
    "WaldResult",
    "WolakKernelResult",
    "WolakResult",
    "andrews_barwick_gms",
    "boundary_robust_pvalue",
    "boundary_robust_pvalue_kernel",
    "chi_bar_squared_sf",
    "chi_bar_squared_sf_kernel",
    "chi_bar_squared_weights_mc",
    "chi_bar_squared_weights_mc_kernel",
    "cluster_pairs",
    "gel_test",
    "gmm_wald",
    "gmm_wald_kernel",
    "hansen_j",
    "hansen_j_kernel",
    "indicator_gms",
    "kappa_delta_kernel",
    "kappa_delta_method",
    "linear_gel_test",
    "linear_hansen_j",
    "linear_hansen_j_kernel",
    "moment_jacobian_concentration",
    "moment_jacobian_concentration_kernel",
    "wolak_statistic",
    "wolak_test",
    "wolak_test_kernel",
]
# --- end generic moment-condition inference ---
