"""Step 2: tests of the zero-transport-excess W2 covariance benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from jcor.inference import (
    cluster_pairs,
    kappa_delta_method,
    linear_hansen_j,
    moment_jacobian_concentration,
    wolak_test,
)

from pipeline.stages.papers.paper3._empirical.contracts import (
    IdentityTestConfig,
    Paper3EmpiricalError,
)
from pipeline.stages.substrate.energy_shared import _cluster_assignments

if TYPE_CHECKING:
    from pipeline.stages.papers.paper3._empirical.contracts import (
        Float,
        IdentityTestInputs,
    )

_MIN_MOMENT_OBSERVATIONS = 2


def _pair_moment_process(
    returns: Float,
    squared_distances: Float,
    kappa: float,
) -> tuple[Float, Float]:
    """Per-period moments for saturation of the pairwise W2 covariance ceiling.

    For demeaned returns ``u`` and ``c_T=T/(T-1)``, the pair contribution is
    ``g_t = c_T[u_it u_jt - (u_it²+u_jt²)/2] + κ²D²_ij/2``.
    Its sample mean is exactly the ``ddof=1`` covariance-identity residual,
    and its derivative w.r.t. κ is ``∂g/∂κ = κ·D²_ij``.

    Args:
        returns: Return panel, shape ``(T, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.
        kappa: Calibrated κ.

    Returns:
        Tuple ``(g, dg)`` each of shape ``(T, P)`` with ``P = C(n, 2)``.

    """
    g0, dtheta = _pair_linear_theta_moment_process(returns, squared_distances)
    g = g0 + kappa**2 * dtheta
    dg = 2.0 * kappa * dtheta
    return g, dg


def _pair_linear_theta_moment_process(
    returns: Float,
    squared_distances: Float,
) -> tuple[Float, Float]:
    """Affine saturation moments for ``theta=κ²``.

    The per-period contribution includes both covariance and marginal-variance
    terms.  The previous implementation inserted full-sample variances as
    constants, which preserved the sample mean but omitted their sampling
    variation from the HAC covariance.  Scaling by ``T/(T-1)`` makes the mean
    match the ``ddof=1`` covariance identity used by :func:`calibrate_kappa`.

    Args:
        returns: Return panel, shape ``(T, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.

    Returns:
        Tuple ``(g0, dtheta)`` of shape ``(T, P)`` with ``P=C(n,2)``.

    """
    centered_returns = returns - returns.mean(axis=0, keepdims=True)
    n_observations, n = returns.shape
    if n_observations < _MIN_MOMENT_OBSERVATIONS:
        message = "the identity moment process requires at least two observations."
        raise ValueError(message)
    iu = np.triu_indices(n, k=1)
    prod = centered_returns[:, iu[0]] * centered_returns[:, iu[1]]  # (T, P)
    marginal = 0.5 * (centered_returns[:, iu[0]] ** 2 + centered_returns[:, iu[1]] ** 2)
    ddof_scale = n_observations / (n_observations - 1)
    g0 = ddof_scale * (prod - marginal)
    dtheta = np.broadcast_to((0.5 * squared_distances[iu])[None, :], g0.shape).copy()
    return g0, dtheta


def identity_tests(
    inputs: IdentityTestInputs,
    config: IdentityTestConfig | None = None,
) -> dict[str, object]:
    """Test equality saturation and one-sided coverage of the W2 benchmark.

    Args:
        inputs: Return panel, squared W2 distances, and calibrated κ estimate.
        config: Portfolio-clustering and Monte-Carlo controls.

    Returns:
        Dict of the test statistics and p-values.

    """
    config = IdentityTestConfig() if config is None else config
    g, dg = _pair_moment_process(
        inputs.returns, inputs.squared_distances, inputs.kappa_hat
    )
    g0, dtheta = _pair_linear_theta_moment_process(
        inputs.returns, inputs.squared_distances
    )
    assign = _cluster_assignments(
        inputs.squared_distances, config.n_clusters, seed=config.seed
    )
    gc = cluster_pairs(g, assign, config.n_clusters)
    dgc = cluster_pairs(dg, assign, config.n_clusters)
    g0c = cluster_pairs(g0, assign, config.n_clusters)
    dtheta_c = cluster_pairs(dtheta, assign, config.n_clusters)

    # The cross-sectional through-origin κ calibration is not the minimizer of
    # this clustered HAC-GMM criterion.  Estimate theta=κ² inside the affine
    # moment system before using the chi-square(m-1) Hansen-J reference.
    # The long-run HAC doors return JAX arrays; this surrounding precision scope
    # is retained for the other float64 report doors in the same calculation.
    with jax.enable_x64(new_val=True):
        jres = linear_hansen_j(g0c, dtheta_c[:, :, None], centered=True)
        if float(jres.params[0]) <= 0.0:
            message = (
                "the GMM estimate of theta=kappa^2 is on/outside its zero "
                "boundary; the interior chi-square Hansen-J reference is not "
                "valid"
            )
            raise Paper3EmpiricalError(message)
        # ``g`` is realized covariance minus the pairwise ceiling. The theorem
        # implies ``E[g] <= 0``, while the Wolak API tests moments pointing
        # nonnegative, so the tested ceiling slack is ``-g``.
        wol = wolak_test(-gc, n_mc=config.n_mc, seed=config.seed)
        jacobian = moment_jacobian_concentration(gc, dgc)
    # ``kappa_delta_method`` is an eager JAX report door that owns its local
    # float64 precision scope; it does not depend on process-global JAX state.
    kd = kappa_delta_method(inputs.squared_distances, inputs.kappa_hat, inputs.kappa_se)

    return {
        "n_clusters": config.n_clusters,
        "gmm_profiled_overid_stat": jres.stat,
        "gmm_profiled_overid_dof": jres.dof,
        "gmm_profiled_overid_pvalue": jres.pvalue,
        "gmm_profiled_theta_hat": float(jres.params[0]),
        "hansen_j_stat": jres.stat,
        "hansen_j_dof": jres.dof,
        "hansen_j_pvalue": jres.pvalue,
        "hansen_j_theta_hat": float(jres.params[0]),
        # Historical publication keys now alias the same internally profiled
        # over-identification quadratic. They no longer denote a second Wald
        # test evaluated at the external cross-sectional kappa estimate.
        "gmm_wald_stat": jres.stat,
        "gmm_wald_dof": jres.dof,
        "gmm_wald_pvalue": jres.pvalue,
        "gmm_wald_compatibility_alias": True,
        "wolak_stat": wol.stat,
        "wolak_pvalue": wol.pvalue,
        "wolak_n_binding": wol.n_binding,
        "kappa_hat": inputs.kappa_hat,
        "kappa_se": inputs.kappa_se,
        "moment_jacobian_concentration": jacobian.concentration,
        "moment_jacobian_flagged_weak": bool(jacobian.flagged_weak),
        "kappa_entry_var_mean": float(
            np.mean(kd.entry_var[np.triu_indices(inputs.squared_distances.shape[0], 1)])
        ),
    }
