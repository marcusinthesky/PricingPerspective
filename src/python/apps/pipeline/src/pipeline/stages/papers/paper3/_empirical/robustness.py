"""Descriptive profiled GEL and uncorrected indicator GMS diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from jcor.inference import (
    cluster_pairs,
    indicator_gms,
    linear_gel_test,
    linear_hansen_j,
    wolak_test,
)

from pipeline.stages.papers.paper3._empirical.contracts import (
    SIGNIFICANCE_LEVEL,
    IdentityTestConfig,
    Paper3EmpiricalError,
)
from pipeline.stages.papers.paper3._empirical.identity import (
    _pair_linear_theta_moment_process,
    _pair_moment_process,
)
from pipeline.stages.substrate.energy_shared import _cluster_assignments

if TYPE_CHECKING:
    from pipeline.stages.papers.paper3._empirical.contracts import (
        IdentityTestInputs,
    )


def _profile_gel_carrier(
    g0: jax.Array,
    dtheta: jax.Array,
    criterion: str,
) -> dict[str, object]:
    """Profile one GEL carrier and enforce convergence/reference guards."""
    try:
        result = linear_gel_test(
            g0,
            dtheta[:, :, None],
            criterion=criterion,
            kind="lr",
            assume_iid=False,
        )
        theta_hat = float(result.params[0])
        interior = theta_hat > 0.0
        descriptive_available = bool(result.converged and interior)
        trustworthy = bool(descriptive_available and result.reference_valid)
        return {
            "stat": float(result.stat) if descriptive_available else float("nan"),
            "dof": int(result.dof),
            "pvalue": float(result.pvalue) if trustworthy else float("nan"),
            "converged": bool(result.converged),
            "inner_converged": bool(result.inner_converged),
            "outer_converged": bool(result.outer_converged),
            # False means the identified moment rank moved along the profile
            # path, so `dof` would describe where the optimizer stopped rather
            # than the model. Recorded because it gates `converged`.
            "subspace_stable": bool(result.subspace_stable),
            "theta_hat": theta_hat,
            "interior_theta": interior,
            "iid_reference_acknowledged": bool(result.reference_valid),
            "descriptive_available": descriptive_available,
            "descriptive_only": bool(descriptive_available and not trustworthy),
            "trustworthy": trustworthy,
        }
    except (
        np.linalg.LinAlgError,
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "trustworthy": False}


def robustness_gel_gms(
    inputs: IdentityTestInputs,
    config: IdentityTestConfig | None = None,
) -> dict[str, object]:
    """Profiled GEL and uncorrected indicator-GMS robustness diagnostics.

    Reuses the *identical* pair-moment process and pair→portfolio clustering as
    :func:`identity_tests`, so the GEL/GMS verdicts are directly comparable to
    the profiled GMM/Hansen-J and Wolak baseline. Two Paper-3 robustness items:

    * **GEL over-identification** (equality-restriction alternative to profiled
      GMM/Hansen J). CUE / EL / ET carriers, LR-form statistic, dof = ``m − 1``.
      Each carrier estimates ``theta=κ²`` inside its own saddle-point criterion;
      an external κ estimate is never used to subtract a parameter dimension.
      This observation-level GEL uses ``g'g/T`` rather than a long-run
      covariance. Because the return-moment panel may be serially dependent,
      its LR objective is reported as a descriptive sensitivity diagnostic and
      its iid chi-square p-value is deliberately withheld.
    * **Uncorrected indicator GMS** — adjusted-QLR with φ⁽¹⁾ moment selection
      and a simulated critical value on the same clustered ceiling-slack moments. The
      implementation uses ``eta=0`` and is not labelled as the exact
      Andrews--Barwick Table-I RMS calibration.

    Args:
        inputs: Return panel, squared W2 values, and calibrated κ estimate.
        config: Clustering and Monte-Carlo controls, shared with
            :func:`identity_tests` when comparing the two procedures.

    Returns:
        Dict of GEL / GMS statistics, p-values and agreement flags vs baseline.

    """
    config = IdentityTestConfig(n_mc=5000) if config is None else config
    g, _dg = _pair_moment_process(
        inputs.returns, inputs.squared_distances, inputs.kappa_hat
    )
    g0, dtheta = _pair_linear_theta_moment_process(
        inputs.returns, inputs.squared_distances
    )
    assign = _cluster_assignments(
        inputs.squared_distances, config.n_clusters, seed=config.seed
    )
    gc = cluster_pairs(g, assign, config.n_clusters)
    g0c = cluster_pairs(g0, assign, config.n_clusters)
    dtheta_c = cluster_pairs(dtheta, assign, config.n_clusters)
    m = gc.shape[1]

    # GEL over-identification (equality test, dof = m - 1 for profiled theta).
    # The clustered-pair second-moment Omega=g'g/T is near-collinear at m~15, so
    # the inner sup is solved on Omega's identified row-space with a trust-region
    # cap. A converged, interior profile yields a descriptive LR objective, but
    # not a chi-square test: ordinary observation-level GEL assumes iid moments,
    # whereas this panel may be serially dependent. Nonconverged carriers are
    # withheld completely.
    gel: dict[str, dict[str, object]] = {}
    for crit in ("cue", "el", "et"):
        gel[crit] = _profile_gel_carrier(g0c, dtheta_c, crit)

    # The pair moment g_t = realized - ceiling has mean zero under saturation.
    # The W2 theorem gives E[g] <= 0, so GMS/Wolak consume the nonnegative
    # ceiling slack -g. Equality-profiled GEL/GMM remain on g itself.
    # The long-run HAC doors return JAX arrays; this surrounding precision scope
    # is retained for the other float64 report doors in the same calculation.
    with jax.enable_x64(new_val=True):
        ceiling_slack = -gc
        gms = indicator_gms(
            ceiling_slack, alpha=0.05, n_mc=config.n_mc, seed=config.seed
        )
        # Indicator GMS with no moment selection (kappa_n → ∞) uses the SAME
        # QLR statistic and
        # the SAME least-favourable (all-binding) configuration as Wolak,
        # differing only in that its critical value comes from a direct
        # simulation of the QLR distance rather than the χ̄² weight-count
        # approximation. Recording it isolates whether the floor verdict is
        # driven by moment *selection* or by the critical-value *calibration*.
        gms_noselect = indicator_gms(
            ceiling_slack,
            alpha=0.05,
            n_mc=config.n_mc,
            seed=config.seed,
            kappa_n=1e12,
        )
        wol = wolak_test(ceiling_slack, n_mc=config.n_mc, seed=config.seed)
        gmm = linear_hansen_j(g0c, dtheta_c[:, :, None], centered=True)
    if float(gmm.params[0]) <= 0.0:
        message = (
            "the GMM estimate of theta=kappa^2 is on/outside its zero boundary; "
            "the interior chi-square Hansen-J reference is not valid"
        )
        raise Paper3EmpiricalError(message)

    # Agreement across tests (chi-bar-squared bug fixed):
    #   equality verdict: REJECTED by profiled HAC-GMM/Hansen J.
    #   ceiling verdict: Wolak and indicator-GMS evaluate nonnegative slack.
    # No GEL carrier counts toward an inferential verdict without an explicit
    # iid moment assumption; optimizer convergence alone is insufficient.
    gel_p_trust = [
        float(pv)
        for v in gel.values()
        if v.get("trustworthy") and isinstance(pv := v.get("pvalue"), (int, float))
    ]
    descriptive_carriers = [c for c, v in gel.items() if v.get("descriptive_available")]
    inferential_carriers = [c for c, v in gel.items() if v.get("trustworthy")]
    gel_rejects_equality = bool(gel_p_trust) and all(
        p < SIGNIFICANCE_LEVEL for p in gel_p_trust
    )
    wolak_rejects_ceiling = bool(wol.pvalue < SIGNIFICANCE_LEVEL)
    gms_rejects_ceiling = bool(gms.reject)

    return {
        "n_clusters": config.n_clusters,
        "n_moments": int(m),
        "baseline": {
            "gmm_profiled_overid_stat": float(gmm.stat),
            "gmm_profiled_overid_dof": int(gmm.dof),
            "gmm_profiled_overid_pvalue": float(gmm.pvalue),
            "gmm_profiled_theta_hat": float(gmm.params[0]),
            "gmm_profiled_rejects_equality": bool(gmm.pvalue < SIGNIFICANCE_LEVEL),
            "wolak_pvalue": float(wol.pvalue),
            "wolak_rejects_ceiling": wolak_rejects_ceiling,
        },
        "gel_over_id": gel,
        "gms": {
            "procedure": gms.calibration,
            "rms_table_i_calibrated": bool(gms.rms_table_i_calibrated),
            "eta": float(gms.eta),
            "statistic": "qlr",
            "stat": float(gms.stat),
            "crit_0.05": float(gms.crit),
            "pvalue": float(gms.pvalue),
            "reject": bool(gms.reject),
            "n_selected_binding": int(gms.n_selected),
            "kappa_n": float(gms.kappa_n),
        },
        "gms_no_selection": {
            "procedure": gms_noselect.calibration,
            "rms_table_i_calibrated": bool(gms_noselect.rms_table_i_calibrated),
            "eta": float(gms_noselect.eta),
            "stat": float(gms_noselect.stat),
            "crit_0.05": float(gms_noselect.crit),
            "pvalue": float(gms_noselect.pvalue),
            "reject": bool(gms_noselect.reject),
            "note": (
                "kappa_n -> inf: no moment selection, all-binding LF config, "
                "same QLR statistic as Wolak. Rejecting here isolates the "
                "critical-value calibration (direct QLR-distance simulation) "
                "from the moment selection."
            ),
        },
        "agreement": {
            "gel_agrees_equality_rejected": gel_rejects_equality,
            "gel_converged_carriers": descriptive_carriers,
            "gel_descriptive_carriers": descriptive_carriers,
            "gel_inferential_carriers": inferential_carriers,
            "wolak_and_gms_agree_ceiling_rejected": bool(
                wolak_rejects_ceiling and gms_rejects_ceiling
            ),
            "interpretation": (
                "The scalar-transmission, zero-transport-excess benchmark is "
                "REJECTED as an equality "
                "by the internally fitted HAC-GMM Hansen-J test. Each GEL "
                "carrier profiles theta=kappa^2 in its own outer criterion, "
                "but the observation-level carrier uses g'g/T rather than a "
                "long-run covariance. Because the panel may be serially "
                "dependent, converged GEL LR objectives are descriptive only "
                "and their iid chi-square p-values are withheld; failed carrier "
                "fits are withheld completely. On "
                "the one-sided covariance-ceiling inequality, the output records "
                "the Wolak "
                "chi-bar-square test separately from the eta=0 uncorrected "
                "indicator-GMS and no-selection diagnostics; their computed "
                "flags, p-values, and Monte Carlo controls determine each "
                "verdict. The indicator result is not the exact "
                "Andrews-Barwick Table-I RMS calibration. The machine-checked "
                "W2 Frechet ceiling and nonnegative transport-excess theorem are "
                "unaffected: they concern latent exposure laws and a certified "
                "coupling, whereas this empirical inequality substitutes observed "
                "characteristic-law W2 through a scalar transmission restriction."
            ),
        },
    }
