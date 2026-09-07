"""Decision disclosures and diagnostics for the H1 pilot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.stages.substrate.w2_exposure_frontier.contracts import (
    DEFAULT_WOLAK_ELL0 as _DEFAULT_WOLAK_ELL0,
)

if TYPE_CHECKING:
    from pipeline.stages.substrate.w2_exposure_frontier.returns_loadings import (
        LoadingsResult,
    )

from pipeline.stages.substrate.w2_exposure_frontier.frontier import cap_reduction_pct

# The external ell_0 anchor and target MDE await the plan's author calibration.
# Defaulted to the weakest economically-motivated anchor (0.0, "the text
# distance provides SOME nonnegative lower bound on return-exposure
# distance") so the mechanism is exercised end-to-end; this is NOT a
# considered anchor and must be replaced before the verdict is treated as
# final. See run_h1_pilot's `wolak_ell0` parameter.
DEFAULT_WOLAK_ELL0 = _DEFAULT_WOLAK_ELL0
TOP_FACTOR_DEGENERACY_THRESHOLD = 0.60


DISCLOSURE = {
    "identification": (
        "The lower/upper Lipschitz frontier (ell_hat > 0, L_hat < inf) is a "
        "TESTED HYPOTHESIS on the returns<->text-distance relationship, NOT a "
        "machine-checked theorem. The certified variance cap that consumes "
        "ell is proven in Lean (Energy.lean:585, portfolio_variance_upper_"
        "bound / certified_floor); the premise ell > 0 that makes it bite is "
        "only empirically supported here."
    ),
    "conservative_kill": (
        "The kill uses the conservative bootstrap floor ell_lo (2.5% "
        "percentile of the return-block bootstrap), never the point estimate "
        "ell_hat. EIV widens the CI (more beta noise -> smaller ell_lo -> "
        "looser but valid cap)."
    ),
    "lean_binding": (
        "The certified cap binds to the NON-HOMOGENEOUS envelope "
        "portfolio_variance_upper_bound (Energy.lean:585) via "
        "simulation.bounds.certified_floor_np, matching "
        "||beta_i|| != const (whitened-PCA loadings are real, not "
        "price-anchored). The homogeneous certified_risk_cap (Energy.lean:695, "
        "requires ||beta_i|| = sigma) is NOT the headline and is retired from "
        "this pilot (plan Author decision #4)."
    ),
    "single_frontier": (
        "There is a SINGLE frontier and a SINGLE ell_lo (F3 fix): the "
        "text-native / price-anchored split of the prior embeddings-only "
        "pilot is retired. Numerator is returns-based whitened-PCA loadings "
        "(pipeline.returns_loadings); denominator is the typed text W2 "
        "distance D."
    ),
    "wolak_anchor": (
        "wolak_stat/wolak_p test the moment inequalities against the "
        "EXTERNALLY ANCHORED wolak_ell0 (see config.wolak_ell0), not a "
        "self-fitted quantile (F4 fix). wolak_ell0 is currently a "
        "TODO(author) PLACEHOLDER (module DEFAULT_WOLAK_ELL0 = 0.0, the "
        "weakest economically-motivated anchor) pending the author's "
        "considered external anchor + target MDE (plan Author decision #6); "
        "mde is reported alongside for power assessment."
    ),
    "beta_hat_provenance": (
        "beta_hat (the frontier numerator) is the whitened-PCA loadings "
        "estimated from the DAILY RETURN PANEL via "
        "pipeline.returns_loadings.estimate_loadings, not from text "
        "embeddings (F1 fix). This is a returns-based provenance claim, not "
        "a Lean-certified one."
    ),
    "wolak_ell0_placeholder": (
        "wolak_ell0 (module DEFAULT_WOLAK_ELL0 = 0.0) is a NON-INFORMATIVE "
        "PLACEHOLDER anchor -- it tests the weakest possible hypothesis "
        "(text distance provides some nonnegative lower bound on "
        "return-exposure distance) and is NOT the author's considered "
        "external anchor. wolak_stat/wolak_p computed against this "
        "placeholder should not be read as a substantive identification "
        "test until the anchor is replaced (plan Author decision #6, "
        "PENDING)."
    ),
    "variance_cap_status": "FAIL",
}


def _weight_scheme_spread(
    schemes: list[tuple[str, np.ndarray]],
    betas: np.ndarray,
    d2: np.ndarray,
    ell: float,
) -> dict[str, float]:
    """Cap-reduction pct across weight schemes: equal-weight headline + spread."""
    equal = next(w for name, w in schemes if name == "equal")
    equal_pct = cap_reduction_pct(equal, betas, d2, ell).pct
    dirichlet = [
        cap_reduction_pct(w, betas, d2, ell).pct
        for name, w in schemes
        if name.startswith("dirichlet")
    ]
    return {
        "equal_weight": float(equal_pct),
        "dirichlet_min": float(np.min(dirichlet)) if dirichlet else float("nan"),
        "dirichlet_median": (
            float(np.median(dirichlet)) if dirichlet else float("nan")
        ),
        "dirichlet_max": float(np.max(dirichlet)) if dirichlet else float("nan"),
    }


def _k_diagnostics(
    sweep: dict[int, LoadingsResult], k_chosen: int
) -> dict[str, object]:
    """Compute the variance sweep and top-factor degeneracy diagnostic."""
    full_k = max(sweep)
    top_share = float(sweep[full_k].eigenvalues[0] / sweep[full_k].eigenvalues.sum())
    return {
        "k_chosen": k_chosen,
        "explained_variance_by_k": {
            int(k): res.explained_variance_ratio for k, res in sweep.items()
        },
        "top_factor_share_of_swept_trace": top_share,
        "top_factor_degenerate": bool(top_share >= TOP_FACTOR_DEGENERACY_THRESHOLD),
    }
