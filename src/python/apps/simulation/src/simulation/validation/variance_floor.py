"""Monte-Carlo validation of the Lean theorem ``portfolio_variance_lower_bound``.

Theorem (PricingPerspective/Continuous/Energy.lean:380-394):
    For weights w >= 0 and exposures beta_i with pairwise Lipschitz control
    ||beta_i - beta_j|| <= l * D_ij  (D_ij >= 0, l >= 0):

        sum_{ij} w_i w_j * 0.5 * (||beta_i||^2 + ||beta_j||^2 - (l*D_ij)^2)
            <= ||sum_i w_i beta_i||^2

Two arms, because "expects ZERO violations" alone certifies nothing
------------------------------------------------------------------
``l`` is estimated as the *tightest* constant satisfying the hypothesis,
``l_tight = max ||beta_i - beta_j|| / D_ij``, and the floor is decreasing in
``l``. So on a grid of ``lipschitz_scale_factors`` that are all ``>= 1`` the
conclusion is satisfied by construction and a zero-violation result is not
evidence about the theorem — it is evidence about the monotonicity of the
expression. The run is split accordingly:

``certify`` (``monte_carlo.lipschitz_scale_factors``, all ``>= 1``)
    The hypothesis holds on every evaluated pair. Zero violations is a hard
    gate; the CLI exits non-zero otherwise.

``probe`` (``monte_carlo.lipschitz_tightness_probe_factors``, all ``< 1``)
    The hypothesis is deliberately broken. Violations are *expected* here, and
    their absence would mean the floor is slack enough to be uninformative. In
    this landing the probe arm is **reported and logged, never gated** — a
    knife-edge geometry that happened not to violate would otherwise turn a
    research stage red. Promote it to a gate once a run has been inspected.

Pairs with ``D_ij < eps`` cannot be Lipschitz-controlled by any finite ``l``
and are skipped when estimating it, so the hypothesis is *not established* for
them while the conclusion is still asserted. They are counted per config and
the total is gated in the ``certify`` arm — see ``hypothesis_complete``.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003  # runtime beartype contract
from typing import Any, cast

import jax
import jax.numpy as jnp
import pandas as pd
from jax.typing import ArrayLike  # noqa: TC002  # runtime array contract
from jcor.core.random import cell_key
from jcor.discrepancy.energy import energy_distance_matrix

from simulation.generators.assets import cluster_embeddings as sample_exposures
from simulation.portfolios import weight_schemes as make_weight_schemes

#: Scale factors ``>= 1``: the theorem's hypothesis holds, so zero violations
#: is a hard gate.
CERTIFY_ARM = "certify"
#: Scale factors ``< 1``: the hypothesis is deliberately broken, so violations
#: are the expected outcome and their absence is the signal worth reading.
PROBE_ARM = "probe"


def compute_energy_distances(
    sample_sets: Sequence[object],
) -> jax.Array:
    """Compute pairwise energy distances D_ij between asset sample sets.

    ``metric="euclidean"`` and the square root of the energy functional are
    load-bearing: the theorem uses a genuine distance on the exposure clouds.
    """
    a = jnp.stack([jnp.asarray(samples, dtype=jnp.float64) for samples in sample_sets])
    functional = energy_distance_matrix(  # noqa: PD011
        list(a), exponent=1.0, metric="euclidean"
    ).values
    return jnp.sqrt(jnp.maximum(functional, 0.0))


def compute_lipschitz_constant(
    betas: ArrayLike, distance_matrix: ArrayLike, eps: float = 1e-12
) -> tuple[float, int]:
    """Compute the tightest Lipschitz constant satisfying the pairwise bound.

    Returns (l, n_skipped_pairs).
    """
    betas = jnp.asarray(betas)
    distance_matrix = jnp.asarray(distance_matrix)
    n = betas.shape[0]
    iu = jnp.triu_indices(n, k=1)
    diff_norm = jnp.linalg.norm(betas[:, None, :] - betas[None, :, :], axis=2)[iu]
    d_pairs = distance_matrix[iu]
    valid = d_pairs >= eps
    n_skipped = int(jnp.sum(~valid))
    safe_d_pairs = jnp.where(valid, d_pairs, 1.0)
    ratios = jnp.where(valid, diff_norm / safe_d_pairs, 0.0)
    lipschitz = float(jnp.max(ratios)) if ratios.size else 0.0
    return lipschitz, n_skipped


def _variance_floor_term(
    betas: ArrayLike, distance_matrix: ArrayLike, lipschitz: float
) -> jax.Array:
    """Build the weight-invariant matrix in the variance-floor expression."""
    betas = jnp.asarray(betas)
    distance_matrix = jnp.asarray(distance_matrix)
    beta_norms_sq = jnp.sum(betas**2, axis=1)
    l2_d2 = (lipschitz * distance_matrix) ** 2
    return 0.5 * (beta_norms_sq[:, None] + beta_norms_sq[None, :] - l2_d2)


def _floor_realized_from_term(
    w: ArrayLike, betas: ArrayLike, term: ArrayLike
) -> tuple[float, float]:
    """Per-scheme floor/realized from a precomputed :func:`_variance_floor_term`."""
    w = jnp.asarray(w)
    betas = jnp.asarray(betas)
    term = jnp.asarray(term)
    ww = jnp.outer(w, w)
    floor = float(jnp.sum(ww * term))
    portfolio_beta = w @ betas
    realized = float(jnp.dot(portfolio_beta, portfolio_beta))
    return floor, realized


def compute_floor_and_realized(
    w: ArrayLike,
    betas: ArrayLike,
    distance_matrix: ArrayLike,
    lipschitz: float,
) -> tuple[float, float]:
    """Compute the variance floor and realized portfolio variance.

    floor = sum_{ij} w_i w_j * 0.5 * (||beta_i||^2 + ||beta_j||^2 - (l*D_ij)^2)
    realized = ||sum_i w_i beta_i||^2
    """
    term = _variance_floor_term(betas, distance_matrix, lipschitz)
    return _floor_realized_from_term(w, betas, term)


def _largest_violating_scale(probe: pd.DataFrame) -> float | None:
    """Largest probe scale factor that still produced a violation, if any.

    The floor is decreasing in ``l``, so violations appear once ``l`` drops far
    enough below ``l_tight``. The largest violating factor is therefore the
    boundary of the certified region and the honest measure of how much slack
    the certificate at ``l_tight`` carries.

    Args:
        probe: The ``probe``-arm rows of the results frame.

    Returns:
        The largest violating ``lipschitz_scale``, or ``None`` when the probe
        arm is empty or nothing violated at any probed factor.

    """
    if not len(probe):
        return None
    violating = probe.loc[probe["violation"], "lipschitz_scale"]
    if not len(violating):
        return None
    return float(violating.max())


def run_variance_floor(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the variance-floor Monte-Carlo validation.

    Rows carry an ``arm`` column — ``"certify"`` for the ``>= 1`` scale factors
    whose hypothesis holds, ``"probe"`` for the deliberately sub-tight ones.
    See the module docstring for why a single arm certifies nothing.

    Args:
        cfg: ``params["monte_carlo"]`` dict from ``params.yaml``.
            ``lipschitz_tightness_probe_factors`` is optional and defaults to
            an empty probe arm, so an un-updated ``params.yaml`` behaves
            exactly as before this split.

    Returns:
        (results_df, summary) — DataFrame of per-config results and summary dict.

    """
    n_assets_list: list[int] = cfg["n_assets"]
    embedding_dims: list[int] = cfg["embedding_dim"]
    n_points: int = cfg["n_points_per_asset"]
    n_clusters: int = cfg["n_clusters"]
    n_weight_draws: int = cfg["n_weight_draws"]
    lipschitz_scales: list[float] = cfg["lipschitz_scale_factors"]
    probe_scales: list[float] = list(cfg.get("lipschitz_tightness_probe_factors") or [])
    tolerance: float = cfg["tolerance"]
    base_seed: int = cfg["seed"]

    # (scale, arm) pairs. `certify` runs first so its rows lead the frame.
    scale_arms: list[tuple[float, str]] = [
        *((float(s), CERTIFY_ARM) for s in lipschitz_scales),
        *((float(s), PROBE_ARM) for s in probe_scales),
    ]

    rows = []

    for n_assets in n_assets_list:
        for embedding_dim in embedding_dims:
            key = cell_key(
                base_seed,
                "mc_variance_floor",
                "n_assets",
                n_assets,
                "embedding_dim",
                embedding_dim,
            )
            exposure_key, weight_key = jax.random.split(key)
            exposure_sample = sample_exposures(
                exposure_key, n_assets, n_points, embedding_dim, n_clusters
            )
            betas = exposure_sample.betas
            sample_sets = [
                jnp.asarray(sample, dtype=jnp.float64)
                for sample in exposure_sample.samples
            ]
            distance_matrix = compute_energy_distances(sample_sets)

            l_tight, n_skipped = compute_lipschitz_constant(betas, distance_matrix)

            weight_schemes = make_weight_schemes(weight_key, n_assets, n_weight_draws)

            for l_scale, arm in scale_arms:
                lipschitz = l_tight * l_scale
                # This term is shared by every weight scheme in the cell.
                # compute it once per l_scale instead of once per scheme.
                term = _variance_floor_term(betas, distance_matrix, lipschitz)

                for scheme_name, weights in weight_schemes:
                    w = jnp.asarray(weights)
                    floor, realized = _floor_realized_from_term(w, betas, term)
                    slack = realized - floor
                    tol_abs = tolerance * max(1.0, abs(realized))
                    violation = slack < -tol_abs

                    rows.append(
                        {
                            "n_assets": n_assets,
                            "embedding_dim": embedding_dim,
                            "lipschitz_scale": l_scale,
                            "arm": arm,
                            "L_tight": l_tight,
                            "L_used": lipschitz,
                            "n_skipped_pairs": n_skipped,
                            "weight_scheme": scheme_name,
                            "floor": floor,
                            "realized": realized,
                            "slack": slack,
                            "violation": violation,
                        }
                    )

    results_df = pd.DataFrame(rows)

    # Every published statistic below is the CERTIFY arm's. The probe arm is
    # engineered to violate, so folding it into `violations` or the slack
    # distribution would corrupt both. With an empty probe arm — the default —
    # `certify` is the whole frame and every key keeps its pre-split value.
    certify = results_df[results_df["arm"] == CERTIFY_ARM]
    probe = results_df[results_df["arm"] == PROBE_ARM]
    violation_count = int(certify["violation"].sum())

    # Collect empirical Lipschitz values under the established output schema.
    empirical_l_values = {}
    for n_assets in n_assets_list:
        for d in embedding_dims:
            mask = (results_df["n_assets"] == n_assets) & (
                results_df["embedding_dim"] == d
            )
            subset = results_df[mask]
            if len(subset) > 0:
                l_tight = float(subset["L_tight"].iloc[0])
                empirical_l_values[f"n{n_assets}_d{d}"] = l_tight

    n_skipped_total = int(
        cast(
            "int",
            results_df.groupby(["n_assets", "embedding_dim"])["n_skipped_pairs"]
            .first()
            .sum(),
        )
    )

    certify_slack = jnp.asarray(certify["slack"].tolist())
    summary = {
        "total_configs": len(certify),
        "violations": violation_count,
        "mean_slack": float(jnp.mean(certify_slack)),
        "min_slack": float(jnp.min(certify_slack)),
        "slack_p1": float(jnp.percentile(certify_slack, 1)),
        "empirical_L_values": empirical_l_values,
        "n_skipped_zero_D_pairs_total": n_skipped_total,
        # A pair with D_ij < eps admits no finite Lipschitz constant, so the
        # theorem's hypothesis is not established for it while its conclusion
        # is still asserted. Zero skipped pairs is what makes the certify arm
        # a certificate rather than a survey.
        "hypothesis_complete": n_skipped_total == 0,
        "probe_rows": len(probe),
        "probe_scale_factors": [float(s) for s in probe_scales],
        "probe_violations": int(probe["violation"].sum()) if len(probe) else 0,
        # The largest sub-tight scale factor that still breaks the bound — the
        # measured Lipschitz margin. `1 / probe_binding_scale` is how many times
        # smaller `l` can get before the floor stops holding, so a value of 0.15
        # means the certificate at `l_tight` had ~6.7x of slack. `None` when the
        # probe arm is empty or nothing violated at any probed factor.
        "probe_binding_scale": _largest_violating_scale(probe),
        "tolerance": tolerance,
        "parameters": cfg,
    }

    return results_df, summary
