"""κ identity-form audit (ROADMAP §4 item 3).

Data-free simulator that settles the figure-10 vs figure-22 scaling
contradiction empirically: which of the three candidate identity forms makes
the residual ``g_ij = Cov_sys(i,j) − ½(σ_i² + σ_j² − scale·D²_ij)`` vanish
under a white-noise factor field, and how the coloured (exponential /
Matérn-3/2) kernels break it.

Candidate forms:
    * ``"plain"``          — scale = D²_ij            (no κ)
    * ``"half_kappa2"``    — scale = ½·κ²·D²_ij       (homogeneous κ)
    * ``"half_kappa_ij"``  — scale = ½·κ_i·κ_j·D²_ij  (heterogeneous κ)

Ground truth.  Assets have exposure densities ``p_i`` (Gaussian mixtures on
``Ω = [0,1]^d``, cluster-structured); returns ``R_i(t) = κ_i·∫p_i F(x,t)dx +
ε_i(t)`` with a Gaussian factor field ``F``.  The *population* systematic
covariance is ``Cov_sys(i,j) = κ_i κ_j · pᵢᵀ K pⱼ``, where ``K`` is the field
kernel on the grid. The *population* energy distance ``D²_ij`` is computed from
the exposure clouds.  We report, per form, the mean absolute residual ``|g_ij|``
over off-diagonal pairs — the form with the smallest residual is the correct
identity, and the gap across kernels calibrates the misspecification tolerance.

This module uses key-only JAX draws and performs explicit Python scalar
conversion only when constructing the audit table and summary.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pandas as pd
from jcor.core.random import cell_key
from jcor.core.typing import ArrayLike  # noqa: TC002  # runtime array contract

from simulation.generators.fields import grid_points, kernel_matrix


class AuditRow(NamedTuple):
    """One audit cell result.

    Attributes:
        kernel: Field kernel kind.
        form: Identity form label.
        mean_abs_residual: Mean ``|g_ij|`` over off-diagonal pairs.
        max_abs_residual: Max ``|g_ij|``.
        kappa_regime: ``"homogeneous"`` or ``"heterogeneous"``.

    """

    kernel: str
    form: str
    mean_abs_residual: float
    max_abs_residual: float
    kappa_regime: str


def _exposure_densities(
    key: jax.Array,
    n: int,
    resolution: int,
    d: int,
    n_clusters: int,
    bandwidth: float = 0.12,
) -> tuple[jax.Array, jax.Array]:
    """Draw ``n`` normalised exposure densities on the grid + their samples.

    Each asset is a Gaussian bump centred at a cluster location on ``[0,1]^d``.

    Returns:
        Tuple ``(p, centers)`` where ``p`` is ``(n, M)`` row-normalised
        densities on the ``M = resolution^d`` grid and ``centers`` are the
        ``(n, d)`` bump centres (used as exposure "samples" for energy
        distance).

    """
    pts = grid_points(resolution, d)  # (M, d)
    center_key, assignment_key, jitter_key = jax.random.split(key, 3)
    cluster_centers = jax.random.uniform(center_key, (n_clusters, d))
    assign = jax.random.randint(assignment_key, (n,), 0, n_clusters)
    jitter = 0.05 * jax.random.normal(jitter_key, (n, d))
    centers_jax = jnp.clip(cluster_centers[assign] + jitter, 0.0, 1.0)

    diff = centers_jax[:, None, :] - pts[None, :, :]  # (n, M, d)
    r2 = jnp.sum(diff**2, axis=-1)
    p = jnp.exp(-r2 / (2.0 * bandwidth**2))
    p = p / p.sum(axis=1, keepdims=True)  # row-normalised
    return p, centers_jax


def _energy_distance_sq_matrix(centers: ArrayLike) -> jax.Array:
    """Compute population squared energy distance between bump centres.

    For point-mass exposures (bump centres), ``E²(δ_a, δ_b) = ‖a − b‖`` (the
    energy distance of two Diracs is the ground metric); we use the squared
    Euclidean surrogate ``‖a−b‖²`` consistently with the ``D²`` convention
    used throughout the covariance identity.

    Args:
        centers: Bump centres, shape ``(n, d)``.

    Returns:
        Symmetric ``(n, n)`` matrix of ``D²_ij`` with zero diagonal.

    """
    centers = jnp.asarray(centers)
    diff = centers[:, None, :] - centers[None, :, :]
    return jnp.sum(diff**2, axis=-1)


def _residuals(
    covariance: ArrayLike,
    sigma_sq: ArrayLike,
    d2: ArrayLike,
    form: str,
    kappa_vec: ArrayLike,
) -> jax.Array:
    """Off-diagonal residual matrix ``g_ij`` for a given identity form."""
    covariance = jnp.asarray(covariance)
    sigma_sq = jnp.asarray(sigma_sq)
    d2 = jnp.asarray(d2)
    kappa_vec = jnp.asarray(kappa_vec)
    n = covariance.shape[0]
    if form == "plain":
        scale = d2
    elif form == "half_kappa2":
        kappa2 = jnp.mean(kappa_vec) ** 2
        scale = 0.5 * kappa2 * d2
    elif form == "half_kappa_ij":
        scale = 0.5 * kappa_vec[:, None] * kappa_vec[None, :] * d2
    else:
        message = f"unknown form {form!r}"
        raise ValueError(message)
    implied = 0.5 * (sigma_sq[:, None] + sigma_sq[None, :]) - scale
    g = covariance - implied
    mask = ~jnp.eye(n, dtype=bool)
    return g[mask]


def run_kappa_audit(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the κ identity-form audit across kernels and κ regimes.

    Args:
        cfg: Config dict with keys ``n_assets``, ``resolution``, ``d``,
            ``n_clusters``, ``lengthscale``, ``kappa_base``, ``kappa_spread``,
            ``kernels`` (list), ``seed``.

    Returns:
        Tuple ``(results_df, summary)``.

    """
    n = int(cfg["n_assets"])
    resolution = int(cfg["resolution"])
    d = int(cfg["d"])
    n_clusters = int(cfg["n_clusters"])
    lengthscale = float(cfg["lengthscale"])
    kappa_base = float(cfg["kappa_base"])
    kappa_spread = float(cfg["kappa_spread"])
    kernels = list(cfg["kernels"])
    seed = int(cfg["seed"])

    key = cell_key(
        seed,
        "mc_kappa_audit",
        "n_assets",
        n,
        "resolution",
        resolution,
        "dimension",
        d,
    )
    exposure_key, kappa_key = jax.random.split(key)
    p, centers = _exposure_densities(exposure_key, n, resolution, d, n_clusters)
    pts = grid_points(resolution, d)
    d2 = _energy_distance_sq_matrix(centers)

    kappa_homo = jnp.full(n, kappa_base)
    kappa_hetero = kappa_base + kappa_spread * jax.random.normal(kappa_key, (n,))

    rows: list[AuditRow] = []
    for kernel in kernels:
        field_covariance = kernel_matrix(pts, kernel, lengthscale)  # (M, M)
        p_k_p = p @ field_covariance @ p.T  # (n, n) — pᵢᵀ K pⱼ
        for regime, kappa_vec in (
            ("homogeneous", kappa_homo),
            ("heterogeneous", kappa_hetero),
        ):
            # Population systematic covariance under the field.
            covariance = kappa_vec[:, None] * kappa_vec[None, :] * p_k_p
            sigma_sq = jnp.diag(covariance)
            for form in ("plain", "half_kappa2", "half_kappa_ij"):
                g = _residuals(covariance, sigma_sq, d2, form, kappa_vec)
                rows.append(
                    AuditRow(
                        kernel=kernel,
                        form=form,
                        mean_abs_residual=float(jnp.mean(jnp.abs(g))),
                        max_abs_residual=float(jnp.max(jnp.abs(g))),
                        kappa_regime=regime,
                    )
                )

    df = pd.DataFrame([r._asdict() for r in rows])

    # Headline: which form minimises the residual under the white kernel.
    summary: dict[str, Any] = {"by_kernel": {}}
    for kernel in kernels:
        sub = df[df["kernel"] == kernel]
        best = sub.loc[sub["mean_abs_residual"].idxmin()]
        summary["by_kernel"][kernel] = {
            "best_form": str(best["form"]),
            "best_regime": str(best["kappa_regime"]),
            "best_mean_abs_residual": float(best["mean_abs_residual"]),
        }
    white = summary["by_kernel"].get("white", {})
    summary["headline"] = (
        f"Under the white-noise field the '{white.get('best_form', 'n/a')}' "
        f"identity form minimises the residual "
        f"(mean |g_ij| = {white.get('best_mean_abs_residual', float('nan')):.3e}); "
        "coloured kernels inflate the residual, calibrating the "
        "misspecification tolerance."
    )
    summary["params"] = cfg
    return df, summary
