"""Monte-Carlo validation of the energy-barycentre hedging-error bound.

Bound (empirically tested):
    For a target asset T and candidates {1, ..., n_candidates} with optimal
    mixture weights w* minimising the exact finite-mixture energy, and
    Lipschitz constant
        lipschitz = max_i  ||beta_T - beta_i|| / D_E(T, i)   (over i with D_E > 0):

        ||beta_T - sum_i w*_i beta_i||  <=  lipschitz * sum_i w*_i * D_E(T, i)

    The machine-checked upper bound uses the weighted pairwise distances on its
    right-hand side.  That quantity is distinct from the target-to-mixture
    energy objective used to select w* and is recorded separately.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003  # runtime beartype contract
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import pandas as pd
from jax.typing import ArrayLike  # noqa: TC002  # runtime array contract
from jcor.core.random import cell_key
from jcor.discrepancy.balancing import energy_distance_kernel_mixture
from jcor.discrepancy.energy import energy_distance, mean_distance_matrix
from jcor.geometry import EnergyBarycentreDiagnostics, energy_barycentre_weights_batched

from simulation.generators.assets import cluster_embeddings
from simulation.optimize import energy_barycentre_weights_qp

if TYPE_CHECKING:
    from jcor.ground import GroundDistanceName


@partial(jax.jit, static_argnames=("exponent", "metric"))
def _energy_distance_to_candidates_batched(
    target: jnp.ndarray,
    candidates: jnp.ndarray,
    exponent: float = 1.0,
    metric: GroundDistanceName = "euclidean",
) -> jnp.ndarray:
    """Evaluate the existing scalar energy kernel for stacked candidates once.

    The kernel arithmetic is unchanged.  The outer ``vmap`` removes the n_candidates
    per-candidate device dispatches and host scalar conversions.
    """
    return jax.vmap(
        lambda candidate: energy_distance(
            target, candidate, exponent=exponent, metric=metric
        )
    )(candidates)


def compute_energy_distance_to_candidates(
    target_samples: ArrayLike,
    candidate_samples: Sequence[object],
) -> jax.Array:
    """Return the energy metric ``sqrt(E^2)`` to each candidate."""
    if not candidate_samples:
        return jnp.empty(0, dtype=jnp.float64)

    target = jnp.asarray(target_samples, dtype=jnp.float64)
    candidates = jnp.stack(
        [jnp.asarray(candidate, dtype=jnp.float64) for candidate in candidate_samples]
    )
    energy_sq = _energy_distance_to_candidates_batched(target, candidates)
    return jnp.sqrt(jnp.maximum(energy_sq, 0.0))


def run_target(
    target_idx: int,
    all_betas: ArrayLike,
    all_samples: Sequence[object],
    solver: str,
    tol: float,
    maxiter: int,
    eps: float = 1e-12,
    precomputed_matrix: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Run hedging experiment for one target asset.

    Args:
        target_idx: Index of the target asset within the panel.
        all_betas: Asset factor-loading matrix.
        all_samples: Per-asset empirical return samples.
        solver: Barycentre solver name.
        tol: Solver convergence tolerance.
        maxiter: Maximum solver iterations.
        eps: Numerical floor used for distance ratios.
        precomputed_matrix: Optional full ``(n_assets, n_assets)`` pairwise
            mean-distance matrix (see
            :func:`jcor.discrepancy.energy.mean_distance_matrix`)
            for the cell's ``all_samples``, computed once and shared across
            targets. When supplied, the target's Gram/cross/within energy
            terms are sliced from it instead of rebuilt from scratch,
            eliminating the redundant O(n) per-target Gram construction that
            otherwise makes a cell O(n^3).

    Raises:
        RuntimeError: If the exact barycentre solver does not converge.
        ValueError: If the empirical candidate Gram matrix fails the CND check.

    """
    n_assets = len(all_samples)
    candidate_indices = [i for i in range(n_assets) if i != target_idx]

    target_samples = jnp.asarray(all_samples[target_idx], dtype=jnp.float64)
    candidate_samples = [
        jnp.asarray(all_samples[j], dtype=jnp.float64) for j in candidate_indices
    ]

    betas = jnp.asarray(all_betas)
    beta_target = betas[target_idx]
    betas_cand = betas[jnp.asarray(candidate_indices)]
    n_candidates = len(candidate_indices)

    candidate_arrays = [jnp.asarray(c) for c in candidate_samples]

    precomputed_components = None
    precomputed_within = None
    if precomputed_matrix is not None:
        candidate_idx_arr = jnp.asarray(candidate_indices)
        xx_target = precomputed_matrix[jnp.ix_(candidate_idx_arr, candidate_idx_arr)]
        yx_target = precomputed_matrix[target_idx, candidate_idx_arr]
        within_t = precomputed_matrix[target_idx, target_idx]
        precomputed_components = (xx_target, yx_target)
        precomputed_within = within_t

    w_opt_jax, diagnostics = energy_barycentre_weights_qp(
        jnp.asarray(target_samples),
        candidate_arrays,
        metric="euclidean",
        exponent=1.0,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        validate_cnd=True,
        precomputed_components=precomputed_components,
    )
    if not diagnostics.converged:
        message = (
            "energy barycentre solver failed to converge "
            f"for target {target_idx}: iterations={diagnostics.iterations}, "
            f"Frank-Wolfe gap={diagnostics.frank_wolfe_gap:.3e}"
        )
        raise RuntimeError(message)
    return _postprocess_target(
        target_idx=target_idx,
        w_opt_jax=w_opt_jax,
        diagnostics=diagnostics,
        target_samples=target_samples,
        candidate_samples=candidate_samples,
        candidate_arrays=candidate_arrays,
        beta_target=beta_target,
        betas_cand=betas_cand,
        n_candidates=n_candidates,
        eps=eps,
        precomputed_components=precomputed_components,
        precomputed_within=precomputed_within,
    )


def _postprocess_target(
    target_idx: int,
    w_opt_jax: jnp.ndarray,
    diagnostics: EnergyBarycentreDiagnostics,
    target_samples: ArrayLike,
    candidate_samples: Sequence[object],
    candidate_arrays: list[jnp.ndarray],
    beta_target: ArrayLike,
    betas_cand: ArrayLike,
    n_candidates: int,
    eps: float,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None,
    precomputed_within: jnp.ndarray | None,
) -> dict[str, Any]:
    """Shared post-solve summary computation for one target.

    Factored out of :func:`run_target` so the vmap-batched cell path (see
    :func:`_hedging_error_cell_vmap_batched`) can reuse the identical,
    non-solver arithmetic per lane after computing all lanes' weights in one
    batched QP call — only the solver dispatch differs between the two
    paths.
    """
    w_opt = jnp.asarray(w_opt_jax)
    beta_target = jnp.asarray(beta_target)
    betas_cand = jnp.asarray(betas_cand)

    w_neg_violation = float(jnp.sum(jnp.minimum(w_opt, 0.0)))
    w_sum = float(jnp.sum(w_opt))

    d_vec = compute_energy_distance_to_candidates(target_samples, candidate_samples)

    beta_opt = betas_cand.T @ w_opt
    te_opt_jax = jnp.linalg.norm(beta_target - beta_opt)

    w_eq = jnp.ones(n_candidates) / n_candidates
    beta_eq = betas_cand.T @ w_eq
    te_equal_jax = jnp.linalg.norm(beta_target - beta_eq)

    te_single = jnp.linalg.norm(beta_target[None, :] - betas_cand, axis=1)
    te_best_single_jax = jnp.min(te_single) if n_candidates else jnp.asarray(0.0)

    safe_d_vec = jnp.where(d_vec > eps, d_vec, 1.0)
    ratios = jnp.where(d_vec > eps, te_single / safe_d_vec, 0.0)
    lipschitz_jax = jnp.max(ratios) if n_candidates else jnp.asarray(0.0)

    weighted_pairwise_distance_jax = jnp.dot(w_opt, d_vec)
    bound_rhs_jax = lipschitz_jax * weighted_pairwise_distance_jax
    bound_slack_jax = bound_rhs_jax - te_opt_jax

    mixture_energy_opt_jax = energy_distance_kernel_mixture(
        jnp.asarray(target_samples),
        candidate_arrays,
        w_opt_jax,
        metric="euclidean",
        exponent=1.0,
        precomputed_components=precomputed_components,
        precomputed_within=precomputed_within,
    )
    mixture_energy_equal_jax = energy_distance_kernel_mixture(
        jnp.asarray(target_samples),
        candidate_arrays,
        w_eq,
        metric="euclidean",
        exponent=1.0,
        precomputed_components=precomputed_components,
        precomputed_within=precomputed_within,
    )
    energy_scale = jnp.maximum(jnp.abs(mixture_energy_equal_jax), eps)
    mixture_energy_improvement_pct_jax = (
        (mixture_energy_equal_jax - mixture_energy_opt_jax) / energy_scale * 100.0
    )

    return {
        "target_idx": target_idx,
        "K": n_candidates,
        "optimizer_method": "exact_energy_barycentre_qp",
        "w_sum": w_sum,
        "w_neg_violation": w_neg_violation,
        "TE_opt": float(te_opt_jax),
        "TE_equal": float(te_equal_jax),
        "TE_best_single": float(te_best_single_jax),
        "bound_rhs": float(bound_rhs_jax),
        "bound_slack": float(bound_slack_jax),
        "L": float(lipschitz_jax),
        "weighted_pairwise_distance": float(weighted_pairwise_distance_jax),
        "mixture_energy_opt": float(mixture_energy_opt_jax),
        "mixture_energy_equal": float(mixture_energy_equal_jax),
        "mixture_energy_improvement_pct": float(mixture_energy_improvement_pct_jax),
        "solver_converged": diagnostics.converged,
        "solver_iterations": diagnostics.iterations,
        "solver_energy": diagnostics.energy,
        "solver_initial_energy": diagnostics.initial_energy,
        "solver_frank_wolfe_gap": diagnostics.frank_wolfe_gap,
        "solver_projected_gradient_norm": diagnostics.projected_gradient_norm,
        "solver_cnd_tangent_max_eigenvalue": (diagnostics.cnd_tangent_max_eigenvalue),
        "solver_tangent_curvature_min": diagnostics.tangent_curvature_min,
        "solver_simplex_sum_error": diagnostics.simplex_sum_error,
        "solver_min_weight": diagnostics.min_weight,
    }


def _hedging_error_cell_vmap_batched(
    n_assets: int,
    d: int,
    betas: jax.Array,
    sample_sets: list[jax.Array],
    pairwise_matrix: jnp.ndarray,
    qp_tol: float,
    qp_maxiter: int,
) -> list[dict[str, Any]]:
    """Batched (opt-in, EXPERIMENTAL) variant of the per-target loop.

    Solves all ``n_assets`` targets' exact barycentre QPs in one
    ``jax.vmap`` call (:func:`jcor.geometry.energy_barycentre_weights_batched`)
    instead of a Python ``for`` loop over :func:`run_target`, then reuses
    :func:`_postprocess_target` per lane for the identical non-solver
    arithmetic.

    .. warning::
        NOT bit-identical to the serial path. An A/B harness
        (``scratch/ab_hedging_vmap.py`` in the t14 working tree) measured
        per-lane drift of the QP weights and every diagnostic field on the
        order of ~1e-15 (float64) to ~1e-6 (float32) relative to the serial
        loop — bounded floating-point summation-order drift from ``qpax``'s
        ``lax.while_loop`` running to the batch-max iteration count rather
        than each lane's own count, not a correctness bug, but NOT the
        bit-for-bit reproducibility the certified gate assumes. One
        float32 configuration in that harness additionally flipped a
        single lane's boolean ``converged`` flag relative to the serial
        run. This path must stay opt-in (``hedging_error_vmap_batch: true``
        in cfg) and must never be treated as a drop-in replacement for the
        certified default without a fresh certification rerun against its
        own outputs.

    Raises:
        RuntimeError: If any target's exact barycentre solver does not
            converge (post-hoc all-lanes check; the message names every
            failing target index).

    """
    candidate_indices_by_target = [
        [i for i in range(n_assets) if i != t] for t in range(n_assets)
    ]
    grams = jnp.stack(
        [
            pairwise_matrix[jnp.ix_(jnp.asarray(idx), jnp.asarray(idx))]
            for idx in candidate_indices_by_target
        ]
    )
    crosses = jnp.stack(
        [
            pairwise_matrix[t, jnp.asarray(idx)]
            for t, idx in enumerate(candidate_indices_by_target)
        ]
    )
    target_selfs = jnp.stack([pairwise_matrix[t, t] for t in range(n_assets)])

    weights, diagnostics = energy_barycentre_weights_batched(
        grams,
        crosses,
        target_selfs,
        tol=qp_tol,
        maxiter=qp_maxiter,
        validate_cnd=True,
    )

    # The batched solver returns length-``n_assets`` array diagnostic fields;
    # index the JAX arrays directly for per-lane checks.
    converged_lanes = jnp.asarray(diagnostics.converged)
    fw_gap_lanes = jnp.asarray(diagnostics.frank_wolfe_gap)
    failed = [t for t in range(n_assets) if not bool(converged_lanes[t])]
    if failed:
        gaps = ", ".join(
            f"target={t}: Frank-Wolfe gap={float(fw_gap_lanes[t]):.3e}" for t in failed
        )
        message = (
            "energy barycentre solver failed to converge (vmap-batched path) "
            f"for targets {failed} ({gaps})"
        )
        raise RuntimeError(message)

    rows: list[dict[str, Any]] = []
    for t in range(n_assets):
        candidate_indices = candidate_indices_by_target[t]
        target_samples = jnp.asarray(sample_sets[t], dtype=jnp.float64)
        candidate_samples = [
            jnp.asarray(sample_sets[j], dtype=jnp.float64) for j in candidate_indices
        ]
        candidate_arrays = [jnp.asarray(c) for c in candidate_samples]
        lane_diagnostics = EnergyBarycentreDiagnostics(
            *(
                getattr(diagnostics, f)[t].item()
                for f in EnergyBarycentreDiagnostics._fields
            )
        )
        result = _postprocess_target(
            target_idx=t,
            w_opt_jax=weights[t],
            diagnostics=lane_diagnostics,
            target_samples=target_samples,
            candidate_samples=candidate_samples,
            candidate_arrays=candidate_arrays,
            beta_target=betas[t],
            betas_cand=betas[jnp.asarray(candidate_indices)],
            n_candidates=len(candidate_indices),
            eps=1e-12,
            precomputed_components=(grams[t], crosses[t]),
            precomputed_within=target_selfs[t],
        )
        rows.append({"n_assets": n_assets, "embedding_dim": d, **result})

    return rows


def _hedging_error_cell(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """One (n_assets, d) cell of :func:`run_hedging_error` (picklable worker)."""
    (
        n_assets,
        d,
        base_seed,
        n_points,
        n_clusters,
        solver,
        qp_tol,
        qp_maxiter,
        vmap_batch,
    ) = task

    key = cell_key(
        base_seed,
        "mc_hedging_error",
        "n_assets",
        n_assets,
        "embedding_dim",
        d,
    )
    sample = cluster_embeddings(key, n_assets, n_points, d, n_clusters)
    betas = sample.betas
    sample_sets = [jnp.asarray(points, dtype=jnp.float64) for points in sample.samples]

    # Full pairwise mean-distance matrix, computed ONCE per cell. Each
    # target's Gram/cross/within energy terms are sliced from this matrix
    # (see run_target) instead of being rebuilt per target, which removes
    # the redundant O(n) rebuild that made a cell O(n^3) overall.
    pairwise_matrix = mean_distance_matrix(
        [jnp.asarray(s, dtype=jnp.float64) for s in sample_sets],
        exponent=1.0,
        metric="euclidean",
    )

    if vmap_batch:
        return _hedging_error_cell_vmap_batched(
            n_assets, d, betas, sample_sets, pairwise_matrix, qp_tol, qp_maxiter
        )

    rows: list[dict[str, Any]] = []
    for t in range(n_assets):
        result = run_target(
            target_idx=t,
            all_betas=betas,
            all_samples=sample_sets,
            solver=solver,
            tol=qp_tol,
            maxiter=qp_maxiter,
            eps=1e-12,
            precomputed_matrix=pairwise_matrix,
        )
        rows.append(
            {
                "n_assets": n_assets,
                "embedding_dim": d,
                **result,
            }
        )

    return rows


def run_hedging_error(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the hedging-error-bound Monte-Carlo validation.

    Args:
        cfg: ``params["monte_carlo"]`` dict from ``params.yaml``.

    Returns:
        (results_df, summary) — DataFrame of per-target results and summary dict.

    """
    n_assets_list: list[int] = cfg["n_assets"]
    embedding_dims: list[int] = cfg["embedding_dim"]
    n_points: int = cfg["n_points_per_asset"]
    n_clusters: int = cfg["n_clusters"]
    tolerance: float = cfg["tolerance"]
    base_seed: int = cfg["seed"]
    solver: str = cfg.get("hedging_error_solver", "boxosqp")
    qp_tol: float = cfg.get("hedging_error_tol", 1e-6)
    qp_maxiter: int = cfg.get("hedging_error_maxiter", 1000)
    # EXPERIMENTAL, opt-in only (default False): batches the per-cell target
    # loop's exact-barycentre QP solves via jax.vmap
    # (jcor.geometry.energy_barycentre_weights_batched) instead of a Python
    # for loop over run_target. NOT bit-identical to the serial path — an
    # A/B harness measured bounded but nonzero per-lane drift (~1e-15 f64,
    # up to ~1e-6 f32, and one f32 configuration flipped a lane's converged
    # flag). Do not enable for certified runs without a fresh certification
    # rerun against the batched path's own outputs. See
    # _hedging_error_cell_vmap_batched's docstring for detail.
    vmap_batch: bool = bool(cfg.get("hedging_error_vmap_batch", False))

    total_qp_failures = 0
    cells = [
        (
            n_assets,
            d,
            base_seed,
            n_points,
            n_clusters,
            solver,
            qp_tol,
            qp_maxiter,
            vmap_batch,
        )
        for n_assets in n_assets_list
        for d in embedding_dims
    ]

    cell_rows = [_hedging_error_cell(cell) for cell in cells]
    rows = [r for cr in cell_rows for r in cr]

    results_df = pd.DataFrame(rows)

    tol_abs = tolerance
    bound_violations = int((results_df["bound_slack"] < -tol_abs).sum())

    results_df["TE_improvement_pct"] = (
        (results_df["TE_equal"] - results_df["TE_opt"])
        / results_df["TE_equal"].clip(lower=1e-12)
    ) * 100.0

    median_te_improvement_pct = float(results_df["TE_improvement_pct"].median())
    corr_de_te = float(results_df["mixture_energy_opt"].corr(results_df["TE_opt"]))

    summary = {
        "optimizer_method": "exact_energy_barycentre_qp",
        "total_targets_tested": len(results_df),
        "total_qp_failures": total_qp_failures,
        "qp_failure_rate": (
            total_qp_failures / max(1, len(results_df) + total_qp_failures)
        ),
        "bound_violations": bound_violations,
        "median_TE_improvement_opt_vs_equal_pct": median_te_improvement_pct,
        "correlation_achieved_energy_dist_vs_TE": corr_de_te,
        "median_mixture_energy_improvement_opt_vs_equal_pct": float(
            results_df["mixture_energy_improvement_pct"].median()
        ),
        "mean_mixture_energy_opt": float(results_df["mixture_energy_opt"].mean()),
        "mean_mixture_energy_equal": float(results_df["mixture_energy_equal"].mean()),
        "mean_TE_opt": float(results_df["TE_opt"].mean()),
        "mean_TE_equal": float(results_df["TE_equal"].mean()),
        "mean_TE_best_single": float(results_df["TE_best_single"].mean()),
        "mean_bound_slack": float(results_df["bound_slack"].mean()),
        "min_bound_slack": float(results_df["bound_slack"].min()),
        "max_solver_frank_wolfe_gap": float(results_df["solver_frank_wolfe_gap"].max()),
        "max_solver_cnd_tangent_eigenvalue": float(
            results_df["solver_cnd_tangent_max_eigenvalue"].max()
        ),
        "max_solver_simplex_sum_error": float(
            results_df["solver_simplex_sum_error"].max()
        ),
        "min_solver_weight": float(results_df["solver_min_weight"].min()),
        "tolerance": tolerance,
        "parameters": {
            "n_assets": n_assets_list,
            "embedding_dim": embedding_dims,
            "n_points_per_asset": n_points,
            "n_clusters": n_clusters,
            "seed": base_seed,
            "hedging_error_solver": solver,
            "hedging_error_tol": qp_tol,
            "hedging_error_maxiter": qp_maxiter,
            "hedging_error_vmap_batch": vmap_batch,
        },
    }

    return results_df, summary
