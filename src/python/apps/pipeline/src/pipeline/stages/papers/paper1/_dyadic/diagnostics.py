"""Paper-1 design, influence, and residual-scale diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from jcor.model.dyadic import build_dyadic_design
from jcor.operators.design import symmetric_dyadic_effects

from pipeline.stages.papers.paper1._dyadic.estimation import _fwl_display

if TYPE_CHECKING:
    from jcor.model.dyadic import DyadicModel

_build_design = build_dyadic_design
_symmetric_firm_effects = symmetric_dyadic_effects


def _scaled_condition_number(x: np.ndarray) -> float:
    """Return the condition number after scaling non-intercept columns."""
    scaled = np.empty_like(x, dtype=float)
    scaled[:, 0] = 1.0 / np.sqrt(x.shape[0])
    for column in range(1, x.shape[1]):
        centered = x[:, column] - float(np.mean(x[:, column]))
        norm = float(np.linalg.norm(centered))
        scaled[:, column] = centered / norm if norm > 0.0 else 0.0
    return float(np.linalg.cond(scaled))


def _design_diagnostic_row(
    spec_index: int,
    model: DyadicModel[str],
    *,
    firm_effects: bool,
    focal_name: str = "energy_distance",
) -> dict[str, object]:
    """Return reviewer-facing design conditioning for one ladder rung."""
    jax_design, names, dropped = _build_design(
        model.regressor_names,
        model.columns,
        model.sample.endpoint_i,
        model.sample.endpoint_j,
        node_effects=firm_effects,
        effect_name_prefix="firm_fe",
    )
    # Land the design on the host before any of the NumPy algebra below. Since
    # t65 `build_dyadic_design` returns a `jax.Array` built under an internal
    # `jax.enable_x64` scope; outside that scope it keeps its float64 dtype
    # label but computes its next op in float32, so `energy.mean()` and the
    # residual products silently lost precision and pushed `energy_vif` 8.5e-07
    # off the retired NumPy path.
    design = np.asarray(jax_design, dtype=np.float64)
    focal_position = names.index(focal_name)
    nuisance = np.delete(design, focal_position, axis=1)
    focal = design[:, focal_position]
    residual = focal - nuisance @ np.linalg.lstsq(nuisance, focal, rcond=None)[0]
    total = float(np.sum((focal - focal.mean()) ** 2))
    residual_ss = float(residual @ residual)
    residual_share = residual_ss / total if total > 0.0 else 0.0
    vif = 1.0 / residual_share if residual_share > 0.0 else float("inf")
    firms = sorted(
        set(model.sample.endpoint_i.tolist()) | set(model.sample.endpoint_j.tolist())
    )
    degrees = {
        firm: int(
            np.sum(
                (model.sample.endpoint_i == firm) | (model.sample.endpoint_j == firm)
            )
        )
        for firm in firms
    }
    n_dyads = len(model.sample.endpoint_i)
    total_dyad_pairs = n_dyads * (n_dyads - 1) / 2
    shared_endpoint_pairs = sum(
        degree * (degree - 1) / 2 for degree in degrees.values()
    )
    result: dict[str, object] = {
        "spec_index": int(spec_index),
        "block": "firm_effects" if firm_effects else "pooled",
        "n_nodes": len(firms),
        "n_dyads": int(n_dyads),
        "degree_min": int(min(degrees.values())),
        "degree_max": int(max(degrees.values())),
        "endpoint_overlap_share": float(shared_endpoint_pairs / total_dyad_pairs),
        "design_columns": int(design.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "scaled_condition_number": _scaled_condition_number(design),
        "focal_regressor": focal_name,
        "focal_vif": float(vif),
        "focal_fwl_sd_ratio": float(np.sqrt(max(residual_share, 0.0))),
        "dropped_columns": dropped,
    }
    if focal_name == "energy_distance":
        result["energy_vif"] = float(vif)
        result["energy_fwl_sd_ratio"] = float(np.sqrt(max(residual_share, 0.0)))
    return result


def _primary_fit_frame(
    design: np.ndarray,
    beta: np.ndarray,
    outcome: np.ndarray,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
) -> pd.DataFrame:
    """Persist point-fit ingredients needed by post-fit diagnostics."""
    fitted = design @ beta
    residual = outcome - fitted
    orthogonal, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(orthogonal * orthogonal, axis=1)
    degrees_of_freedom = max(len(outcome) - design.shape[1], 1)
    mse = float(residual @ residual) / degrees_of_freedom
    denominator = np.sqrt(np.maximum(mse * (1.0 - leverage), np.finfo(float).eps))
    return pd.DataFrame(
        {
            "ticker_i": firm_i,
            "ticker_j": firm_j,
            "outcome": outcome,
            "fitted": fitted,
            "residual": residual,
            "leverage": leverage,
            "studentized_residual": residual / denominator,
        }
    )


def _assemble_primary_fit_frame(
    primary_fit: dict[str, object],
    outcome: np.ndarray,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
    functional_display: pd.DataFrame,
    focal_name: str = "energy_distance",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the persisted per-dyad frame and FWL audit payload."""
    design = cast("np.ndarray", primary_fit["_point_design"])
    names = cast("list[str]", primary_fit["_point_design_names"])
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    frame = _primary_fit_frame(design, beta, outcome, firm_i, firm_j)
    for name in functional_display.columns:
        frame[name] = functional_display[name].to_numpy()
    fwl_display, fwl_diagnostics = _fwl_display(
        design,
        names,
        outcome,
        min(set(firm_i.tolist()) | set(firm_j.tolist())),
        focal_name,
    )
    for name in fwl_display.columns:
        frame[name] = fwl_display[name].to_numpy()
    reported = float(cast("float | int | str", primary_fit[f"coef_{focal_name}"]))
    fwl_diagnostics["reported_coefficient"] = reported
    fwl_diagnostics["abs_slope_reproduction_error"] = abs(
        float(cast("float", fwl_diagnostics["slope"])) - reported
    )
    return frame, fwl_diagnostics


def _influence_diagnostics(
    primary_fit: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    n_design_columns: int,
) -> dict[str, object]:
    """Summarize dyad leverage, studentization, and node-deletion stability."""
    max_leverage_row = primary_fit.loc[primary_fit["leverage"].idxmax()]
    max_studentized_row = primary_fit.loc[
        primary_fit["studentized_residual"].abs().idxmax()
    ]
    leverage_threshold = 2.0 * n_design_columns / len(primary_fit)
    max_leverage = float(cast("float | int | str", max_leverage_row["leverage"]))
    most_extreme_studentized = float(
        cast("float | int | str", max_studentized_row["studentized_residual"])
    )
    return {
        "max_abs_lofo_delta": float(leave_one_out["delta_beta"].abs().max()),
        "max_abs_lofo_delta_in_bootstrap_se": float(
            leave_one_out["delta_in_bootstrap_se"].abs().max()
        ),
        "lofo_sign_flip_count": int(leave_one_out["sign_flip"].sum()),
        "max_leverage": max_leverage,
        "max_leverage_dyad": (
            f"{max_leverage_row['ticker_i']}--{max_leverage_row['ticker_j']}"
        ),
        "leverage_flag_threshold_2p_over_n": float(leverage_threshold),
        "leverage_exceeds_heuristic": bool(max_leverage > leverage_threshold),
        "leverage_threshold_scope": (
            "descriptive 2p/n screening heuristic; not a test or rejection cutoff"
        ),
        "most_extreme_studentized_residual": most_extreme_studentized,
        "most_extreme_studentized_residual_dyad": (
            f"{max_studentized_row['ticker_i']}--{max_studentized_row['ticker_j']}"
        ),
    }


def _point_focal_fit(
    columns: dict[str, np.ndarray],
    outcome: np.ndarray,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
    focal_name: str,
) -> tuple[float, float]:
    """Return the focal symmetric-FE fit for a leave-one-firm-out subgraph."""
    jax_design, names, _ = _build_design(
        [focal_name],
        columns,
        firm_i,
        firm_j,
        node_effects=True,
        effect_name_prefix="firm_fe",
    )
    # Same host landing as `_design_diagnostic_row`: `within_r2` differences two
    # near-equal residual sums, so a float32 product here showed up as 1.5e-07
    # absolute drift against the retired NumPy path.
    design = np.asarray(jax_design, dtype=np.float64)
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residual = outcome - design @ beta
    jax_effects, _ = _symmetric_firm_effects(firm_i, firm_j)
    effects = np.asarray(jax_effects, dtype=np.float64)
    effect_only = np.column_stack([np.ones(len(outcome)), effects])
    effect_residual = (
        outcome - effect_only @ np.linalg.lstsq(effect_only, outcome, rcond=None)[0]
    )
    denominator = float(effect_residual @ effect_residual)
    within_r2 = 1.0 - float(residual @ residual) / denominator
    return float(beta[names.index(focal_name)]), float(within_r2)


def _leave_one_firm_out(
    model: DyadicModel[str],
    full_beta: float,
    bootstrap_se: float,
    focal_name: str = "energy_distance",
) -> pd.DataFrame:
    """Return point fits after deleting each endpoint in turn."""
    rows: list[dict[str, object]] = []
    sample = model.sample
    firms = sorted(set(sample.endpoint_i.tolist()) | set(sample.endpoint_j.tolist()))
    for firm in firms:
        keep = (sample.endpoint_i != firm) & (sample.endpoint_j != firm)
        sub_columns = {name: values[keep] for name, values in model.columns.items()}
        beta, within_r2 = _point_focal_fit(
            sub_columns,
            sample.outcome[keep],
            sample.endpoint_i[keep],
            sample.endpoint_j[keep],
            focal_name,
        )
        delta = beta - full_beta
        row: dict[str, object] = {
            "left_out_ticker": firm,
            "n_nodes": int(len(firms) - 1),
            "n_dyads": int(np.sum(keep)),
            "beta_focal": beta,
            "delta_beta": delta,
            "delta_in_bootstrap_se": delta / bootstrap_se,
            "within_r2": within_r2,
            "sign_flip": bool(np.sign(beta) != np.sign(full_beta)),
        }
        if focal_name == "energy_distance":
            row["beta_energy"] = beta
        elif focal_name == "w2_distance":
            row["beta_w2"] = beta
        rows.append(row)
    return pd.DataFrame(rows)


def _shared_endpoint_scale_statistic(
    z: np.ndarray,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
) -> float:
    """Return squared-residual association for shared versus disjoint dyads."""
    centered = z - float(np.mean(z))
    total_sum = float(np.sum(centered))
    total_squared = float(centered @ centered)
    all_pair_sum = 0.5 * (total_sum * total_sum - total_squared)
    shared_sum = 0.0
    shared_count = 0
    for firm in sorted(set(firm_i.tolist()) | set(firm_j.tolist())):
        incident = centered[(firm_i == firm) | (firm_j == firm)]
        shared_sum += 0.5 * (float(np.sum(incident)) ** 2 - float(incident @ incident))
        shared_count += len(incident) * (len(incident) - 1) // 2
    all_count = len(centered) * (len(centered) - 1) // 2
    disjoint_count = all_count - shared_count
    shared_mean = shared_sum / shared_count
    disjoint_mean = (all_pair_sum - shared_sum) / disjoint_count
    scale = float(np.mean(centered * centered))
    return float((shared_mean - disjoint_mean) / scale) if scale > 0.0 else 0.0


def _residual_scale_diagnostic(
    primary_fit: pd.DataFrame,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
    *,
    n_permutations: int = 999,
    seed: int = 10_042,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run a descriptive permutation diagnostic for residual-scale dependence."""
    z = primary_fit["studentized_residual"].to_numpy(dtype=float) ** 2 - 1.0
    observed = _shared_endpoint_scale_statistic(z, firm_i, firm_j)
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _shared_endpoint_scale_statistic(rng.permutation(z), firm_i, firm_j)
            for _ in range(n_permutations)
        ],
        dtype=float,
    )
    pvalue = float((1 + np.sum(null >= observed)) / (1 + n_permutations))
    q95 = float(np.percentile(null, 95.0))
    summary: dict[str, object] = {
        "statistic": observed,
        "permutation_pvalue": pvalue,
        "null_q95": q95,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "failure": bool(observed > q95),
        "definition": (
            "standardized difference in leverage-adjusted squared-residual products "
            "for dyad pairs sharing an endpoint versus disjoint dyad pairs"
        ),
        "interpretation": (
            "descriptive endpoint-scale diagnostic; non-rejection is not proof of "
            "residual independence"
        ),
    }
    null_frame = pd.DataFrame(
        {
            "permutation_id": np.arange(n_permutations, dtype=np.int32),
            "statistic": null,
        }
    )
    return summary, null_frame
