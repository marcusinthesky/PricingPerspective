"""Persist bounded inference checks without granting corporate eligibility."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from hotelling.dynamics.artifacts import file_hash, write_arrays

from pipeline._kernels.paper6.inference_audit import (
    FIXTURE_FIRMS,
    FIXTURE_REPLICATIONS,
    FIXTURE_SEED,
    FIXTURE_VARIANCE_FACTOR,
    ONE_SIDED_Z,
    bound_diagnostics,
    iid_replications,
)
from pipeline.stages.papers.paper6.artifacts import write

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _number(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _history_stress(
    history: int, redraws: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Separate a constructed null from arithmetic residue and missing targets."""
    targets = np.array([_number(row, "oracle_beta") for row in redraws])
    available_targets = targets[np.isfinite(targets)]
    if len(available_targets) and not np.all(available_targets == available_targets[0]):
        msg = "conditional audit requires one saved target per history"
        raise ValueError(msg)
    flags = {
        bool(row["oracle_constructed_zero"])
        for row in redraws
        if "oracle_constructed_zero" in row
    }
    if len(flags) > 1:
        msg = "constructed-zero identity changed within a fixed history"
        raise ValueError(msg)
    constructed_zero = flags == {True}
    target = (
        (0.0 if constructed_zero else float(available_targets[0]))
        if len(available_targets)
        else None
    )
    diagnostics = bound_diagnostics(
        [_number(row, "beta") for row in redraws],
        [_number(row, "candidate_se") for row in redraws],
        0.0 if target is None else target,
    )
    arithmetic = np.array(
        [
            _number(row, "oracle_arithmetic_beta")
            if "oracle_arithmetic_beta" in row
            else _number(row, "oracle_beta")
            for row in redraws
        ]
    )
    available_arithmetic = arithmetic[np.isfinite(arithmetic)]
    population_se = np.array(
        [_number(row, "exact_population_candidate_se") for row in redraws]
    )
    available_se = population_se[np.isfinite(population_se)]
    result: dict[str, Any] = {
        "history": history,
        **diagnostics,
        "target": target,
        "target_status": "available" if target is not None else "unavailable",
        "target_unavailable_reason": (
            "No finite population oracle was produced for this history."
            if target is None
            else None
        ),
        "oracle_constructed_zero": constructed_zero,
        "arithmetic_oracle_beta": (
            float(available_arithmetic[0]) if len(available_arithmetic) else None
        ),
        "missing_oracle_attempts": int((~np.isfinite(targets)).sum()),
        "exact_population_candidate_se": (
            float(available_se[0]) if len(available_se) else None
        ),
        "exact_population_conditional_measurement_sd": (
            0.0 if target is not None else None
        ),
        "positive_bound_interpretation": (
            "Unavailable: the population target could not be computed."
            if target is None
            else "Experimental-rule rejection at the conditional zero target."
            if target == 0
            else "Positive-bound frequency at a nonzero reported-law "
            "target; not a Type I error rate."
        ),
    }
    if target is None:
        result["target_containment"] = None
        result["positive_bound"] = None
    return result


def _stratified_rates(
    per_history: Sequence[Mapping[str, Any]], *, attempts: int
) -> dict[str, object]:
    """Withhold an aggregate if any fixed stratum lacks an assessable target."""
    aggregated: dict[str, object] = {}
    for field in ("target_containment", "positive_bound"):
        unavailable = [
            record["history"] for record in per_history if record[field] is None
        ]
        rates = [record[field] for record in per_history]
        aggregated[field] = {
            "equal_history_rate": (
                None
                if unavailable
                else float(np.mean([rate["rate"] for rate in rates]))
            ),
            "mcse": (
                None
                if unavailable
                else float(
                    np.sqrt(sum(rate["mcse"] ** 2 for rate in rates)) / len(rates)
                )
            ),
            "histories": len(per_history),
            "attempts": attempts,
            "unavailable_histories": unavailable,
            "unavailable_reason": (
                "At least one declared history has no finite population target."
                if unavailable
                else None
            ),
            "sampling": (
                "Within-history independent article redraws; histories enumerated."
            ),
        }
    return aggregated


def _conditional_stress(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Preserve fixed-history strata when summarizing the conditional stress test."""
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row["scenario_id"]), str(row["regime"]), str(row["budget"])
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for (scenario, regime, budget), members in sorted(groups.items()):
        histories: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in members:
            histories[int(str(row["history"]))].append(row)
        per_history = [
            _history_stress(history, redraws)
            for history, redraws in sorted(histories.items())
        ]
        # Histories are fixed strata. Only redraw variation contributes MCSE.
        aggregated_rates = _stratified_rates(per_history, attempts=len(members))
        summaries.append(
            {
                "scenario_id": scenario,
                "regime": regime,
                "budget": budget,
                "sampling_target": "Fixed firms, histories and reporting laws.",
                "status": "target_mismatch_stress_diagnostics",
                "reason": (
                    "The retained candidate estimates iid-firm-history variation; "
                    "article redraws generate conditional measurement variation."
                ),
                "rates": aggregated_rates,
                "per_history": per_history,
            }
        )
    return summaries


def run_audit(
    output: Path,
    *,
    conditional_rows: Sequence[Mapping[str, object]],
    fixture_replications: int = FIXTURE_REPLICATIONS,
    fixture_firms: int = FIXTURE_FIRMS,
    seed: int = FIXTURE_SEED,
) -> dict[str, Any]:
    """Save iid-fixture calibration and separately labeled conditional diagnostics.

    This is composed into the directional-validation stage, whose manifest owns
    its semantic identity and result hashes. It does not read or rewrite any
    historical corporate/recovery artifact.
    """
    arrays = iid_replications(
        replications=fixture_replications, firms=fixture_firms, seed=seed
    )
    output.mkdir(parents=True, exist_ok=True)
    replicates_path = output / "fixture-replicates.npz"
    write_arrays(replicates_path, arrays)
    scenarios = []
    for index, beta in enumerate(arrays["targets"]):
        diagnostics = bound_diagnostics(
            arrays["estimates"][index], arrays["candidate_se"][index], float(beta)
        )
        scenarios.append(
            {
                **diagnostics,
                "sampling_target": "The iid-history fixture population pooled ratio.",
                "rate_interpretation": (
                    "Coverage and rejection of H0: beta <= 0 in this iid fixture only."
                ),
                "asymptotic_standard_error": float(
                    np.sqrt(FIXTURE_VARIANCE_FACTOR / fixture_firms)
                ),
            }
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": (
            "Existing complete-triple first projection; ratio derivative; Cov(IF)/N."
        ),
        "candidate_bound": "beta_hat - z_0.95 * candidate_se",
        "one_sided_z": ONE_SIDED_Z,
        "null": "H0: beta <= 0; H1: beta > 0",
        "iid_fixture": {
            "firms": fixture_firms,
            "years": 4,
            "replications_per_target": fixture_replications,
            "seed": seed,
            "stream": "SeedSequence([seed, replicate]); paired targets 0 and 0.5.",
            "law": (
                "X,E iid standard normal; U independent Rademacher per firm; "
                "R[t,i,j]=c[t]*X[j]; A=beta*R+g[t]*(E[j]+.5*E[i]*X[j])+t*U[i]; "
                "c=(1,2,1,2), g=(1,1,2,2). Variables persist across years and roles."
            ),
            "population_first_influence": ".9*E*(X+.5)",
            "asymptotic_variance": "1.0125 / N",
            "scenarios": scenarios,
            "replicates": replicates_path.name,
            "replicates_sha256": file_hash(replicates_path),
        },
        "conditional_stress": _conditional_stress(conditional_rows),
        "empirical_inference": {
            "eligible": False,
            "lower_bound": None,
            "p_value": None,
            "target": "Complete retained corporate archive projection beta_archive.",
            "reason": (
                "No matching iid-firm sampling design is established for the complete "
                "corporate archive. Article redraws condition on firms and histories; "
                "the retained covariance candidate concerns iid firm histories. "
                "Calibration in the fixture cannot transfer coverage to the archive."
            ),
            "permitted_uncertainty": (
                "Conditional article-selection sensitivity quantiles."
            ),
        },
        "decision": (
            "Corporate evidence remains descriptive with article-selection sensitivity."
        ),
    }
    write(output / "inference-status.json", result)
    return result
