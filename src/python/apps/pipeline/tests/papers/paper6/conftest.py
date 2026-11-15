"""Saved-record fixtures for semantic publication, including adverse results."""

from typing import Any

import pytest


@pytest.fixture
def validation_extension() -> dict[str, Any]:
    """Supply new scenario and inference records without running any estimator."""
    scenarios = []
    identities = [
        ("A", regime, budget, "beta0_h")
        for regime in ("common", "changing")
        for budget in (1, 4, 16)
    ]
    identities += [("B", "common", budget, "beta_lambda_h") for budget in (1, 4, 16)]
    identities += [
        (scenario, "common", 1, "beta0_h") for scenario in ("D_half", "D_double")
    ]
    identities.append(("C", "reporting_0.005", 1, "beta_lambda_h"))
    for scenario, regime, budget, target in identities:
        stationary = scenario == "B"
        scenarios.append(
            {
                "scenario_id": scenario,
                "regime": regime,
                "budget": budget,
                "target": target,
                "correct_sign_rate": None if stationary else 0.4,
                "correct_sign_mcse": None if stationary else 0.009,
                "correct_sign_denominator": None if stationary else 2560,
                "wrong_sign": None
                if stationary
                else {"rate": 0.6, "mcse": 0.009, "denominator": 2560},
                "positive_rate": 0.49 if stationary else 0.4,
                "positive_mcse": 0.009,
                "positive_denominator": 2560,
                "rmse": 12.0,
                "mean_oracle_ratio": 0.0 if stationary else 0.5,
                "normalized_rmse": None if stationary else 9.0,
                "normalized_rmse_mcse": None if stationary else 0.2,
                "normalization_reason": "zero_oracle_scale" if stationary else None,
                "sign_ineligibility_reason": (
                    "zero_or_unresolved_oracle" if stationary else None
                ),
                "undefined": 3,
                "total": 2560,
                "failed_premises": 160,
                "domain_departures": 1,
            }
        )
    inference_scenarios = []
    for target, rejection, count in ((0.0, 0.2, 500), (0.5, 0.6, 1500)):
        inference_scenarios.append(
            {
                "target": target,
                "average_candidate_se": 0.15,
                "empirical_dispersion": 0.3,
                "candidate_se_to_dispersion": 0.5,
                "target_containment": {
                    "rate": 0.8,
                    "mcse": 0.008,
                    "count": 2000,
                    "attempts": 2500,
                },
                "positive_bound": {
                    "rate": rejection,
                    "mcse": 0.008,
                    "count": count,
                    "attempts": 2500,
                },
            }
        )
    return {
        "histories": 16,
        "redraws": 160,
        "scenarios": scenarios,
        "inference_status": {
            "one_sided_z": 1.6448536269514715,
            "iid_fixture": {
                "firms": 16,
                "years": 4,
                "replications_per_target": 2500,
                "scenarios": inference_scenarios,
            },
            "empirical_inference": {
                "eligible": False,
                "lower_bound": None,
                "p_value": None,
                "reason": "No matching corporate sampling design is established.",
            },
        },
    }
