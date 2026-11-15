"""Tests for Paper 5's numerical Perron and finite-rho witnesses."""

from __future__ import annotations

import numpy as np
import pytest
from pipeline.stages.papers.paper5.network_certificate import (
    _factor_conditioned_w2_rho,
    build_network_certificate,
    dobrushin_coefficient,
)

_CERTIFICATE_TOLERANCE = 1e-12


def _mapping(value: object, label: str) -> dict[str, object]:
    """Validate a serialized certificate object before inspecting its fields."""
    if not isinstance(value, dict):
        message = f"{label} must be a JSON object"
        raise TypeError(message)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            message = f"{label} has a non-string key"
            raise TypeError(message)
        result[key] = item
    return result


def _float_field(record: dict[str, object], field: str) -> float:
    """Return a numeric certificate field while rejecting booleans and objects."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{field} must be numeric"
        raise TypeError(message)
    return float(value)


def test_positive_stochastic_operator_certifies_perron_and_finite_rho() -> None:
    """A positive second power supplies both Perron and resolvent witnesses."""
    weights = np.array(
        [
            [0.0, 0.7, 0.3],
            [0.2, 0.0, 0.8],
            [0.6, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    certificate = build_network_certificate(weights, {"w_flat": 0.75})

    structural = _mapping(certificate["structural"], "structural")
    perron = _mapping(certificate["perron"], "perron")
    finite_rho = _mapping(certificate["finite_rho"], "finite_rho")
    finite = _mapping(finite_rho["w_flat"], "finite_rho.w_flat")
    assert structural["w_squared_strictly_positive"] is True
    assert _float_field(perron, "dobrushin_coefficient") == pytest.approx(0.8)
    residual_tolerance = 1e-11
    assert _float_field(perron, "stationary_l1_residual") < residual_tolerance
    assert _float_field(finite, "normalized_resolvent_error_linf") <= _float_field(
        finite, "dobrushin_upper_bound_linf"
    )
    assert (
        _float_field(finite, "normalized_common_mode_identity_error")
        < _CERTIFICATE_TOLERANCE
    )
    assert (
        _float_field(finite, "resolvent_common_mode_identity_error")
        < _CERTIFICATE_TOLERANCE
    )
    assert _float_field(finite, "normalized_common_mode_mean") == pytest.approx(1.0)
    assert _float_field(finite, "own_exposure_weight_mean") + _float_field(
        finite, "other_exposure_weight_mean"
    ) == pytest.approx(1.0)


def test_dobrushin_detects_noncontracting_identity() -> None:
    """The identity matrix has no total-variation contraction."""
    assert dobrushin_coefficient(np.eye(3, dtype=np.float64)) == pytest.approx(1.0)


def test_factor_conditioned_extractor_selects_primary_spherical_w2() -> None:
    """Publication witnesses use the factor-conditioned spherical W2 estimate."""
    payload = {
        "model_rows": [
            {
                "scope": "spatial_value",
                "variance_model": "spherical",
                "model": "factor_only",
                "rho_hat": 0.0,
            },
            {
                "scope": "spatial_value",
                "variance_model": "spherical",
                "model": "w2",
                "rho_hat": 0.6865955453727131,
            },
        ]
    }
    assert _factor_conditioned_w2_rho(payload) == pytest.approx(0.6865955453727131)


def test_certificate_rejects_nonprimitive_operator() -> None:
    """A periodic two-cycle cannot pass the strict-positive-square witness."""
    with pytest.raises(ValueError, match="W squared"):
        build_network_certificate(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            {"w_flat": 0.5},
        )
