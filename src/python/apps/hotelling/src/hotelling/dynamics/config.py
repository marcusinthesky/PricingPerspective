"""Validated, serializable primitives for joint-response experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

MAX_SOBOL_DIMENSION = 21201
MAX_SEED = 2**32


@dataclass(frozen=True)
class DynamicsConfig:
    """Keep economic, quadrature and article dimensions distinct.

    Float64 is the numerical contract because the certified relative solve
    target is smaller than float32 machine precision. Command entrypoints select
    float64 explicitly; numerical helpers never change process configuration.
    """

    firms: int = 100
    activities: int = 16
    nodes: int = 256
    dimension: int = 4
    article_dimension: int | None = None
    periods: int = 60
    kappa: float = 1.0
    rho: float = 0.5
    discount: float = 0.95
    shock_size: float = 0.01
    seed: int = 0
    shock_kind: str = "mixed"
    own_curvature: float = 0.5
    congestion: float = 0.5
    outside: float = 0.1
    advantage: float = 0.1
    tau: float = 16.0
    solve_target: float = 1e-10
    max_iterations: int = 10000
    solve_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Reject invalid controls instead of silently repairing a design."""
        self._validate_counts()
        self._validate_scalars()
        if type(self.shock_kind) is not str or self.shock_kind not in {
            "opportunity",
            "cost",
            "mixed",
        }:
            msg = "shock_kind must be opportunity, cost or mixed"
            raise ValueError(msg)

    def _validate_counts(self) -> None:
        """Validate scientific counts and Sobol's supported seed/dimension domain."""
        for name in (
            "firms",
            "activities",
            "nodes",
            "dimension",
            "periods",
            "max_iterations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                msg = f"{name} must be a positive integer"
                raise ValueError(msg)
        if self.article_dimension is not None and (
            type(self.article_dimension) is not int or self.article_dimension <= 0
        ):
            msg = "article_dimension must be a positive integer or null"
            raise ValueError(msg)
        if self.dimension > MAX_SOBOL_DIMENSION:
            msg = f"Sobol dimension cannot exceed {MAX_SOBOL_DIMENSION}"
            raise ValueError(msg)
        if type(self.seed) is not int or not 0 <= self.seed < MAX_SEED:
            msg = "seed must be an integer in [0, 2**32)"
            raise ValueError(msg)

    def _validate_scalars(self) -> None:
        """Validate scales, curvature, persistence and numerical budgets."""
        positive = ("own_curvature", "outside", "solve_target", "solve_seconds")
        nonnegative = ("kappa", "shock_size", "congestion", "advantage", "tau")
        for name in (*positive, *nonnegative, "rho", "discount"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                msg = f"{name} must be a finite number"
                raise ValueError(msg)
            if name in positive and value <= 0:
                msg = f"{name} must be strictly positive"
                raise ValueError(msg)
            if name in nonnegative and value < 0:
                msg = f"{name} must be nonnegative"
                raise ValueError(msg)
        if not -1 < self.rho < 1 or not 0 < self.discount < 1:
            msg = "require |rho| < 1 and 0 < discount < 1"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DynamicsConfig:
        """Read a partial design, rejecting unknown keys and invalid types."""
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            msg = f"unknown dynamics controls: {sorted(unknown)}"
            raise ValueError(msg)
        return replace(cls(), **dict(value))

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return JSON-compatible controls including all explicit defaults."""
        return {item.name: getattr(self, item.name) for item in fields(self)}
