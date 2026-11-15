"""Shared record shapes emitted by the Paper 5 held-out scoring stages.

Both the factor-conditioned two-field stage and the economic-value stage
summarize a paired held-out score contrast into the same record, which is then
serialized to JSON and read back by renderers, tests, and publication values.
Declaring the shape once keeps those consumers typed without casts and keeps the
two producers from drifting apart.
"""

from __future__ import annotations

from typing import TypedDict


class ScoreContrast(TypedDict):
    """Summary of one paired held-out score contrast."""

    estimate: float
    bootstrap_mean: float
    ci_lower: float
    ci_upper: float
    positive_share: float
    bootstrap_valid_draws: int
    unit: str


class PairedScoreContrast(ScoreContrast):
    """A score contrast that names the candidate and reference arms compared."""

    candidate: str
    reference: str
    confidence_level: float
