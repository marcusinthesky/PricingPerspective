"""Post-corpus evaluation window and annual folds for the Paper 5 stages."""

from __future__ import annotations

POST_CORPUS_START_DATE = "2023-01-01"
POST_CORPUS_END_DATE = "2026-04-06"

ANNUAL_FOLDS: dict[str, tuple[str, str]] = {
    "eval2023": (POST_CORPUS_START_DATE, "2023-12-31"),
    "eval2024": ("2024-01-01", "2024-12-31"),
    "eval2025": ("2025-01-01", "2025-12-31"),
    "eval2026": ("2026-01-01", POST_CORPUS_END_DATE),
}


__all__ = ["ANNUAL_FOLDS", "POST_CORPUS_END_DATE", "POST_CORPUS_START_DATE"]
