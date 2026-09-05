# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for the isolated EttaX pilot workflow."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from ettax.pooling import (
    ARTICLE_POLICIES,
    POOLING_POLICIES,
    ArticlePolicy,
    PoolingPolicy,
)

type PilotArm = Literal["V0", "V1", "V3"]

__all__ = ["ArticlePolicy", "PilotArm", "PilotConfig", "PilotTrial", "PoolingPolicy"]

_MIN_CONTEXT = 256
_MAX_CONTEXT = 2_048
_MIN_EVALUATION_ROWS = 3


@dataclass(frozen=True, slots=True)
class PilotTrial:
    """One predeclared training recipe in the pilot matrix."""

    name: str
    maximum_context: int | None
    training_buckets: tuple[int, ...]
    decoder_mask_rate: float = 0.0
    bow_weight: float = 0.0
    inherits_context_winner: bool = False

    def __post_init__(self) -> None:
        """Reject an ambiguous or scientifically invalid trial."""
        if not self.name or not self.training_buckets:
            raise ValueError("pilot trials require a name and training buckets")
        if tuple(sorted(set(self.training_buckets))) != self.training_buckets:
            raise ValueError("training buckets must be unique and increasing")
        if self.inherits_context_winner != (self.maximum_context is None):
            raise ValueError(
                "exactly one of maximum_context or inherits_context_winner is required"
            )
        if (
            self.maximum_context is not None
            and self.maximum_context != self.training_buckets[-1]
        ):
            raise ValueError("maximum_context must equal the largest training bucket")
        if not 0.0 <= self.decoder_mask_rate < 1.0:
            raise ValueError("decoder_mask_rate must lie in [0, 1)")
        if not 0.0 <= self.bow_weight <= 1.0:
            raise ValueError("bow_weight must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Complete outcome-free pilot specification and output boundary."""

    pilot_id: str
    sample_documents: int
    sequence_buckets: tuple[int, ...]
    allocator_fraction: float
    training_ceiling_seconds: int
    budget_seconds: int
    target_slots_per_update: int
    probe_validation_steps: int
    probe_max_batch: int
    sample_seed: str
    training_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    evaluation_sample_size: int
    final_content_tokens: int
    work_root: Path
    experiment_config: Path
    vintages_config: Path
    tokenizer_path: Path
    plan_attachment_dir: Path
    preparation_arms: tuple[PilotArm, ...]
    pooling_policies: tuple[PoolingPolicy, ...]
    article_policies: tuple[ArticlePolicy, ...]
    trials: tuple[PilotTrial, ...]

    def __post_init__(self) -> None:
        """Validate bounded runtime, sampling, and matrix invariants."""
        if not self.pilot_id or "/" in self.pilot_id or ".." in self.pilot_id:
            raise ValueError("pilot_id must be one safe path component")
        if self.sample_documents < 1:
            raise ValueError("sample_documents must be positive")
        buckets = self.sequence_buckets
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError("sequence_buckets must be unique and increasing")
        if any(
            value < _MIN_CONTEXT or value > _MAX_CONTEXT or value & (value - 1)
            for value in buckets
        ):
            raise ValueError("sequence buckets must be powers of two in [256, 2048]")
        if not 0.0 < self.allocator_fraction <= 1.0:
            raise ValueError("allocator_fraction must lie in (0, 1]")
        if not 1 <= self.budget_seconds < self.training_ceiling_seconds <= 1_200:
            raise ValueError("budget seconds must be below a ceiling of at most 1200")
        if self.target_slots_per_update < max(buckets):
            raise ValueError("target slots per update must cover the longest bucket")
        if min(self.probe_validation_steps, self.probe_max_batch) < 1:
            raise ValueError("probe limits must be positive")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if self.evaluation_sample_size < _MIN_EVALUATION_ROWS:
            raise ValueError("evaluation_sample_size must be at least three")
        if self.final_content_tokens < 1:
            raise ValueError("final_content_tokens must be positive")
        if not self.preparation_arms:
            raise ValueError("preparation_arms cannot be empty")
        if set(self.pooling_policies) != set(POOLING_POLICIES):
            raise ValueError("all three pooling policies must be predeclared")
        if set(self.article_policies) != set(ARTICLE_POLICIES):
            raise ValueError("both article policies must be predeclared")
        names = tuple(trial.name for trial in self.trials)
        if len(names) != len(set(names)):
            raise ValueError("pilot trial names must be unique")
        allowed = set(buckets)
        if any(not set(trial.training_buckets) <= allowed for trial in self.trials):
            raise ValueError("trial training buckets must be declared globally")

    @property
    def run_root(self) -> Path:
        """Return the ignored, pilot-specific output root."""
        return self.work_root / self.pilot_id

    @property
    def probe_manifest_path(self) -> Path:
        """Return the immutable resolved capacity manifest path."""
        return self.run_root / "probe.resolved.json"

    @property
    def probe_progress_path(self) -> Path:
        """Return the resumable controller-only probe progress path."""
        return self.run_root / "probe.progress.json"

    @property
    def prepared_root(self) -> Path:
        """Return the multi-length prepared-data root."""
        return self.run_root / "prepared"

    @property
    def evaluation_root(self) -> Path:
        """Return the intrinsic evaluation root."""
        return self.run_root / "evaluations"

    def trial(self, name: str) -> PilotTrial:
        """Resolve one named predeclared trial."""
        try:
            return next(trial for trial in self.trials if trial.name == name)
        except StopIteration as error:
            raise ValueError(f"unknown pilot trial: {name}") from error

    def trial_root(self, arm: PilotArm, trial: str) -> Path:
        """Return one arm/trial output root without touching frozen vintage runs."""
        return self.run_root / "trials" / arm / trial

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load a closed pilot TOML schema."""
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        allowed = {"pilot", "trials"}
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"unknown top-level pilot keys: {sorted(unknown)}")
        values = cast("dict[str, object]", raw.get("pilot", {}))
        raw_trials = raw.get("trials")
        if not isinstance(raw_trials, dict) or not raw_trials:
            raise ValueError("pilot config requires a non-empty [trials] table")
        trials = tuple(
            _parse_trial(str(name), cast("dict[str, object]", trial))
            for name, trial in raw_trials.items()
        )
        expected = {
            "id",
            "sample_documents",
            "sequence_buckets",
            "allocator_fraction",
            "training_ceiling_seconds",
            "budget_seconds",
            "target_slots_per_update",
            "probe_validation_steps",
            "probe_max_batch",
            "sample_seed",
            "training_seed",
            "bootstrap_seed",
            "bootstrap_resamples",
            "evaluation_sample_size",
            "final_content_tokens",
            "work_root",
            "experiment_config",
            "vintages_config",
            "tokenizer_path",
            "plan_attachment_dir",
            "preparation_arms",
            "pooling_policies",
            "article_policies",
        }
        unexpected = values.keys() - expected
        missing = expected - values.keys()
        if unexpected or missing:
            raise ValueError(
                f"pilot keys missing={sorted(missing)} unknown={sorted(unexpected)}"
            )
        return cls(
            pilot_id=str(values["id"]),
            sample_documents=int(cast("int", values["sample_documents"])),
            sequence_buckets=_int_tuple(values["sequence_buckets"]),
            allocator_fraction=float(cast("float", values["allocator_fraction"])),
            training_ceiling_seconds=int(
                cast("int", values["training_ceiling_seconds"])
            ),
            budget_seconds=int(cast("int", values["budget_seconds"])),
            target_slots_per_update=int(cast("int", values["target_slots_per_update"])),
            probe_validation_steps=int(cast("int", values["probe_validation_steps"])),
            probe_max_batch=int(cast("int", values["probe_max_batch"])),
            sample_seed=str(values["sample_seed"]),
            training_seed=int(cast("int", values["training_seed"])),
            bootstrap_seed=int(cast("int", values["bootstrap_seed"])),
            bootstrap_resamples=int(cast("int", values["bootstrap_resamples"])),
            evaluation_sample_size=int(cast("int", values["evaluation_sample_size"])),
            final_content_tokens=int(cast("int", values["final_content_tokens"])),
            work_root=Path(str(values["work_root"])),
            experiment_config=Path(str(values["experiment_config"])),
            vintages_config=Path(str(values["vintages_config"])),
            tokenizer_path=Path(str(values["tokenizer_path"])),
            plan_attachment_dir=Path(str(values["plan_attachment_dir"])),
            preparation_arms=cast(
                "tuple[PilotArm, ...]", _str_tuple(values["preparation_arms"])
            ),
            pooling_policies=cast(
                "tuple[PoolingPolicy, ...]", _str_tuple(values["pooling_policies"])
            ),
            article_policies=cast(
                "tuple[ArticlePolicy, ...]", _str_tuple(values["article_policies"])
            ),
            trials=trials,
        )


def _parse_trial(name: str, raw: dict[str, object]) -> PilotTrial:
    allowed = {
        "maximum_context",
        "training_buckets",
        "decoder_mask_rate",
        "bow_weight",
    }
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"unknown trial {name} keys: {sorted(unknown)}")
    maximum = raw.get("maximum_context")
    inherits = maximum == "winner"
    return PilotTrial(
        name=name,
        maximum_context=None if inherits else int(cast("int", maximum)),
        training_buckets=_int_tuple(raw.get("training_buckets", [])),
        decoder_mask_rate=float(cast("float", raw.get("decoder_mask_rate", 0.0))),
        bow_weight=float(cast("float", raw.get("bow_weight", 0.0))),
        inherits_context_winner=inherits,
    )


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("expected a TOML integer array")
    return tuple(int(item) for item in value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("expected a TOML string array")
    return tuple(str(item) for item in value)
