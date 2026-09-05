# SPDX-License-Identifier: Apache-2.0
"""Eligibility, lexicographic selection, and outcome-free recipe freezing."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ettax.pilot.evaluation import paired_bootstrap_interval
from ettax.pilot.frozen import source_tree_sha256

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ettax.pilot.config import PilotConfig


@dataclass(frozen=True, slots=True)
class RecipeScore:
    """One complete training-and-inference recipe flattened for selection."""

    arm: str
    trial: str
    policy: str
    maximum_context: int
    decoder_mask_rate: float
    bow_weight: float
    mrr: float
    qwen_spearman: float
    gpu_seconds_per_content_token: float
    stable_rank: float
    effective_rank: float
    mean_off_diagonal_cosine: float
    duplicate_rows: int
    finite: bool
    reciprocal_ranks: tuple[float, ...]
    eligible: bool = False
    ineligibility_reasons: tuple[str, ...] = ()

    @property
    def recipe_id(self) -> str:
        """Return the training/inference identity shared across vintages."""
        return f"{self.trial}:{self.policy}"


def summarize_and_freeze(config: PilotConfig) -> dict[str, object]:
    """Rank V0, confirm the top two on V1 when present, and freeze the winner."""
    scores = load_scores(config)
    v0 = [score for score in scores if score.arm == "V0"]
    if not v0:
        raise FileNotFoundError("no V0 pilot evaluations are available")
    eligible_v0 = apply_eligibility(v0)
    ranked_v0 = rank_v0(
        [score for score in eligible_v0 if score.eligible],
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    if len(ranked_v0) < 2:
        raise ValueError("fewer than two eligible complete V0 recipes")
    top_ids = tuple(score.recipe_id for score in ranked_v0[:2])
    v1_all = [score for score in scores if score.arm == "V1"]
    v1 = [score for score in v1_all if score.recipe_id in top_ids]
    confirmed = len({score.recipe_id for score in v1}) == 2
    winner: RecipeScore | None = None
    final_ranking: list[dict[str, object]] = []
    if confirmed:
        # A V1 evaluation contains all six inference policies. Keep those
        # non-final rows available as the applicable policy-matched TSDAE
        # baselines even though only the two predeclared recipe IDs advance.
        eligible_v1 = apply_eligibility(v1_all, reference_baselines=v0)
        by_id_v0 = {score.recipe_id: score for score in ranked_v0[:2]}
        by_id_v1 = _eligible_recipe_map(eligible_v1, top_ids)
        if set(by_id_v1) != set(top_ids):
            raise ValueError("a V1-confirmed recipe failed an eligibility gate")
        final_rows = [
            _confirmation_row(recipe_id, by_id_v0, by_id_v1) for recipe_id in top_ids
        ]
        final_ranking = sorted(
            final_rows,
            key=lambda row: (
                -_floating(row, "mean_mrr"),
                -_floating(row, "mean_qwen_spearman"),
                _floating(row, "mean_gpu_seconds_per_content_token"),
                by_id_v0[str(row["recipe_id"])].maximum_context,
                _complexity(by_id_v0[str(row["recipe_id"])]),
            ),
        )
        winner = by_id_v0[str(final_ranking[0]["recipe_id"])]

    summary: dict[str, object] = {
        "confirmed_on_v1": confirmed,
        "eligibility": [asdict(score) for score in eligible_v0],
        "final_ranking": final_ranking,
        "pilot_id": config.pilot_id,
        "top_two_v0": [asdict(score) for score in ranked_v0[:2]],
        "winner": asdict(winner) if winner is not None else None,
    }
    summary_path = config.run_root / "selection-summary.json"
    _write_json_atomic(summary_path, summary)
    decision_path = config.run_root / "decision.md"
    _write_text_atomic(decision_path, _decision_table(summary))
    if winner is not None:
        freeze_recipe(config, winner, summary_path=summary_path)
    copy_compact_plan_attachments(config, summary, decision_path)
    return summary


def load_scores(config: PilotConfig) -> list[RecipeScore]:
    """Load every evaluation JSON without accessing prices or returns."""
    results: list[RecipeScore] = []
    for path in sorted(config.evaluation_root.glob("*/*.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        arm = str(evaluation["arm"])
        trial_name = str(evaluation["trial"])
        trial = config.trial(trial_name)
        throughput = _floating(evaluation, "content_throughput")
        for policy, raw_metrics in cast(
            "Mapping[str, Mapping[str, object]]", evaluation["policies"]
        ).items():
            geometry = cast("Mapping[str, object]", raw_metrics["document_geometry"])
            retrieval = cast(
                "Mapping[str, object]", raw_metrics["title_body_retrieval"]
            )
            qwen = cast("Mapping[str, object]", raw_metrics["qwen_geometry"])
            scalars = (
                _floating(retrieval, "mrr"),
                _floating(qwen, "distance_spearman"),
                _floating(geometry, "stable_rank"),
                _floating(geometry, "effective_rank"),
                _floating(geometry, "mean_off_diagonal_cosine"),
                throughput,
            )
            results.append(
                RecipeScore(
                    arm=arm,
                    trial=trial_name,
                    policy=policy,
                    maximum_context=_integer(evaluation, "maximum_context"),
                    decoder_mask_rate=trial.decoder_mask_rate,
                    bow_weight=trial.bow_weight,
                    mrr=scalars[0],
                    qwen_spearman=scalars[1],
                    gpu_seconds_per_content_token=1.0 / throughput,
                    stable_rank=scalars[2],
                    effective_rank=scalars[3],
                    mean_off_diagonal_cosine=scalars[4],
                    duplicate_rows=_integer(geometry, "duplicate_rows"),
                    finite=bool(np.all(np.isfinite(scalars))),
                    reciprocal_ranks=tuple(
                        float(value)
                        for value in cast(
                            "Sequence[float]", raw_metrics["paired_reciprocal_ranks"]
                        )
                    ),
                )
            )
    return results


def apply_eligibility(
    scores: Sequence[RecipeScore],
    *,
    reference_baselines: Sequence[RecipeScore] = (),
) -> list[RecipeScore]:
    """Apply gates against same-arm TSDAE or frozen reference thresholds."""
    baselines = {
        (score.arm, score.maximum_context, score.policy): score
        for score in scores
        if score.trial == f"context-{score.maximum_context}"
    }
    frozen_baselines = {
        (score.maximum_context, score.policy): score
        for score in reference_baselines
        if score.trial == f"context-{score.maximum_context}"
    }
    output: list[RecipeScore] = []
    for score in scores:
        baseline = baselines.get((score.arm, score.maximum_context, score.policy))
        if baseline is None:
            baseline = frozen_baselines.get((score.maximum_context, score.policy))
        reasons: list[str] = []
        if not score.finite:
            reasons.append("non-finite metric")
        if score.duplicate_rows:
            reasons.append("duplicate vectors")
        if baseline is None:
            reasons.append("missing applicable TSDAE baseline")
        else:
            if score.stable_rank < 0.9 * baseline.stable_rank:
                reasons.append("stable rank below 90% of baseline")
            if score.effective_rank < 0.9 * baseline.effective_rank:
                reasons.append("effective rank below 90% of baseline")
            if (
                score.mean_off_diagonal_cosine
                > baseline.mean_off_diagonal_cosine + 0.002
            ):
                reasons.append("mean off-diagonal cosine exceeds baseline by 0.002")
        output.append(
            RecipeScore(
                **{
                    **asdict(score),
                    "eligible": not reasons,
                    "ineligibility_reasons": tuple(reasons),
                }
            )
        )
    return output


def rank_v0(
    scores: Sequence[RecipeScore], *, resamples: int, seed: int
) -> list[RecipeScore]:
    """Rank eligible V0 recipes under the predeclared lexicographic rule."""
    if not scores:
        return []

    def compare(left: RecipeScore, right: RecipeScore) -> int:
        lower, upper = paired_bootstrap_interval(
            left.reciprocal_ranks,
            right.reciprocal_ranks,
            resamples=resamples,
            seed=seed,
        )
        if not lower <= 0.0 <= upper:
            return -1 if left.mrr > right.mrr else 1
        if left.qwen_spearman != right.qwen_spearman:
            return -1 if left.qwen_spearman > right.qwen_spearman else 1
        if left.gpu_seconds_per_content_token != right.gpu_seconds_per_content_token:
            return (
                -1
                if left.gpu_seconds_per_content_token
                < right.gpu_seconds_per_content_token
                else 1
            )
        left_simple = (left.maximum_context, _complexity(left), left.recipe_id)
        right_simple = (right.maximum_context, _complexity(right), right.recipe_id)
        return (
            -1
            if left_simple < right_simple
            else (1 if left_simple > right_simple else 0)
        )

    return sorted(scores, key=cmp_to_key(compare))


def freeze_recipe(
    config: PilotConfig, winner: RecipeScore, *, summary_path: Path
) -> dict[str, object]:
    """Freeze hashes and source revision before any outcome analysis."""
    prepared_manifest = config.prepared_root / "manifest.json"
    prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    prepared_arms = cast("Mapping[str, object]", prepared.get("arms", {}))
    missing_arms = {"V0", "V1", "V3"} - prepared_arms.keys()
    if missing_arms:
        raise RuntimeError(
            "prepare all frozen-retrain arms before selection; missing "
            + ", ".join(sorted(missing_arms))
        )
    objective = (
        "mask70-bow10"
        if winner.bow_weight > 0.0
        else ("mask70" if winner.decoder_mask_rate > 0.0 else "tsdae")
    )
    run_id = f"v4-{winner.maximum_context}-{objective}"
    freeze: dict[str, object] = {
        "article_policy": winner.policy.split("-", 1)[1],
        "bow_weight": winner.bow_weight,
        "decoder_mask_rate": winner.decoder_mask_rate,
        "final_arms": ["V0", "V1", "V3"],
        "final_content_tokens_per_arm": config.final_content_tokens,
        "maximum_context": winner.maximum_context,
        "pilot_id": config.pilot_id,
        "pooling_policy": winner.policy.split("-", 1)[0],
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_sha256": _sha256(prepared_manifest),
        "probe_manifest": str(config.probe_manifest_path),
        "probe_manifest_sha256": _sha256(config.probe_manifest_path),
        "run_id": run_id,
        "selection_report": str(summary_path),
        "selection_report_sha256": _sha256(summary_path),
        "source_revision": _source_revision(),
        "source_tree_sha256": source_tree_sha256(),
        "tokenizer": str(config.tokenizer_path),
        "tokenizer_sha256": _sha256(config.tokenizer_path),
        "training_buckets": [
            value
            for value in config.sequence_buckets
            if value <= winner.maximum_context
        ],
        "training_seed": config.training_seed,
        "trial": winner.trial,
    }
    _write_json_atomic(config.run_root / "frozen-recipe.json", freeze)
    _write_text_atomic(config.run_root / "frozen-recipe.toml", _freeze_toml(freeze))
    return freeze


def copy_compact_plan_attachments(
    config: PilotConfig, summary: Mapping[str, object], decision_path: Path
) -> None:
    """Copy only compact governed outputs into the owning task attachment."""
    destination = config.plan_attachment_dir
    destination.mkdir(parents=True, exist_ok=True)
    probe = json.loads(config.probe_manifest_path.read_text(encoding="utf-8"))
    compact_probe = {
        key: probe.get(key)
        for key in ("allocator", "device", "environment", "pilot_id", "resolutions")
    }
    _write_json_atomic(destination / "probe-manifest.json", compact_probe)
    _write_json_atomic(
        destination / "selection-summary.json", _compact_selection_summary(summary)
    )
    shutil.copyfile(decision_path, destination / "decision.md")


def _compact_selection_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Remove paired-resample vectors from the governed decision attachment."""

    def compact_score(raw: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in raw.items() if key != "reciprocal_ranks"}

    return {
        "confirmed_on_v1": bool(summary["confirmed_on_v1"]),
        "eligibility": [
            compact_score(raw)
            for raw in cast("Sequence[Mapping[str, object]]", summary["eligibility"])
        ],
        "final_ranking": summary["final_ranking"],
        "pilot_id": summary["pilot_id"],
        "top_two_v0": [
            compact_score(raw)
            for raw in cast("Sequence[Mapping[str, object]]", summary["top_two_v0"])
        ],
        "winner": (
            compact_score(cast("Mapping[str, object]", summary["winner"]))
            if isinstance(summary.get("winner"), dict)
            else None
        ),
    }


def _complexity(score: RecipeScore) -> tuple[float, float]:
    return score.decoder_mask_rate, score.bow_weight


def _eligible_recipe_map(
    scores: Sequence[RecipeScore], recipe_ids: Sequence[str]
) -> dict[str, RecipeScore]:
    """Return only eligible rows whose complete recipe was predeclared."""
    selected = set(recipe_ids)
    return {
        score.recipe_id: score
        for score in scores
        if score.eligible and score.recipe_id in selected
    }


def _confirmation_row(
    recipe_id: str,
    v0: Mapping[str, RecipeScore],
    v1: Mapping[str, RecipeScore],
) -> dict[str, object]:
    return {
        "mean_gpu_seconds_per_content_token": float(
            np.mean(
                (
                    v0[recipe_id].gpu_seconds_per_content_token,
                    v1[recipe_id].gpu_seconds_per_content_token,
                )
            )
        ),
        "mean_mrr": float(np.mean((v0[recipe_id].mrr, v1[recipe_id].mrr))),
        "mean_qwen_spearman": float(
            np.mean((v0[recipe_id].qwen_spearman, v1[recipe_id].qwen_spearman))
        ),
        "recipe_id": recipe_id,
    }


def _decision_table(summary: Mapping[str, object]) -> str:
    lines = [
        "# EttaX pilot decision",
        "",
        "| Rank | Recipe | V0 MRR | Qwen Spearman | Eligible |",
        "| ---: | --- | ---: | ---: | :---: |",
    ]
    for index, raw in enumerate(
        cast("Sequence[Mapping[str, object]]", summary["top_two_v0"]), start=1
    ):
        lines.append(
            f"| {index} | `{raw['trial']}:{raw['policy']}` | "
            f"{_floating(raw, 'mrr'):.6f} | "
            f"{_floating(raw, 'qwen_spearman'):.6f} | yes |"
        )
    lines.extend(
        (
            "",
            f"V1 confirmation complete: **{str(summary['confirmed_on_v1']).lower()}**.",
            "",
        )
    )
    winner = summary.get("winner")
    if isinstance(winner, dict):
        lines.append(f"Frozen winner: `{winner['trial']}:{winner['policy']}`.")
        lines.append("")
    return "\n".join(lines)


def _freeze_toml(values: Mapping[str, object]) -> str:
    lines = ["[recipe]"]
    for key in (
        "pilot_id",
        "run_id",
        "trial",
        "maximum_context",
        "decoder_mask_rate",
        "bow_weight",
        "pooling_policy",
        "article_policy",
        "training_seed",
        "final_content_tokens_per_arm",
        "source_revision",
        "source_tree_sha256",
        "tokenizer_sha256",
        "prepared_manifest_sha256",
        "probe_manifest_sha256",
        "selection_report_sha256",
    ):
        value = values[key]
        rendered = json.dumps(value) if isinstance(value, str) else str(value).lower()
        lines.append(f"{key} = {rendered}")
    lines.append(
        "training_buckets = ["
        + ", ".join(
            str(item) for item in cast("Sequence[int]", values["training_buckets"])
        )
        + "]"
    )
    lines.append('final_arms = ["V0", "V1", "V3"]')
    return "\n".join(lines) + "\n"


def _source_revision() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _integer(values: Mapping[str, object], key: str) -> int:
    return int(cast("int", values[key]))


def _floating(values: Mapping[str, object], key: str) -> float:
    return float(cast("float", values[key]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)
