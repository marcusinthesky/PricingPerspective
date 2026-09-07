# SPDX-License-Identifier: Apache-2.0
"""Vintage-locked orchestration over the repository's existing Wikipedia corpora."""

from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

from ettax.config import ExperimentConfig
from ettax.data import (
    BYTE_LEVEL_ALPHABET,
    TOKENIZER_RECIPE,
    CorpusMetadata,
    prepare_corpus,
    train_tokenizer,
)
from ettax.pooling import ARTICLE_POLICIES, POOLING_POLICIES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ettax.pooling import ArticlePolicy, PoolingPolicy

type CheckpointLayout = Literal["vintage", "frozen"]

_CHECKPOINT_LAYOUTS: tuple[CheckpointLayout, ...] = ("vintage", "frozen")


@dataclass(frozen=True, slots=True)
class VintageArm:
    """One corpus-vintage arm and its scientific role."""

    name: str
    identifier: str
    source: Path
    role: str


@dataclass(frozen=True, slots=True)
class VintageConfig:
    """Repository-specific inputs surrounding the reusable encoder package."""

    tokenizer_arm: str
    tokenizer_documents: int
    prepared_documents: int
    sample_seed: str
    work_root: Path
    run_id: str
    budget_manifest: Path
    experiment_config: Path
    arms: tuple[VintageArm, ...]
    checkpoint_layout: CheckpointLayout = "vintage"
    pooling_policy: PoolingPolicy = "doc"
    article_policy: ArticlePolicy = "first"

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load the governed vintage inventory."""
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        raw_arms = raw.get("arms")
        if not isinstance(raw_arms, dict) or not raw_arms:
            message = "vintage config requires a non-empty [arms] table"
            raise ValueError(message)
        arms = tuple(
            VintageArm(
                name=str(name),
                identifier=str(values["identifier"]),
                source=Path(str(values["source"])),
                role=str(values["role"]),
            )
            for name, values in raw_arms.items()
        )
        config = cls(
            tokenizer_arm=str(raw["tokenizer_arm"]),
            tokenizer_documents=int(raw["tokenizer_documents"]),
            prepared_documents=int(raw["prepared_documents"]),
            sample_seed=str(raw["sample_seed"]),
            work_root=Path(str(raw["work_root"])),
            run_id=str(raw["run_id"]),
            budget_manifest=Path(str(raw["budget_manifest"])),
            experiment_config=Path(str(raw["experiment_config"])),
            arms=arms,
            checkpoint_layout=cast(
                "CheckpointLayout", str(raw.get("checkpoint_layout", "vintage"))
            ),
            pooling_policy=cast("PoolingPolicy", str(raw.get("pooling_policy", "doc"))),
            article_policy=cast(
                "ArticlePolicy", str(raw.get("article_policy", "first"))
            ),
        )
        if config.checkpoint_layout not in _CHECKPOINT_LAYOUTS:
            message = f"checkpoint_layout must be one of {_CHECKPOINT_LAYOUTS}"
            raise ValueError(message)
        if config.pooling_policy not in POOLING_POLICIES:
            message = f"pooling_policy must be one of {POOLING_POLICIES}"
            raise ValueError(message)
        if config.article_policy not in ARTICLE_POLICIES:
            message = f"article_policy must be one of {ARTICLE_POLICIES}"
            raise ValueError(message)
        if config.tokenizer_arm not in {arm.name for arm in arms}:
            message = "tokenizer_arm is absent from [arms]"
            raise ValueError(message)
        if min(config.tokenizer_documents, config.prepared_documents) < 1:
            message = "document sample sizes must be positive"
            raise ValueError(message)
        if Path(config.run_id).name != config.run_id or config.run_id in {
            "",
            ".",
            "..",
        }:
            message = "run_id must be one non-empty path segment"
            raise ValueError(message)
        return config

    def arm(self, name: str) -> VintageArm:
        """Resolve an arm by its stable short name."""
        for arm in self.arms:
            if arm.name == name:
                return arm
        message = f"unknown arm {name!r}; expected {[arm.name for arm in self.arms]}"
        raise ValueError(message)

    @property
    def tokenizer_path(self) -> Path:
        """Return the active recipe's shared V0-only tokenizer path."""
        return self.run_root / "tokenizer.json"

    @property
    def tokenizer_manifest_path(self) -> Path:
        """Return the active recipe's tokenizer provenance path."""
        return self.run_root / "tokenizer.manifest.json"

    def prepared_path(self, arm: VintageArm) -> Path:
        """Return one arm's tokenizer- and recipe-specific memory map."""
        return self.run_root / "prepared" / f"{arm.name}.u16"

    @property
    def run_root(self) -> Path:
        """Return the isolated output root for the active frozen recipe."""
        return self.work_root / "runs" / self.run_id

    def checkpoint_dir(self, arm: VintageArm) -> Path:
        """Return one arm's resumable checkpoint root under the declared layout.

        The ``vintage`` layout is what ``run_vintages`` writes. The ``frozen``
        layout is what the pilot's frozen queue writes, since it reuses
        ``run_trial`` with a per-arm ``output_root``.
        """
        if self.checkpoint_layout == "frozen":
            return self.run_root / arm.name / "checkpoints"
        return self.run_root / "checkpoints" / arm.name

    def metrics_path(self, arm: VintageArm) -> Path:
        """Return one arm's terminal metrics path."""
        return self.run_root / "metrics" / f"{arm.name}.json"

    def live_dir(self, arm: VintageArm) -> Path:
        """Return one arm's step-indexed DVCLive directory."""
        return self.run_root / "dvclive" / arm.name

    def manifest_path(self, arm: VintageArm) -> Path:
        """Return one arm's provenance manifest path."""
        return self.run_root / "manifests" / f"{arm.name}.json"


def prepare_vintages(config: VintageConfig, arm_names: tuple[str, ...] = ()) -> None:
    """Build missing tokenizer/shards from existing corpora without acquisition work."""
    budget = _load_budget(config)
    tokenizer_arm = config.arm(config.tokenizer_arm)
    tokenizer_manifest = config.tokenizer_manifest_path
    if not config.tokenizer_path.is_file():
        _require_source(tokenizer_arm)
        source_documents = _source_documents(budget, tokenizer_arm)
        train_tokenizer(
            [tokenizer_arm.source],
            config.tokenizer_path,
            max_documents=config.tokenizer_documents,
            source_documents=source_documents,
            sample_seed=f"{config.sample_seed}:tokenizer",
        )
        _write_json_atomic(
            tokenizer_manifest,
            {
                "arm": tokenizer_arm.name,
                "initial_alphabet_size": len(BYTE_LEVEL_ALPHABET),
                "identifier": tokenizer_arm.identifier,
                "recipe": TOKENIZER_RECIPE,
                "run_id": config.run_id,
                "sample_seed": f"{config.sample_seed}:tokenizer",
                "source": str(tokenizer_arm.source),
                "source_documents": source_documents,
                "target_documents": config.tokenizer_documents,
                "tokenizer_sha256": _sha256(config.tokenizer_path),
            },
        )
    elif not tokenizer_manifest.is_file():
        message = (
            f"{config.tokenizer_path} exists without {tokenizer_manifest}; "
            "refusing unaudited reuse"
        )
        raise RuntimeError(message)
    else:
        _validate_tokenizer_manifest(config, tokenizer_arm)

    selected = (
        config.arms if not arm_names else tuple(config.arm(name) for name in arm_names)
    )
    base = ExperimentConfig.from_toml(config.experiment_config)
    tokenizer_sha = _sha256(config.tokenizer_path)
    for arm in selected:
        _require_source(arm)
        output = config.prepared_path(arm)
        if output.is_file() and output.with_suffix(output.suffix + ".json").is_file():
            metadata = CorpusMetadata.read(output)
            if metadata.tokenizer_sha256 != tokenizer_sha:
                message = f"{output} was built with a different tokenizer"
                raise RuntimeError(message)
            if metadata.sequence_length != base.data.sequence_length:
                message = (
                    f"{output} has stale sequence length {metadata.sequence_length}"
                )
                raise RuntimeError(message)
            continue
        prepare_corpus(
            [arm.source],
            output,
            config.tokenizer_path,
            sequence_length=base.data.sequence_length,
            max_documents=config.prepared_documents,
            source_documents=_source_documents(budget, arm),
            sample_seed=f"{config.sample_seed}:corpus",
        )


def train_vintage(
    config: VintageConfig,
    arm_name: str,
    *,
    require_gpu: bool = True,
) -> dict[str, float | int | str | bool]:
    """Train or resume exactly one independently initialized vintage arm."""
    import flax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from ettax.train import config_manifest, run  # noqa: PLC0415

    arm = config.arm(arm_name)
    prepared = config.prepared_path(arm)
    if not prepared.is_file():
        message = f"{prepared} is missing; run `ettax vintage prepare {arm.name}`"
        raise FileNotFoundError(message)
    if require_gpu and jax.default_backend() != "gpu":
        message = (
            f"refusing long training on backend={jax.default_backend()!r}; "
            "set JAX_PLATFORMS=cuda"
        )
        raise RuntimeError(message)
    experiment = _arm_experiment(config, arm)
    latest = _latest_step(config.checkpoint_dir(arm))
    if latest >= experiment.steps and config.metrics_path(arm).is_file():
        return json.loads(config.metrics_path(arm).read_text(encoding="utf-8"))
    metrics = run(experiment, resume=True)
    budget = _load_budget(config)
    metadata = CorpusMetadata.read(prepared)
    manifest = {
        "arm": asdict(arm),
        "budget_source": _budget_arm(budget, arm),
        "checkpoint": str(config.checkpoint_dir(arm)),
        "environment": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "flax": flax.__version__,
            "jax": jax.__version__,
            "optax": optax.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "experiment": config_manifest(experiment),
        "metrics": metrics,
        "prepared": asdict(metadata),
        "run_id": config.run_id,
        "tokenizer_sha256": _sha256(config.tokenizer_path),
    }
    _write_json_atomic(config.manifest_path(arm), manifest)
    return metrics


def run_vintages(config: VintageConfig, *, require_gpu: bool = True) -> None:
    """Prepare missing artifacts, then train every arm sequentially on one device."""
    prepare_vintages(config)
    for arm in config.arms:
        train_vintage(config, arm.name, require_gpu=require_gpu)


def vintage_status(config: VintageConfig) -> list[dict[str, object]]:
    """Return compact state for every arm without importing JAX."""
    tokenizer = config.tokenizer_path.is_file()
    rows: list[dict[str, object]] = []
    for arm in config.arms:
        prepared = config.prepared_path(arm)
        metadata_path = prepared.with_suffix(prepared.suffix + ".json")
        metadata = (
            CorpusMetadata.read(prepared)
            if prepared.is_file() and metadata_path.is_file()
            else None
        )
        rows.append(
            {
                "arm": arm.name,
                "checkpoint_step": _latest_step(config.checkpoint_dir(arm)),
                "identifier": arm.identifier,
                "manifest": config.manifest_path(arm).is_file(),
                "metrics": config.metrics_path(arm).is_file(),
                "training_curve": (config.live_dir(arm) / "metrics.json").is_file(),
                "prepared_rows": metadata.rows if metadata else 0,
                "run_id": config.run_id,
                "source": arm.source.is_file(),
                "tokenizer": tokenizer,
            }
        )
    return rows


def _arm_experiment(config: VintageConfig, arm: VintageArm) -> ExperimentConfig:
    base = ExperimentConfig.from_toml(config.experiment_config)
    return replace(
        base,
        data=replace(base.data, paths=(str(config.prepared_path(arm)),)),
        train=replace(
            base.train,
            checkpoint_dir=str(config.checkpoint_dir(arm)),
            live_dir=str(config.live_dir(arm)),
            metrics_path=str(config.metrics_path(arm)),
        ),
    )


def _load_budget(config: VintageConfig) -> dict[str, object]:
    if not config.budget_manifest.is_file():
        message = (
            f"{config.budget_manifest} is missing; existing vintage corpora "
            "are not auditable"
        )
        raise FileNotFoundError(message)
    value = json.loads(config.budget_manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("arms"), dict):
        message = f"invalid budget manifest: {config.budget_manifest}"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _source_documents(budget: dict[str, object], arm: VintageArm) -> int:
    values = _budget_arm(budget, arm)
    articles = values["articles_written"]
    if not isinstance(articles, int) or isinstance(articles, bool):
        message = f"budget arm {arm.name} has a non-integer article count"
        raise TypeError(message)
    return articles


def _budget_arm(
    budget: dict[str, object],
    arm: VintageArm,
) -> Mapping[str, object]:
    arms = budget["arms"]
    if not isinstance(arms, dict) or arm.name not in arms:
        message = f"budget manifest has no arm {arm.name}"
        raise ValueError(message)
    values = cast("Mapping[str, object]", arms[arm.name])
    if not isinstance(values, dict):
        message = f"budget arm {arm.name} is not an object"
        raise TypeError(message)
    return values


def _require_source(arm: VintageArm) -> None:
    if not arm.source.is_file() or arm.source.stat().st_size == 0:
        message = (
            f"existing corpus {arm.source} is absent; acquisition is "
            "deliberately outside ettax"
        )
        raise FileNotFoundError(message)


def _validate_tokenizer_manifest(
    config: VintageConfig,
    tokenizer_arm: VintageArm,
) -> None:
    manifest = json.loads(config.tokenizer_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "arm": tokenizer_arm.name,
        "identifier": tokenizer_arm.identifier,
        "initial_alphabet_size": len(BYTE_LEVEL_ALPHABET),
        "recipe": TOKENIZER_RECIPE,
        "run_id": config.run_id,
        "sample_seed": f"{config.sample_seed}:tokenizer",
        "target_documents": config.tokenizer_documents,
        "tokenizer_sha256": _sha256(config.tokenizer_path),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        message = (
            f"{config.tokenizer_manifest_path} is incompatible with the active "
            f"tokenizer recipe: {mismatches}"
        )
        raise RuntimeError(message)


def _latest_step(checkpoint_root: Path) -> int:
    latest = checkpoint_root / "latest.json"
    if not latest.is_file():
        return 0
    value = json.loads(latest.read_text(encoding="utf-8"))
    return int(value["step"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)
