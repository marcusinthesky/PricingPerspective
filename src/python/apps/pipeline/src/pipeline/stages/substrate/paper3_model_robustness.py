"""Cross-encoder robustness driver for Paper 3's W2 empirical design."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

import yaml

from pipeline.stages.papers.paper3.empirical import (
    Paper3EmpiricalConfig,
    Paper3EmpiricalPaths,
    run_paper3_empirical,
)
from pipeline.stages.papers.paper3.pit_distance import compute_pit_matrices_for_provider

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class Paper3RobustnessKwargs(TypedDict):
    """Keyword-compatible configuration for one or all encoder arms."""

    baseline_model: str
    returns_dir: Path
    output_dir: Path
    params_file: Path
    distance_artifact_dir: Path
    provider_id: str
    representation_id: str
    distance_id: str
    canonical_summary: NotRequired[Path | None]
    n_boot: NotRequired[int]
    n_mc: NotRequired[int]
    n_clusters: NotRequired[int]
    force: NotRequired[bool]


@dataclass(frozen=True)
class _Config:
    baseline_model: str
    returns_dir: Path
    output_dir: Path
    params_file: Path
    distance_artifact_dir: Path
    provider_id: str
    representation_id: str
    distance_id: str
    canonical_summary: Path | None
    n_boot: int
    n_mc: int
    n_clusters: int
    force: bool


def _config(options: Paper3RobustnessKwargs) -> _Config:
    return _Config(
        baseline_model=options["baseline_model"],
        returns_dir=options["returns_dir"],
        output_dir=options["output_dir"],
        params_file=options["params_file"],
        distance_artifact_dir=options["distance_artifact_dir"],
        provider_id=options["provider_id"],
        representation_id=options["representation_id"],
        distance_id=options["distance_id"],
        canonical_summary=options.get("canonical_summary"),
        n_boot=options.get("n_boot", 300),
        n_mc=options.get("n_mc", 2000),
        n_clusters=options.get("n_clusters", 15),
        force=options.get("force", True),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _paper3_stats(summary: dict[str, Any]) -> dict[str, Any]:
    tests = summary.get("envelope_tests", summary.get("identity_tests"))
    if not isinstance(tests, dict):
        message = "Paper 3 summary has no envelope_tests mapping"
        raise TypeError(message)
    backtest = summary["backtest"]

    def sharpe(constraint: str, method: str, window: int) -> float | None:
        try:
            return float(backtest[constraint][method][window]["sharpe_net_10bps"])
        except (KeyError, TypeError):
            return None

    values: dict[str, Any] = {
        "gmm_wald_pvalue": tests["gmm_wald_pvalue"],
        "wolak_pvalue": tests["wolak_pvalue"],
        "kappa_hat": tests["kappa_hat"],
        "sharpe_net_10bps": {},
        "sigma_dist_beats_sample_10bps": backtest.get("sigma_dist_beats_sample_10bps"),
    }
    for constraint in ("long_only", "unconstrained"):
        for window in (10, 20, 60, 250):
            values["sharpe_net_10bps"].setdefault(constraint, {})[window] = {
                "sigma_dist": sharpe(constraint, "sigma_dist", window),
                "sample": sharpe(constraint, "sample", window),
            }
    return values


def _run_model(config: _Config, model: str, model_dir: Path) -> dict[str, Any]:
    """Resolve the baseline summary or execute one non-baseline empirical arm."""
    if (
        model == config.baseline_model
        and config.canonical_summary
        and config.canonical_summary.exists()
    ):
        return _paper3_stats(_load_yaml(config.canonical_summary))

    empirical_dir = model_dir / "paper3_empirical"
    if model != config.baseline_model:
        compute_pit_matrices_for_provider(
            config.params_file,
            model_dir / "pit_w2" / "matrices.npz",
            model,
            f"{model}-unit",
        )
    if config.force and empirical_dir.exists():
        shutil.rmtree(empirical_dir)
    run_paper3_empirical(
        Paper3EmpiricalPaths(
            returns_dir=config.returns_dir,
            distance_artifact_dir=config.distance_artifact_dir,
            output_dir=empirical_dir,
        ),
        Paper3EmpiricalConfig(
            n_clusters=config.n_clusters,
            n_boot=config.n_boot,
            n_mc=config.n_mc,
            provider_id=config.provider_id,
            representation_id=config.representation_id,
            distance_id=config.distance_id,
        ),
    )
    return _paper3_stats(_load_yaml(empirical_dir / "summary.yaml"))


def run_paper3_model_robustness_model(
    model: str, **options: Unpack[Paper3RobustnessKwargs]
) -> dict[str, Any]:
    """Run Paper 3's W2 robustness design for one encoder model."""
    config = _config(options)
    model_dir = config.output_dir / model
    errors: dict[str, str] = {}
    stats: dict[str, Any] | None = None
    try:
        stats = _run_model(config, model, model_dir)
    except Exception as error:
        logger.exception("[%s] Paper 3 robustness arm failed", model)
        errors[model] = repr(error)

    result = {"paper3": stats, "errors": errors}
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "summary.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(result, file, sort_keys=False, default_flow_style=False)
    return result


def merge_paper3_model_robustness_summaries(
    models: list[str], baseline_model: str, output_dir: Path
) -> None:
    """Merge per-model summaries into Paper 3's consolidated robustness file."""
    result: dict[str, Any] = {"paper3": {}, "errors": {}}
    for model in models:
        path = output_dir / model / "summary.yaml"
        if not path.exists():
            result["errors"][model] = repr(FileNotFoundError(str(path)))
            continue
        summary = _load_yaml(path)
        if summary.get("paper3") is not None:
            result["paper3"][model] = summary["paper3"]
        result["errors"].update(summary.get("errors", {}))
    result["models"] = models
    result["baseline_model"] = baseline_model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(result, file, sort_keys=False, default_flow_style=False)


def run_paper3_model_robustness(
    models: list[str], **options: Unpack[Paper3RobustnessKwargs]
) -> None:
    """Run and merge Paper 3's cross-encoder robustness grid."""
    config = _config(options)
    for model in models:
        run_paper3_model_robustness_model(model=model, **options)
    merge_paper3_model_robustness_summaries(
        models, config.baseline_model, config.output_dir
    )
