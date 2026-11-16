"""Bounded additive iid-history calibration without changing the saved N16 owner."""

from __future__ import annotations

import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from hotelling.artifacts import reject_symlinks
from hotelling.dynamics.artifacts import (
    file_hash,
    read_arrays,
    repository_root,
    value_hash,
    write_arrays,
)
from hotelling.dynamics.resources import ExecutionSettings, choose_resources
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_limits

from pipeline._kernels.paper6.inference_audit import (
    FIXTURE_FIRMS,
    FIXTURE_VARIANCE_FACTOR,
    ONE_SIDED_Z,
    bound_diagnostics,
    iid_replications,
)
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.precision import own_float64
from pipeline.stages.papers.paper6.artifacts import read, write

AUDIT_YEARS = 4

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class InferenceAuditConfig(BaseModel):
    """Fixed scientific settings for an additive iid-history sample-size audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    firms: int = Field(default=100, ge=4)
    replications: int = Field(default=2500, ge=2)
    pilot_replications: int = Field(default=20, ge=2)
    seed: int = Field(default=620260100, ge=0)

    @model_validator(mode="after")
    def bounded_pilot(self) -> InferenceAuditConfig:
        """Keep the timing pilot within the fixed production replication budget."""
        if self.pilot_replications > self.replications:
            msg = "pilot replications cannot exceed the fixed production count"
            raise ValueError(msg)
        return self


def _fixture_task(
    replications: int, firms: int, seed: int, threads: int
) -> tuple[dict[str, NDArray[np.float64]], float]:
    """Own worker precision and execution controls without changing stream keys."""
    own_float64()
    with threadpool_limits(limits=threads):
        started = perf_counter()
        arrays = iid_replications(replications=replications, firms=firms, seed=seed)
        return arrays, perf_counter() - started


def _same_fixture(
    first: Mapping[str, NDArray[np.float64]],
    second: Mapping[str, NDArray[np.float64]],
) -> bool:
    return first.keys() == second.keys() and all(
        first[key].shape == second[key].shape
        and np.allclose(first[key], second[key], rtol=1e-12, atol=1e-12)
        for key in first
    )


def _check_fixture(
    arrays: Mapping[str, NDArray[np.float64]], replications: int
) -> None:
    """Reject incomplete or scientifically incompatible completed checkpoints."""
    expected = {
        "targets": (2,),
        "estimates": (2, replications),
        "candidate_se": (2, replications),
        "candidate_lower_bound": (2, replications),
        "numerator": (2, replications),
        "denominator": (2, replications),
        "analytic_influence_sample_se": (replications,),
    }
    if set(arrays) != set(expected) or any(
        arrays[name].shape != shape for name, shape in expected.items()
    ):
        msg = "iid fixture checkpoint has an incomplete replication roster"
        raise ValueError(msg)
    if not np.array_equal(arrays["targets"], [0.0, 0.5]):
        msg = "iid fixture checkpoint targets differ from the frozen design"
        raise ValueError(msg)


def _audit_output(output: Path) -> Path:
    reject_symlinks(output.absolute())
    output = output.resolve()
    allowed = repository_root() / "data/papers/paper6/inference_audit"
    temporary = Path(tempfile.gettempdir()).resolve()
    if output == temporary or not (
        output.is_relative_to(allowed) or output.is_relative_to(temporary)
    ):
        msg = "inference audit output requires its governed namespace or /tmp"
        raise ValueError(msg)
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "provenance.manifest.json").is_file()
    ):
        msg = "nonempty inference audit output has no resumable manifest"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _baseline_fixture(baseline: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Hash both immutable N16 comparison inputs and verify their saved contract."""
    status_path, arrays_path = (
        baseline / "inference-status.json",
        baseline / "fixture-replicates.npz",
    )
    reject_symlinks(baseline.absolute())
    status = read(status_path)
    hashes = {path.name: file_hash(path) for path in (status_path, arrays_path)}
    fixture = status["iid_fixture"]
    if fixture["firms"] != FIXTURE_FIRMS or fixture["years"] != AUDIT_YEARS:
        msg = "comparison input must contain the retained N16 four-year iid fixture"
        raise ValueError(msg)
    if fixture["replicates_sha256"] != hashes[arrays_path.name]:
        msg = "N16 fixture comparison checkpoint hash mismatch"
        raise ValueError(msg)
    arrays = read_arrays(arrays_path)
    _check_fixture(arrays, fixture["replications_per_target"])
    return fixture, hashes


def _audit_manifest(
    output: Path, design: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    path = output / "provenance.manifest.json"
    manifest: dict[str, Any]
    if path.exists():
        manifest = read(path)
        if manifest.get("identity") != identity or manifest.get(
            "identity_hash"
        ) != value_hash(identity):
            msg = "inference audit manifest has incompatible code, design or inputs"
            raise ValueError(msg)
        required = {"design.json"}
        if manifest["status"] in ("pilot_complete", "fixture_complete", "complete"):
            required.add("pilot.npz")
        if manifest["status"] in ("fixture_complete", "complete"):
            required.add("fixture-replicates.npz")
        if manifest["status"] == "complete":
            required.update(("results.json", "inference-status.json"))
        if not required.issubset(manifest["completed"]):
            msg = "inference audit manifest omits a required completed artifact"
            raise ValueError(msg)
        for name, digest in manifest["completed"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                msg = "inference audit checkpoint path is unsafe"
                raise ValueError(msg)
            if file_hash(output / relative) != digest:
                msg = f"inference audit checkpoint hash mismatch: {name}"
                raise ValueError(msg)
        return manifest
    shutil.copyfile(design, output / "design.json")
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "identity_hash": value_hash(identity),
        "status": "initialized",
        "completed": {"design.json": file_hash(output / "design.json")},
        "execution": {},
    }
    write(path, manifest)
    return manifest


def _audit_pilot(
    output: Path,
    manifest: dict[str, Any],
    config: InferenceAuditConfig,
    executor: ProcessPoolExecutor,
    threads: int,
) -> None:
    if "pilot.npz" in manifest["completed"]:
        _check_fixture(read_arrays(output / "pilot.npz"), config.pilot_replications)
        return
    pilot_seed = config.seed + 1
    serial, serial_seconds = _fixture_task(
        config.pilot_replications, config.firms, pilot_seed, threads
    )
    started = perf_counter()
    parallel, parallel_seconds = executor.submit(
        _fixture_task,
        config.pilot_replications,
        config.firms,
        pilot_seed,
        threads,
    ).result()
    if not _same_fixture(serial, parallel):
        msg = "iid fixture serial and spawned results disagree at 1e-12"
        raise ArithmeticError(msg)
    _check_fixture(parallel, config.pilot_replications)
    write_arrays(output / "pilot.npz", parallel)
    manifest["completed"]["pilot.npz"] = file_hash(output / "pilot.npz")
    manifest["execution"]["pilot"] = {
        "replications_per_target": config.pilot_replications,
        "seed": pilot_seed,
        "serial_seconds": serial_seconds,
        "spawn_numerical_seconds": parallel_seconds,
        "spawn_elapsed_seconds": perf_counter() - started,
        "serial_spawn_agreement": True,
        "estimated_production_numerical_seconds": (
            parallel_seconds * config.replications / config.pilot_replications
        ),
    }
    manifest["status"] = "pilot_complete"
    write(output / "provenance.manifest.json", manifest)


def _sample_size_results(
    arrays: Mapping[str, NDArray[np.float64]],
    config: InferenceAuditConfig,
    baseline: dict[str, Any],
    identity: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    scenarios = []
    for index, target in enumerate(arrays["targets"]):
        scenarios.append(
            {
                **bound_diagnostics(
                    arrays["estimates"][index],
                    arrays["candidate_se"][index],
                    float(target),
                ),
                "sampling_target": "The iid-history fixture population pooled ratio.",
                "rate_interpretation": (
                    "Coverage and rejection of H0: beta <= 0 in this iid fixture only."
                ),
                "asymptotic_standard_error": float(
                    np.sqrt(FIXTURE_VARIANCE_FACTOR / config.firms)
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "method": (
            "Retained complete-triple first projection, ratio derivative and Cov(IF)/N."
        ),
        "candidate_bound": "beta_hat - z_0.95 * candidate_se",
        "one_sided_z": ONE_SIDED_Z,
        "null": "H0: beta <= 0; H1: beta > 0",
        "identity": identity,
        "execution": manifest["execution"],
        "iid_fixture": {
            "firms": config.firms,
            "years": 4,
            "replications_per_target": config.replications,
            "seed": config.seed,
            "stream": "SeedSequence([seed, replicate]); paired targets 0 and 0.5.",
            "law": baseline["law"],
            "population_first_influence": ".9*E*(X+.5)",
            "asymptotic_variance": "1.0125 / N",
            "scenarios": scenarios,
            "replicates": "fixture-replicates.npz",
            "replicates_sha256": manifest["completed"]["fixture-replicates.npz"],
        },
        "baseline_iid_fixture": baseline,
        "baseline_input_hashes": identity["baseline_input_hashes"],
        "comparison_scope": (
            "Sample-size diagnostic under the same iid law; N16 and N100 use "
            "distinct predeclared seeds. This does not validate corporate inference."
        ),
        "empirical_inference": {
            "eligible": False,
            "lower_bound": None,
            "p_value": None,
            "target": "Complete retained corporate archive projection beta_archive.",
            "reason": (
                "An iid-history sample-size calibration cannot establish the missing "
                "iid-firm sampling interpretation for the corporate archive. Article "
                "redraws condition on its firms and economic histories."
            ),
            "permitted_uncertainty": (
                "Conditional article-selection sensitivity quantiles."
            ),
        },
        "decision": (
            "Corporate inference remains ineligible independently of N100 results."
        ),
    }


def run_inference_audit(
    output: Path,
    config: InferenceAuditConfig,
    *,
    design: Path,
    baseline: Path,
    threads: int = 1,
    pilot_only: bool = False,
) -> dict[str, Any]:
    """Run the additive fixed-N audit with one bounded spawned fixture task."""
    record = read(design)
    expected = {
        "schema_version": 1,
        **config.model_dump(),
        "targets": [0.0, 0.5],
        "years": 4,
    }
    if not record.get("frozen_before_new_results") or any(
        record.get(key) != value for key, value in expected.items()
    ):
        msg = "inference audit design differs from its frozen scientific configuration"
        raise ValueError(msg)
    baseline_fixture, baseline_hashes = _baseline_fixture(baseline)
    identity = {
        "schema_version": 1,
        "code_identity": entrypoint_code_identity("paper6-inference-audit"),
        "design_sha256": file_hash(design),
        "config": config.model_dump(),
        "baseline_input_hashes": baseline_hashes,
    }
    output = _audit_output(output)
    manifest = _audit_manifest(output, design, identity)
    if manifest["status"] == "complete":
        return read(output / "results.json")
    choice = choose_resources(
        ExecutionSettings(workers=1, threads=threads),
        deadline=perf_counter() + 60,
        run_count=1,
    )
    manifest["execution"]["resources"] = choice.to_dict()
    with ProcessPoolExecutor(
        max_workers=choice.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        _audit_pilot(output, manifest, config, executor, choice.threads)
        if pilot_only:
            return {"status": manifest["status"], "execution": manifest["execution"]}
        if "fixture-replicates.npz" not in manifest["completed"]:
            arrays, seconds = executor.submit(
                _fixture_task,
                config.replications,
                config.firms,
                config.seed,
                choice.threads,
            ).result()
            _check_fixture(arrays, config.replications)
            write_arrays(output / "fixture-replicates.npz", arrays)
            manifest["completed"]["fixture-replicates.npz"] = file_hash(
                output / "fixture-replicates.npz"
            )
            manifest["execution"]["production_numerical_seconds"] = seconds
            manifest["status"] = "fixture_complete"
            write(output / "provenance.manifest.json", manifest)
    arrays = read_arrays(output / "fixture-replicates.npz")
    _check_fixture(arrays, config.replications)
    result = _sample_size_results(arrays, config, baseline_fixture, identity, manifest)
    for name in ("results.json", "inference-status.json"):
        write(output / name, result)
        manifest["completed"][name] = file_hash(output / name)
    manifest["status"] = "complete"
    write(output / "provenance.manifest.json", manifest)
    return read(output / "results.json")
