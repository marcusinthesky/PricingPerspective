# SPDX-License-Identifier: Apache-2.0
"""Deterministic multi-length pilot data and content-token accounting."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from math import floor
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

import numpy as np
from tokenizers import Tokenizer

from ettax.data import iter_documents
from ettax.vintage import VintageConfig

if TYPE_CHECKING:
    from ettax.pilot.config import PilotConfig

_TOKENIZE_BATCH_SIZE = 256
_WRITE_BUFFER_ROWS = 512


@dataclass(frozen=True, slots=True)
class PilotBucketMetadata:
    """Auditable counters for one arm and sequence-length bucket."""

    arm: str
    bucket: int
    rows: int
    selected_documents: int
    represented_documents: int
    content_tokens: int
    doc_token_slots: int
    eos_token_slots: int
    padding_token_slots: int
    nominal_token_slots: int
    selection_sha256: str
    tokenizer_sha256: str
    data_sha256: str
    dtype: str = "uint16"

    @property
    def padding_efficiency(self) -> float:
        """Return loss-bearing content divided by all fixed-width slots."""
        if self.nominal_token_slots == 0:
            return 0.0
        return self.content_tokens / self.nominal_token_slots


def prepare_pilot_data(
    config: PilotConfig, *, arms: Sequence[str] | None = None
) -> dict[str, object]:
    """Prepare exact deterministic arm samples in a single pass per corpus."""
    if not config.tokenizer_path.is_file():
        raise FileNotFoundError(
            f"missing byte-complete tokenizer: {config.tokenizer_path}"
        )
    vintage = VintageConfig.from_toml(config.vintages_config)
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    manifest_path = config.prepared_root / "manifest.json"
    existing: dict[str, object] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("arms"), dict):
            existing = loaded
    summaries = dict(cast("dict[str, object]", existing.get("arms", {})))
    selected_arms = tuple(arms or config.preparation_arms)
    for arm_name in selected_arms:
        arm = vintage.arm(arm_name)
        if not arm.source.is_file():
            raise FileNotFoundError(f"missing governed corpus: {arm.source}")
        summaries[arm_name] = prepare_bucketed_arm(
            arm_name=arm_name,
            source=arm.source,
            tokenizer=tokenizer,
            output_root=config.prepared_root / arm_name,
            sample_documents=config.sample_documents,
            sample_seed=f"{config.sample_seed}:{arm_name}",
            buckets=config.sequence_buckets,
            tokenizer_path=config.tokenizer_path,
        )
    manifest: dict[str, object] = {
        "arms": summaries,
        "pilot_id": config.pilot_id,
        "sample_documents": config.sample_documents,
        "sample_seed": config.sample_seed,
        "sequence_buckets": list(config.sequence_buckets),
        "tokenizer_path": str(config.tokenizer_path),
        "tokenizer_sha256": _sha256(config.tokenizer_path),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def prepare_bucketed_arm(
    *,
    arm_name: str,
    source: Path,
    tokenizer: Tokenizer,
    output_root: Path,
    sample_documents: int,
    sample_seed: str,
    buckets: tuple[int, ...],
    tokenizer_path: Path,
) -> dict[str, object]:
    """Select exactly ``sample_documents`` and tokenize each selected text once."""
    if sample_documents < 1:
        raise ValueError("sample_documents must be positive")
    if not buckets or tuple(sorted(set(buckets))) != buckets:
        raise ValueError("buckets must be non-empty, unique, and increasing")
    output_root.mkdir(parents=True, exist_ok=True)
    database = output_root / ".selection.partial.sqlite3"
    selected = _reservoir_documents(
        iter_documents((source,)),
        target=sample_documents,
        seed=sample_seed,
        database=database,
    )
    selection_hash = hashlib.sha256()
    rows = dict.fromkeys(buckets, 0)
    represented: dict[int, set[int]] = {}
    content_tokens = dict.fromkeys(buckets, 0)
    buffers: dict[int, list[list[int]]] = {}
    for bucket in buckets:
        represented[bucket] = set()
        buffers[bucket] = []
    partials = {bucket: output_root / f"{bucket}.u16.partial" for bucket in buckets}
    final_paths = {bucket: output_root / f"{bucket}.u16" for bucket in buckets}
    doc_id = _required_id(tokenizer, "[DOC]")
    eos_id = _required_id(tokenizer, "[EOS]")
    pad_id = _required_id(tokenizer, "[PAD]")
    tokenizer_sha = _sha256(tokenizer_path)
    maximum_content = buckets[-1] - 2
    try:
        with ExitStack() as stack:
            streams = {
                bucket: stack.enter_context(partials[bucket].open("wb"))
                for bucket in buckets
            }
            for group in _batched(selected, _TOKENIZE_BATCH_SIZE):
                source_indices = [item[0] for item in group]
                texts = [item[1] for item in group]
                encodings = tokenizer.encode_batch(texts)
                for source_index, encoding in zip(
                    source_indices, encodings, strict=True
                ):
                    selection_hash.update(f"{source_index}\n".encode())
                    token_ids = encoding.ids
                    document_buckets: set[int] = set()
                    chunks = (
                        token_ids[offset : offset + maximum_content]
                        for offset in range(0, max(len(token_ids), 1), maximum_content)
                    )
                    for chunk in chunks:
                        if not chunk:
                            continue
                        bucket = next(
                            value for value in buckets if len(chunk) <= value - 2
                        )
                        row = [doc_id, *chunk, eos_id]
                        row.extend([pad_id] * (bucket - len(row)))
                        buffers[bucket].append(row)
                        rows[bucket] += 1
                        content_tokens[bucket] += len(chunk)
                        document_buckets.add(bucket)
                        if len(buffers[bucket]) >= _WRITE_BUFFER_ROWS:
                            _flush(buffers[bucket], streams[bucket])
                    for bucket in document_buckets:
                        represented[bucket].add(source_index)
            for bucket in buckets:
                _flush(buffers[bucket], streams[bucket])
        for bucket in buckets:
            partials[bucket].replace(final_paths[bucket])
    finally:
        database.unlink(missing_ok=True)

    selection_sha = selection_hash.hexdigest()
    metadata: dict[str, object] = {}
    for bucket in buckets:
        nominal = rows[bucket] * bucket
        specials = 2 * rows[bucket]
        values = PilotBucketMetadata(
            arm=arm_name,
            bucket=bucket,
            rows=rows[bucket],
            selected_documents=sample_documents,
            represented_documents=len(represented[bucket]),
            content_tokens=content_tokens[bucket],
            doc_token_slots=rows[bucket],
            eos_token_slots=rows[bucket],
            padding_token_slots=nominal - specials - content_tokens[bucket],
            nominal_token_slots=nominal,
            selection_sha256=selection_sha,
            tokenizer_sha256=tokenizer_sha,
            data_sha256=_sha256(final_paths[bucket]),
        )
        payload = {**asdict(values), "padding_efficiency": values.padding_efficiency}
        corpus_payload = {
            "content_tokens": values.content_tokens,
            "doc_id": doc_id,
            "dtype": values.dtype,
            "eos_id": eos_id,
            "pad_id": pad_id,
            "rows": values.rows,
            "selected_documents": values.selected_documents,
            "selection_sha256": values.selection_sha256,
            "sequence_length": bucket,
            "tokenizer_sha256": values.tokenizer_sha256,
            "vocab_size": tokenizer.get_vocab_size(),
        }
        _write_json_atomic(final_paths[bucket].with_suffix(".u16.json"), corpus_payload)
        _write_json_atomic(final_paths[bucket].with_suffix(".u16.pilot.json"), payload)
        metadata[str(bucket)] = payload
    summary: dict[str, object] = {
        "arm": arm_name,
        "buckets": metadata,
        "sample_documents": sample_documents,
        "selection_sha256": selection_sha,
        "source": str(source),
    }
    _write_json_atomic(output_root / "manifest.json", summary)
    return summary


def content_token_count(
    tokens: np.ndarray,
    *,
    pad_id: int,
    doc_id: int,
    eos_id: int,
) -> int:
    """Count loss-bearing clean content, excluding boundary and padding slots."""
    values = np.asarray(tokens)
    active = (values != pad_id) & (values != doc_id) & (values != eos_id)
    return int(np.count_nonzero(active))


def resolve_common_content_budget(
    observed_content_rates: Sequence[float],
    *,
    seconds: int,
    content_tokens_per_effective_update: int,
) -> int:
    """Resolve a common budget from the slowest observed real-batch throughput."""
    if not observed_content_rates or any(
        rate <= 0.0 for rate in observed_content_rates
    ):
        raise ValueError("observed content rates must be positive")
    if min(seconds, content_tokens_per_effective_update) < 1:
        raise ValueError("budget seconds and effective-update content must be positive")
    raw = min(observed_content_rates) * seconds
    updates = floor(raw / content_tokens_per_effective_update)
    if updates < 1:
        raise ValueError("observed throughput cannot fund one effective update")
    return updates * content_tokens_per_effective_update


def _reservoir_documents(
    documents: Iterable[str],
    *,
    target: int,
    seed: str,
    database: Path,
) -> Iterator[tuple[int, str]]:
    """Spool an exact deterministic reservoir without retaining article text in RAM."""
    connection = sqlite3.connect(database)
    try:
        statement = (
            "CREATE TABLE sample "
            "(slot INTEGER PRIMARY KEY, source_index INTEGER, text TEXT)"
        )
        connection.execute(statement)
        count = 0
        for count, text in enumerate(documents, start=1):
            source_index = count - 1
            if count <= target:
                connection.execute(
                    "INSERT INTO sample VALUES (?, ?, ?)",
                    (source_index, source_index, text),
                )
                continue
            candidate = _stable_integer(seed, source_index) % count
            if candidate < target:
                connection.execute(
                    "UPDATE sample SET source_index = ?, text = ? WHERE slot = ?",
                    (source_index, text, candidate),
                )
        if count < target:
            message = (
                f"source contains {count} documents, below requested exact sample "
                f"{target}"
            )
            raise ValueError(message)
        connection.commit()
        cursor = connection.execute(
            "SELECT source_index, text FROM sample ORDER BY source_index"
        )
        for source_index, text in cursor:
            yield int(source_index), str(text)
    finally:
        connection.close()


def _stable_integer(seed: str, index: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _required_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"tokenizer is missing {token}")
    return value


def _batched(
    values: Iterable[tuple[int, str]], size: int
) -> Iterator[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _flush(buffer: list[list[int]], stream: IO[bytes]) -> None:
    if buffer:
        np.asarray(buffer, dtype=np.uint16).tofile(stream)
        buffer.clear()


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
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)
