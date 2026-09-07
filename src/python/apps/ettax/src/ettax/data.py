# SPDX-License-Identifier: Apache-2.0
"""Streaming text preparation and memory-mapped Grain input."""

from __future__ import annotations

import bisect
import bz2
import gzip
import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, overload, override

import numpy as np
import orjson
import zstandard
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import IO, TextIO

SPECIAL_TOKENS = ("[PAD]", "[DOC]", "[EOS]", "[UNK]")
TOKENIZER_RECIPE = "byte-complete-bpe-v1"
BYTE_LEVEL_ALPHABET = tuple(pre_tokenizers.ByteLevel.alphabet())
_WRITE_BUFFER_ROWS = 1_024


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    """Description of one fixed-width token file."""

    rows: int
    sequence_length: int
    vocab_size: int
    pad_id: int
    doc_id: int
    eos_id: int
    tokenizer_sha256: str
    dtype: str = "uint16"
    content_tokens: int = 0
    selected_documents: int = 0
    selection_sha256: str = ""

    @classmethod
    def read(cls, token_path: str | Path) -> Self:
        """Read adjacent JSON metadata."""
        with _metadata_path(token_path).open(encoding="utf-8") as stream:
            return cls(**json.load(stream))


class MemmapCorpus(Sequence[np.ndarray]):
    """Random-access view over one or more raw uint16 matrices."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        """Open shards without reading their contents.

        Args:
            paths: Raw ``.u16`` files with adjacent metadata.

        """
        if not paths:
            raise ValueError("at least one corpus path is required")
        self.arrays: list[np.memmap] = []
        self.ends: list[int] = []
        self.sequence_length = 0
        self.vocab_size = 0
        self.pad_id = 0
        self.doc_id = 0
        self.eos_id = 0
        self.tokenizer_sha256 = ""
        total = 0
        for raw_path in paths:
            path = Path(raw_path)
            metadata = CorpusMetadata.read(path)
            signature = (
                metadata.sequence_length,
                metadata.vocab_size,
                metadata.pad_id,
                metadata.doc_id,
                metadata.eos_id,
                metadata.tokenizer_sha256,
            )
            current = (
                self.sequence_length,
                self.vocab_size,
                self.pad_id,
                self.doc_id,
                self.eos_id,
                self.tokenizer_sha256,
            )
            if self.sequence_length and signature != current:
                raise ValueError(
                    "all corpus shards must share tokenizer and sequence metadata"
                )
            (
                self.sequence_length,
                self.vocab_size,
                self.pad_id,
                self.doc_id,
                self.eos_id,
                self.tokenizer_sha256,
            ) = signature
            array = np.memmap(
                path,
                mode="r",
                dtype=np.uint16,
                shape=(metadata.rows, metadata.sequence_length),
            )
            self.arrays.append(array)
            total += metadata.rows
            self.ends.append(total)

    @override
    def __len__(self) -> int:
        """Return total rows across shards."""
        return self.ends[-1]

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[np.ndarray]: ...

    @override
    def __getitem__(self, index: int | slice) -> np.ndarray | Sequence[np.ndarray]:
        """Return one copied row suitable for asynchronous loading."""
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        start = 0 if shard == 0 else self.ends[shard - 1]
        return np.array(self.arrays[shard][index - start], dtype=np.uint16, copy=True)


def grain_batches(
    corpus: MemmapCorpus,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool = True,
    repeat: bool = True,
) -> Iterable[np.ndarray]:
    """Build a lazy Grain pipeline over memory-mapped rows."""
    import grain

    try:
        dataset = grain.MapDataset.source(corpus)
    except TypeError:
        dataset = grain.MapDataset.range(len(corpus)).map(corpus.__getitem__)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size, drop_remainder=True)


def train_tokenizer(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    vocab_size: int = 32_768,
    text_field: str = "text",
    max_documents: int | None = 2_000_000,
    source_documents: int | None = None,
    sample_seed: str = "ettax-tokenizer",
) -> None:
    """Train a byte-level BPE tokenizer from a streaming iterator.

    Args:
        inputs: JSONL or text files, optionally compressed.
        output: Tokenizer JSON path.
        vocab_size: Final vocabulary size including special tokens.
        text_field: JSON field containing article text.
        max_documents: Approximate uniform sample size for tokenizer fitting.
        source_documents: Known input-document count, required for uniform sampling.
        sample_seed: Stable hash-sampling seed.

    """
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[-1]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=list(BYTE_LEVEL_ALPHABET),
        show_progress=True,
    )
    documents: Iterable[str] = _sample_documents(
        iter_documents(inputs, text_field=text_field),
        target=max_documents,
        population=source_documents,
        seed=sample_seed,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer, length=max_documents)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def prepare_corpus(
    inputs: Sequence[str | Path],
    output: str | Path,
    tokenizer_path: str | Path,
    *,
    sequence_length: int = 1_024,
    text_field: str = "text",
    max_documents: int | None = None,
    source_documents: int | None = None,
    sample_seed: str = "ettax-corpus",
) -> CorpusMetadata:
    """Stream documents into a fixed-width raw uint16 matrix.

    Args:
        inputs: JSONL or text files, optionally compressed.
        output: Raw token matrix path.
        tokenizer_path: Trained tokenizer JSON.
        sequence_length: Stored row width.
        text_field: JSON field containing article text.
        max_documents: Approximate uniform article sample size.
        source_documents: Known number of input articles for hash sampling.
        sample_seed: Stable hash-sampling seed.

    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    doc_id = _required_id(tokenizer, "[DOC]")
    eos_id = _required_id(tokenizer, "[EOS]")
    pad_id = _required_id(tokenizer, "[PAD]")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("uint16 storage requires a vocabulary below 65,536 tokens")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    content_tokens = 0
    selected_documents = 0
    selection_hash = hashlib.sha256()
    width = sequence_length - 2
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with partial_path.open("wb") as stream:
        buffer: list[list[int]] = []
        documents = _sample_documents(
            iter_documents(inputs, text_field=text_field),
            target=max_documents,
            population=source_documents,
            seed=sample_seed,
        )
        for group in _batched(documents, 256):
            encodings = tokenizer.encode_batch(group)
            for document, encoding in zip(group, encodings, strict=True):
                selected_documents += 1
                selection_hash.update(document.encode())
                selection_hash.update(b"\n")
                token_ids = encoding.ids
                for offset in range(0, len(token_ids), width):
                    chunk = token_ids[offset : offset + width]
                    if not chunk:
                        continue
                    row = [doc_id, *chunk, eos_id]
                    content_tokens += len(row)
                    row.extend([pad_id] * (sequence_length - len(row)))
                    buffer.append(row)
                    rows += 1
                    if len(buffer) == _WRITE_BUFFER_ROWS:
                        _flush(buffer, stream)
        _flush(buffer, stream)
    partial_path.replace(output_path)
    metadata = CorpusMetadata(
        rows=rows,
        sequence_length=sequence_length,
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=pad_id,
        doc_id=doc_id,
        eos_id=eos_id,
        tokenizer_sha256=hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        content_tokens=content_tokens,
        selected_documents=selected_documents,
        selection_sha256=selection_hash.hexdigest(),
    )
    metadata_path = _metadata_path(output_path)
    partial_metadata = metadata_path.with_suffix(metadata_path.suffix + ".partial")
    with partial_metadata.open("w", encoding="utf-8") as stream:
        json.dump(asdict(metadata), stream, indent=2, sort_keys=True)
        stream.write("\n")
    partial_metadata.replace(metadata_path)
    return metadata


def iter_documents(
    inputs: Sequence[str | Path], *, text_field: str = "text"
) -> Iterator[str]:
    """Yield non-empty documents from supported files."""
    for raw_path in inputs:
        path = Path(raw_path)
        with _open_text(path) as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                if ".jsonl" in path.name or path.suffix == ".json":
                    value = orjson.loads(line)
                    text = value[text_field]
                    if not isinstance(text, str):
                        raise TypeError(f"{path}: field {text_field!r} is not text")
                else:
                    text = line
                if text.strip():
                    yield text


def _open_text(path: Path) -> TextIO:
    """Open plain or compressed UTF-8 text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    if path.suffix == ".zst":
        binary = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(binary)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open(encoding="utf-8")


def _flush(buffer: list[list[int]], stream: IO[bytes]) -> None:
    """Append buffered rows and clear their Python storage."""
    if buffer:
        np.asarray(buffer, dtype=np.uint16).tofile(stream)
        buffer.clear()


def _required_id(tokenizer: Tokenizer, token: str) -> int:
    """Read a required special-token ID."""
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"tokenizer is missing {token}")
    return value


def _metadata_path(token_path: str | Path) -> Path:
    """Return the adjacent metadata path."""
    path = Path(token_path)
    return path.with_suffix(path.suffix + ".json")


def _batched(values: Iterable[str], size: int) -> Iterator[list[str]]:
    """Yield small tokenizer batches without buffering the corpus."""
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sample_documents(
    values: Iterable[str],
    *,
    target: int | None,
    population: int | None,
    seed: str,
) -> Iterator[str]:
    """Yield a stable uniform hash sample without materializing documents."""
    if target is None:
        yield from values
        return
    if target < 1:
        raise ValueError("max_documents must be positive")
    if population is None:
        for index, value in enumerate(values):
            if index >= target:
                break
            yield value
        return
    if population < 1:
        raise ValueError("source_documents must be positive")
    threshold = min(1.0, target / population)
    limit = int(threshold * 2**64)
    for index, value in enumerate(values, start=1):
        digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
        if int.from_bytes(digest, "big") < limit:
            yield value
