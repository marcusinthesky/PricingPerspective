# EttaX

EttaX trains the matched V0/V1/V3 document encoders for the encoder-vintage
staircase. It is a workspace application because it owns a runnable GPU job,
experiment configuration, checkpoints, metrics, and repository-specific corpus
orchestration.

The default encoder is a 12-layer, 320-wide Transformer with exactly
**22,052,480 trainable parameters**. A separate one-layer decoder brings the
training graph to 23,016,960 parameters and is discarded for embedding. Only
the input embedding and vocabulary projection are tied. The active
`v3-byte-complete` recipe uses a fixed 900-million-token-slot budget per arm.

The frozen mixed-precision path keeps parameters and optimizer state in float32,
uses bfloat16 matrix operands with float32 accumulation where supported, and
caches the immutable NNX training graph traversal. Global encoder attention uses
cuDNN on GPU; local and causal-local attention retain XLA semantics.

## Commands

Run from the repository root:

```bash
just python::ettax::architecture
just python::ettax::status
just python::ettax::metrics
just python::ettax::plot --show-vega
just python::ettax::prepare
JAX_PLATFORMS=cuda just python::ettax::benchmark
JAX_PLATFORMS=cuda just python::ettax::run
JAX_PLATFORMS=cuda just python::ettax::semantic-smoke
JAX_PLATFORMS=cuda just python::ettax::embed V3 --output-dir data/shared/embeddings/ettax-v3
just python::ettax::pilot-prepare
just python::ettax::pilot-probe
just python::ettax::pilot-run context-256
just python::ettax::pilot-evaluate context-256
just python::ettax::pilot-summarize
```

The `embed` command materializes one vintage's normalized document vectors as
per-symbol Parquets using the same `url_hash`, `symbol`, `embedding`,
`text_length`, and `embedding_dim` contract consumed by the shared embedding
stages. It is intentionally separate from the published off-the-shelf roster
until semantic preflight results are adjudicated.

`prepare` only reads the existing files under
`data/.encoder_vintage/corpus/`. It does not download dumps, extract XML, or
rebuild the matched corpora. It fits a 32,768-entry byte-level BPE tokenizer on
V0, seeds all 256 byte symbols so arbitrary UTF-8 text cannot fall through to
`[UNK]`, and creates model-specific `uint16` memory maps. The tokenizer and
shards live under `runs/v3-byte-complete/`, so the completed V2 tokenizer and
shards cannot be reused by the corrected recipe.

`run` prepares missing model inputs and then trains V0, V1, and V3 sequentially
on one GPU. Every arm starts from the same seed and independent initialization,
uses the shared tokenizer and architecture, and resumes from an Orbax checkpoint
containing model, optimizer, step, and PRNG state. DVCLive persists raw and
exponentially smoothed loss, gradient norm, learning rate, and throughput at the
configured logging interval. Resume reconciles the curves to the restored
checkpoint before appending, so failed work is never represented as completed.
At each checkpoint it also evaluates the same deterministic 256-row sample from
that arm's training corpus. This is a low-noise training-objective probe, not a
held-out validation set.

`semantic-smoke` is a separate pre-adjudication gate over 2,048 deterministic,
ticker/year-balanced rows from the existing Nasdaq corpus. It embeds titles,
bodies, and their concatenated documents without reading prices or returns. The
single JSON result records title-to-body retrieval, collapse/anisotropy/effective
rank diagnostics, chord/angular Mantel tests, and full pairwise-distance/10-nearest-
neighbour agreement with the already-materialized Qwen3-Embedding-4B vectors.
Tokenization diagnostics separately record truncation, unknown tokens, duplicate
input text, and duplicate token rows for both the corrected and optional legacy
tokenizers, so lossy preprocessing cannot be mistaken for model collapse. JAX
inference exports FP32; the host-side ranking, Mantel, cosine, and singular-value
diagnostics use float64 so ties and rank conclusions do not depend on float32
accumulation order.

The corrected V3 per-arm summaries and plots under
`data/.encoder_vintage/ettax/runs/v3-byte-complete/dvclive/` are declared in
the root `dvc.yaml` and can be inspected with `dvc metrics show` and
`dvc plots show`. The rejected lossy-tokenizer run remains archival evidence
outside the active embedding DAG. Recipe-specific output roots prevent
incompatible tokenizers, prepared shards, or model graphs from being restored
across recipes. DVCLive does not create experiment commits; the repository's
ordinary Git/DVC workflow owns version comparison.

## Outcome-free pilot

`configs/pilots.toml` owns a separate capacity and architecture screen. Pilot
artifacts stay under the ignored
`data/.encoder_vintage/ettax/pilots/<pilot-id>/` tree and never enter root DVC.
The five commands are:

```text
ettax pilot probe       # fresh-process CUDA capacity search
ettax pilot prepare     # exact deterministic V0/V1 multi-length samples
ettax pilot run         # one named content-matched trial
ettax pilot evaluate    # six pooling/chunk inference policies
ettax pilot summarize   # eligibility, V1 confirmation, and recipe freeze
```

The probe controller imports no JAX. Every candidate inherits
`JAX_PLATFORMS=cuda` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` in a fresh
subprocess, distinguishes CUDA OOM from unrelated failure, searches the integer
gap after powers of two, and validates the maximum for ten optimizer steps. Its
resolved physical batches and gradient-accumulation factors are immutable inputs
to later commands. This follows the documented
[JAX allocator controls](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

`prepare` selects exactly 250,000 documents per configured arm in one corpus scan,
tokenizes each selected document once, and writes non-overlapping chunks into the
smallest fitting 256/512/1024/2048 bucket. Metadata reports content, boundary,
padding, and nominal slots separately. `run` benchmarks real prepared batches on
first use, fixes the common round budget to the slowest content-token throughput
times 1,080 seconds, and applies one optimizer update only after all microbatches
have been accumulated. Supplying `--content-budget` is an explicit test/debug
override; `--smoke` defaults to one effective update under a separate `smoke/`
root so acceptance checks cannot seed or resume a screened checkpoint.

Decoder-prefix masking replaces only eligible shifted clean tokens with the
byte-complete tokenizer's `[UNK]` ID. The optional tied-vocabulary document BoW
loss excludes every special token and is logged separately from reconstruction
and total loss. The one-layer asymmetric decoder and 70% decoder mask follow the
[RetroMAE design range](https://aclanthology.org/2022.emnlp-main.35/); the 0.1 BoW
weight follows its
[official Duplex MAE configuration](https://github.com/staoxiao/RetroMAE/blob/master/src/pretrain/arguments.py),
while direct `[DOC]` supervision is an EttaX adaptation.

`summarize` applies the predeclared collapse gates and paired bootstrap rule. It
freezes only after the two leading complete recipes have V1 evaluations. The
V3 prepared-data hashes must already be present (`ettax pilot prepare --arm V3`)
so all three final inputs are frozen before selection. The reserved command
`ettax pilot run frozen` verifies that freeze, writes the runtime forecast first,
and then runs V0/V1/V3 sequentially to at least 700 million loss-bearing content
tokens per arm under a new `v4-*` root.

## Layout

```text
configs/base.toml       frozen architecture, optimizer, and metric recipe
configs/pilots.toml     isolated capacity, screening, evaluation, and freeze recipe
configs/vintages.toml   corpus inventory and arm roles for `v3-byte-complete`
configs/vintages-v4.toml  same inventory for the frozen `v4-256-mask70` run
resources/              historical extraction source retained for provenance
src/ettax/pilot/        probe, bucket data, training, evaluation, selection, freeze
src/ettax/pooling.py    chunking and the six pooling/article policies
src/ettax/              model, preparation, training/tracking, orchestration, CLI
tests/                  unit, numerical, integration, benchmark, and GPU checks
```

One vintage config per frozen run, because a run's checkpoint layout and its
selected `(pooling, article)` policy pair are properties of that run rather than
of the app. Embedding a run means naming its config:

```bash
JAX_PLATFORMS=cuda just python::ettax::embed V0 \
  --config src/python/apps/ettax/configs/vintages-v4.toml \
  --output-dir data/shared/embeddings/ettax-v4-V0
```

Omitting `--config` targets `v3-byte-complete` under `doc`/`first`, which is
what the published `data/shared/embeddings/ettax-v*` artifacts were built with.

## Gates

```bash
just python::ettax::test
uv run --project src/python --package ettax ruff check src/python/apps/ettax
uv run --project src/python --package ettax deptry src/python/apps/ettax
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the placement, model, precision, and
data-boundary decisions.
