# pipeline

DVC data-pipeline stages for the pricing-perspective project.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run pipeline --help
```

Paper 2 also has a discoverable phase view which reuses the flat leaf
commands used by DVC:

```bash
uv run pipeline p2 --help
uv run pipeline p2 estimate --help
uv run pipeline p2 infer --help
uv run pipeline p2 exhibit figures --help
```

Semantic Python dependency fingerprints are exposed through the same composition
root while remaining implemented by the reusable `semflow` package:

```bash
uv run pipeline semflow check
uv run pipeline semflow patch
uv run pipeline semflow patch --apply
```

`patch` is preview-only without `--apply`. See
[`packages/semflow`](../../packages/semflow/README.md) for the JSON contract and
activation safety gate.

The flat names remain compatibility aliases. The grouped surface is argument
plumbing only; it does not add a second computation graph or merge DVC cache
boundaries.

The executable stage inventory and dependency graph are `dvc.yaml` and
`dvc.lock`; `dvc dag` renders them without a parallel semantic catalog:

```bash
dvc dag --collapse-foreach-matrix
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design rationale.

## Runtime settings

Environment and optional current-directory `.env` values are validated by
`pipeline.io.settings`; process environment values win. Prefer the repository's
secret-injection wrapper for paid embedding runs:

```bash
dotenvx run -- uv run pipeline generate-embeddings --help
```

| Variable | Type/default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | optional masked secret | Required only when an embedding request misses the local cache |
| `PP_JAX_CACHE_DIR` | path, `.jax_cache` | Persistent JAX compilation-cache location |
| `PP_BLAS_LIMIT` | positive integer, `2`; `0` disables | Native numerical thread cap |

`PP_JAX_CACHE_DIR` is the only cache knob this app validates. The rest of the
compilation-cache policy — `PP_JAX_CACHE_MAX_GIB` (LRU ceiling) and
`PP_JAX_CACHE_STRICT` (fail rather than silently recompile) — is read directly by
the shared config site in `dotell`, which `pipeline.jax_cache` delegates
to; see that package's README.

Pipeline modules do not read or mutate `os.environ` directly. Long-running stage
progress uses `tqdm` only on interactive stderr, while `dotell` records
structured runtime and resource measurements in every CLI invocation.

## Frozen upstream stages

`subsample` and `generate_embeddings` are marked `frozen: true` in `dvc.yaml`
(root of the repo). They are deterministic but heavy (`subsample` re-samples
the full NASDAQ corpus per ticker) and `generate_embeddings` makes paid
OpenRouter API calls per ticker/model. Freezing means `dvc repro` skips them
unconditionally, regardless of upstream changes — an overnight `dvc repro`
never resamples the corpus or re-hits OpenRouter.

To deliberately refresh either stage (a human action, not automated):

```bash
dvc unfreeze <stage>       # e.g. subsample@AAPL or generate_embeddings@AAPL-qwen3-embedding-4b
dvc repro <stage>
dvc freeze <stage>         # re-freeze once satisfied with the output
```

Even with the freeze lifted, an unfrozen `generate_embeddings` recompute makes
zero OpenRouter calls when the cache is warm: `data/.emb_cache/` is a
content-addressed cache keyed on `sha256(model_id + text)`, so re-running the
stage against unchanged inputs is a pure cache hit.

## Typed statistical stages

The shared typed-analysis registry in `params.yaml` defines the provider,
representation, distance, geometry, solver, feasible-set, and target-policy
identity used by the typed stages. Run the DVC-equivalent commands directly:

```bash
uv run pipeline typed-distance \
  --provider-id qwen3-embedding-4b \
  --representation-id qwen3-embedding-4b-unit \
  --distance-id energy_v \
  --output-dir data/shared/typed_distances/qwen3-embedding-4b/qwen3-embedding-4b-unit/energy_v \
  --params-file params.yaml

uv run pipeline typed-barycentre \
  --provider-id qwen3-embedding-4b \
  --representation-id qwen3-embedding-4b-unit \
  --arm-id energy_simplex \
  --output-dir data/shared/barycentres/qwen3-embedding-4b/qwen3-embedding-4b-unit/energy_simplex \
  --params-file params.yaml
```

The distance command writes `distances.parquet`, `summary.json`, and
`provenance.manifest.json`. The barycentre command writes
`barycentre.parquet`, `summary.json`, and `provenance.manifest.json`.
Target-projection weights and source-measure barycentres are distinct artifact
variants; consumers must use the matching reader.

The Paper 5 Wasserstein roster adds two leave-one-out target-projection cells:
`wasserstein_w1_loo` and `wasserstein_w2_loo`. For each target cloud, the solver
computes balanced pairwise transport maps to the other clouds, aligns those
candidate clouds to the target, and solves the candidate simplex conditional on
those fixed maps. This is an exact restricted target-anchored W1/W2 solve, not
an unrestricted multi-marginal Wasserstein barycentre; its diagnostics record
the per-target convergence certificate. Paper 5 consumes the resulting rows as
`w_w1` and `w_w2` spatial-weight variants.

The source-measure family is expanded by the separate
`typed_measure_barycentre_specs` DVC roster. Its three explicit cells are
`wasserstein_w1_measure` (balanced W1 with two prescribed sources),
`wasserstein_w2_measure` (balanced W2 with two prescribed sources), and
`wasserstein_w2_free_support` (three-source free-support W2, reported as a
local optimum). These cells read the embedding clouds directly and emit the
source-measure artifact variant; they are not target projections and do not
use the leave-one-out policy.

Raw typed geometry is materialized once by the DVC
`typed_geometry_components` family. Its energy, linear-MMD, and RBF-MMD
components are consumed by the final distance and barycentre stages. Paper 2's
three robustness rows then use the same two artifact interfaces: a squared
typed distance matrix and target-projection diagnostics. The published
`energy_mmd_simplex` row aliases the canonical `energy_v` matrix and
`energy_simplex` projection instead of materializing an equivalent second MMD
graph.

Paper 2's target-level conformal pilot is available through either command
surface:

```bash
uv run pipeline target-conformal
uv run pipeline p2 estimate target-conformal
```

It refits the PCA basis and energy-simplex rule on each declared source pool,
then writes `scores.parquet`, `split_results.parquet`, and `summary.json`. Its
claim is marginal coverage for one new cross-sectionally exchangeable target;
the artifact explicitly excludes universal, simultaneous, and temporal
coverage interpretations.
