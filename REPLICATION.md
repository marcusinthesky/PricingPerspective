# Replication source snapshot

This repository is a history-free source snapshot of the private research
monorepo behind *Distributional Information Geometry for Financial Dependence*
(University of Cape Town). It preserves the working path structure that
`dvc.yaml` refers to, so the declared analysis stages resolve here exactly as
they do in the source tree.

## Papers

| Project | Short title | Terminal stage |
| --- | --- | --- |
| `src/latex/projects/01_continuous_bounds` | Wasserstein covariance envelopes | `p1_numbers` |
| `src/latex/projects/03_distance_implied_mpt` | Portfolio risk bounds | `p3_numbers` |
| `src/latex/projects/05_spatial_pricing` | Wasserstein-barycentric interaction fields | `p5_numbers` |

## What is included

- **Python** (`src/python`) — the DVC runtime closure: `pipeline`, `simulation`,
  `ettax`, `jcor` and `semflow`, plus `dotell`, the shared telemetry library the
  first three import. `manimize`, the animated companion to the manuscripts, is
  retained in full but is not part of the DVC graph. `insitu` and
  `insitu-repository` appear as packaging metadata only.
- **Lean 4** (`src/lean`) — the `PricingPerspective`, `WassersteinGeometry`,
  `EnergyStatistics` and `PaperReconstructions` packages, with blueprint sources.
- **LaTeX** (`src/latex`) — the three paper projects above, the shared templates,
  and the vendored Tectonic bundle, so manuscript compiles stay offline.
- **Pipeline definition** — `dvc.yaml`, `dvc.lock`, `params.yaml`, and the
  `.semflow` semantic fingerprints.
- **References** (`.context/reference`) — the CSL frontmatter records used to
  verify the generated bibliographies.
- **A configured DVC remote** (`.dvc/config`) — see *Getting the data*.

`src/python/uv.lock` is locked to exactly the members present here.

## What is not included

- Development history, research notes, correspondence, and agent tooling.
- Test suites, benchmarks, and implementation code outside the runtime closure.
- The news corpus and the vendor price panels — see *What the remote does not
  carry*.
- The thesis aggregate under `src/latex/projects/`. The three paper chains
  above are the supported terminals.

## Checking out

Figures, PDFs, the Tectonic bundle, and the recorded metric series are stored in
Git LFS. Install it first, or the working tree will contain pointer files
instead of content:

```bash
git lfs install
git clone https://github.com/marcusinthesky/PricingPerspective.git
```

## Getting the data

The DVC objects live in a public Hugging Face storage bucket, configured as the
default remote in `.dvc/config`:

<https://huggingface.co/buckets/marcusinthesky/PricingPerspective>

Reads are anonymous. You need **no account, no token and no credentials**:

```bash
dvc pull
```

`dvc pull` with no targets will report errors for the artifacts the remote
deliberately does not carry, which is expected and harmless — everything
publishable still lands. To pull without those errors, name the published trees:

```bash
dvc pull \
  data/shared/embeddings data/shared/corpus/summary.yaml \
  data/shared/co_mentions data/shared/backtest_feasibility \
  data/shared/returns data/shared/oos/returns \
  data/shared/covariance_matrix.parquet data/shared/oos/covariance_matrix.parquet \
  data/shared/typed_distances data/shared/typed_distances_windowed \
  data/shared/typed_geometry_components data/shared/barycentres \
  data/shared/ablations data/shared/dimensionality_analysis \
  data/shared/mantel_tests data/shared/oos/mantel_tests \
  data/shared/validation data/shared/w2_exposure_frontier \
  data/papers data/mc data/publication data/universe.csv
```

### What the remote does not carry

Three classes of artifact are withheld for licensing reasons. Their `dvc.lock`
hashes are published, so the recorded provenance of every stage remains
auditable even where the inputs cannot be redistributed.

| Withheld | Reason |
| --- | --- |
| `data/raw/nasdaq.jsonlines` | Non-redistributable |
| `data/shared/corpus/articles.parquet` | Normalised full article text |
| `data/shared/subsampled_headlines/` | Sampled article bodies |
| `data/shared/market_data/`, `data/shared/oos/market_data/`, `data/archive/market_data/` | Vendor terms restrict redistribution of the price panels |

Consequently these stages cannot be re-run from this snapshot and are governed
only by their recorded hashes: `normalize_corpus`, `subsample`, `co_mentions`,
`backtest_feasibility`, `market_data`, `compute_returns`, and
`p1_dyadic_confound`, which reads `market_data` directly. Everything downstream
of `data/shared/returns` and `data/shared/embeddings` re-runs normally.

Refetching the price panels is **not** an equivalent substitute: vendors restate
adjusted prices, so re-downloaded values will not reproduce the `dvc.lock`
hashes and the frozen `market_data` stages will never verify against them.

`p1_dyadic_confound` is frozen for the same reason, but unlike the others its
outputs are carried in full rather than by hash alone. Every Paper 1 result
derives from them, so without them `p1_numbers` and `p1_tables` could not be
reached from this snapshot at all. They hold regression coefficients, residuals,
Wasserstein distances, leverage and bootstrap draws — no price levels, the same
disclosure class as the published return panels.

### Diagnostics regenerated rather than carried

Seventeen further Paper 1 and Paper 5 diagnostic outputs are declared
`cache: false` in `dvc.yaml`. DVC records their hashes but stores no objects for
them, so no remote can carry them and `dvc pull` will not produce them. Each is
regenerated by running its own stage, and every input those stages need is
published:

`p1_diagnostics`, `p1_disco`, `p1_distributional_comparator`, `p1_dyadic_oos`,
`p1_neighbour_outcome_split`, `p1_representation_sensitivity`,
`p1_sample_descriptives`, `p5_sample_descriptives`, `lean_bindings_check` and
`refs_bib_freshness`.

Running a paper terminal — `dvc repro p1_numbers`, say — produces them along the
way, so this affects only a reader who expects `dvc pull` alone to materialise
every path in `dvc.lock`.

### The published embeddings carry no article text

`data/shared/embeddings/**` holds the document embeddings for four off-the-shelf
encoders (`qwen3-embedding-8b`, `qwen3-embedding-4b`, `bge-m3`,
`bge-large-en-v1.5`) and three EttaX corpus vintages (`ettax-v0/v1/v3`).

Each shard carries the embedding vector plus article metadata — identifiers,
dates, the source `url`, byline and topic — but **not** `title`, `body` or
`publisher`. Nothing in the pipeline reads those columns: the consumer contract
is `{embedding, created_date, url_hash}` (`pipeline/io/typed_providers.py`). A
reader who holds their own licence for the corpus can rejoin the text on
`url_hash` or `url` and re-derive the shards.

## Snapshot checks

The source boundary can be verified without the data store:

```bash
uv lock --check --offline --project src/python
dvc dag --dot
dvc repro --dry --allow-missing p1_numbers p3_numbers p5_numbers
```

The pinned environment runs the same checks:

```bash
nix develop ./infra/nix#replication --command <command>
```

## Reproducing the results

With the data pulled, drop `--dry` and `--allow-missing`:

```bash
dvc repro p1_numbers p3_numbers p5_numbers
```

`dvc status` will not report a clean tree here, by construction: the withheld
artifacts above are reported as `not in cache`, and stages whose dependency is a
source directory the projection prunes (for example `semflow_refresh`) are
reported as changed. Neither indicates a problem with the published objects; to
check those, `dvc pull` the published trees and confirm it completes without a
checksum error.
