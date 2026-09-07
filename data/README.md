# Data Provenance

This document records the reproducibility root of the news-corpus pipeline and
the governed encoder identities used to turn that corpus into paper-specific
artifacts.

## Raw source

- **API endpoint:** `api.nasdaq.com/api/news/topic/articlebysymbol`
- **Raw file:** `data/raw/nasdaq.jsonlines`
- **md5:** `a30d7fe43928f7d99fee118a5cf66b77`
- **Size:** 1,815,644,085 bytes
- **Records:** 397,041 JSON lines (one article per line)
- **Collected:** 2022-12 to 2023-01

The raw file is **immutable**. It is never rewritten by any stage. Corpus-derived
artifacts additionally bind the encoder provider or checkpoint declared in
`params.yaml` and the resolved stage identity recorded in `dvc.lock`; market-data
artifacts retain their separate source provenance.

## Corpus normalization (`normalize_corpus`)

The raw JSON lines are deduplicated, date-parsed, and filtered to the 100-ticker
universe defined in `params.yaml` (`symbols:`). Dedupe statistics recorded in
`data/shared/corpus/summary.yaml`:

| Metric | Value |
| --- | --- |
| rows_in | 397,041 |
| rows_out | 214,480 |
| duplicates_dropped | 182,561 |
| symbols | 100 |
| date span | 2009-10 to 2023-01 |
| per_ticker_min | 449 |

## Declared firm universe

The 100-name news universe is a prespecified, Nasdaq-100-based sampling frame.
It was formed from the original Nasdaq scrape using the
`annual_raw_article_floor: 47` threshold in `params.yaml`: each retained firm
required at least that many articles in every calendar year from 2018 through
2022.
The screen was applied to the raw archive before the current identifier-level
deduplication; it is therefore a universe-selection rule, not a claim that every
normalized firm-year still contains 47 distinct records.

The current normalization validates that all 100 declared symbols are present;
after identifier-level deduplication the least-covered ticker retains 449
articles over the full recovered history. Annual raw-coverage qualification is
applied before that deduplication, so this aggregate minimum is not a substitute
for the firm-year admission rule.

The frame is coverage-conditioned and is not presented as an exhaustive sample
of every Nasdaq firm satisfying the threshold. The source was chosen because
Nasdaq exposes a public per-symbol news endpoint and the recovered records carry
full syndicated article bodies, rather than headline strings alone. The endpoint
is Nasdaq-hosted; the underlying articles come from multiple publishers.

## Encoder and representation provenance

Article embeddings are a shared substrate, but there is no repository-wide
Qwen3-Embedding-4B primary pin. `params.yaml` separately governs the external
model registry (`embedding.models`), raw provider artifacts
(`typed_analysis.providers`), deterministic representation transforms
(`typed_analysis.representations`), and paper-specific selectors. `dvc.yaml`
materializes those identities into path-disjoint artifacts, and `dvc.lock`
records the values resolved by each stage.

The shared canonical provider is:

| Field | Governed value |
| --- | --- |
| Provider ID | `qwen3-embedding-8b` |
| External model ID | `qwen/qwen3-embedding-8b` |
| Hugging Face reference | `Qwen/Qwen3-Embedding-8B` |
| Native width | 4096 |
| Generation access | OpenRouter's OpenAI-compatible embeddings endpoint |
| Frozen raw shards | `data/shared/embeddings/qwen3-embedding-8b/<TICKER>.parquet` |
| Canonical representation | `qwen3-embedding-8b-unit` |
| Representation transform | Row-wise L2 normalization at full width (4096) |

The API endpoint is the configured generation route; the frozen DVC artifacts
are the reproduction route. A full-width representation has `dimension: null`
in the typed registry because it retains the provider's native width, not because
its width is unknown. The canonical typed-distance summary records both the
native and realized dimensions as 4096.

### Shard columns

Every embedding shard — off-the-shelf and EttaX vintage alike — carries the five
fields the writer computes (`url_hash`, `symbol`, `embedding`, `text_length`,
`embedding_dim`) and ten passthrough fields from the article sample (`id`,
`primarysymbol`, `body_len`, `author`, `created_date`, `url`, `primarytopic`,
`related_symbols`, `date_accessed`, `_missing_date`).

It carries **no** `title`, `body` or `publisher`. The shards are redistributed
through the public replication bucket while the corpus they derive from is not,
so the writers exclude article text by construction: an allow-list, because
`subsample` selects `*` from the normalized corpus and a deny-list would pass a
new upstream column through silently. The policy lives in
`pipeline.stages.substrate.embeddings.SHARD_PASSTHROUGH_COLUMNS` and
`ettax.semantic.SHARD_PASSTHROUGH_FIELDS`, which must stay in step.

Dropping those columns cannot change a result: the consumer contract is
`{embedding, created_date, url_hash}` (`pipeline.io.typed_providers`). A holder
of a corpus licence can rejoin the text on `url_hash` or `url`.

### Paper-specific primary bindings

The first three manuscript projects currently share the canonical provider and
representation. Paper 6 provisionally selects Qwen3-Embedding-4B; every paper binds
its provider through a separate selector and derived artifact.
Changing one paper's selector must not be described as changing another paper's
primary representation unless its own binding changes.

| Paper | Governing selector and DVC stage | Binding output | Provider / representation | Effective width | Shared artifact access |
| --- | --- | --- | --- | --- | --- |
| Paper 1 | `paper1_lead_model`; `p1_dyadic_confound` | `data/papers/paper1/dyadic_confound/summary.yaml` | `qwen3-embedding-8b` / `qwen3-embedding-8b-unit` | 4096 | `data/shared/typed_distances/qwen3-embedding-8b/qwen3-embedding-8b-unit/wasserstein_w2` |
| Paper 3 | `scientific_selectors.typed_distance_*` and `paper3.primary_distance_id`; `p3_empirical` | `data/papers/paper3/empirical/provenance.manifest.json` | `qwen3-embedding-8b` / `qwen3-embedding-8b-unit` | 4096 | `data/shared/typed_distances/qwen3-embedding-8b/qwen3-embedding-8b-unit/wasserstein_w2` |
| Paper 5 | `paper5.provider_id`, `representation_id`, `distance_id`, and `barycentre_arm_id`; `p5_post_corpus_qmle` | `data/papers/paper5/post_corpus_qmle/results.json` | `qwen3-embedding-8b` / `qwen3-embedding-8b-unit` | 4096 | The same typed W2 artifact plus `data/shared/barycentres/qwen3-embedding-8b/qwen3-embedding-8b-unit/wasserstein_w2_loo` |
| Paper 6 | `paper6.provider_id`, `representation_id`, `distance_id`, and `barycentre_arm_id`; `p6_feasibility` | `data/papers/paper6/feasibility/contract.yaml` | `qwen3-embedding-4b` / `qwen3-embedding-4b-unit` | 2560 | Frozen raw Qwen-4B shards; t04 will materialize paper-local typed W2 and target artifacts without changing the shared canonical binding. |

Paper 5's merged representation artifact,
`data/papers/paper5/representation_ablation/summary.json`, independently marks
`qwen8b_full` as canonical and records the same provider, representation, and
4096-dimensional effective and native widths.

### Representation-sensitivity roster

`representation_ablation_specs` governs the seven cross-paper base cells below.
The canonical row is carried as the common comparison reference; the other rows
are sensitivity arms and do not replace a paper's primary selector.

| Cell | Provider ID | Representation ID | Effective / native width | Role |
| --- | --- | --- | --- | --- |
| `qwen8b_full` | `qwen3-embedding-8b` | `qwen3-embedding-8b-unit` | 4096 / 4096 | Canonical reference |
| `qwen4b_full` | `qwen3-embedding-4b` | `qwen3-embedding-4b-unit` | 2560 / 2560 | Within-family capacity and native-width arm |
| `qwen8b_1024` | `qwen3-embedding-8b` | `qwen3-embedding-8b-1024-unit` | 1024 / 4096 | Matryoshka truncation |
| `qwen4b_1024` | `qwen3-embedding-4b` | `qwen3-embedding-4b-1024-unit` | 1024 / 2560 | Fixed-width capacity arm |
| `bge_large_full` | `bge-large-en-v1.5` | `bge-large-en-v1.5-unit` | 1024 / 1024 | External architecture/training-provider arm |
| `qwen8b_256` | `qwen3-embedding-8b` | `qwen3-embedding-8b-256-unit` | 256 / 4096 | Matryoshka truncation |
| `qwen8b_64` | `qwen3-embedding-8b` | `qwen3-embedding-8b-64-unit` | 64 / 4096 | Matryoshka truncation |

The three additional `encoder_vintage_ablation_specs` cells are EttaX V0, V1,
and V3. Each uses a distinct `ettax-v*` provider and matching `ettax-v*-unit`
representation at 320 dimensions. They are materialized from governed local
checkpoints under `data/.encoder_vintage/ettax/`, not retrieved through
OpenRouter. Paper 1 treats their paired contrasts as a negative-control panel;
Paper 3 treats them as sensitivity rows. Paper 5 reports the vintage staircase
as a descriptive sensitivity and identifies V3 separately as a deliberate
post-window negative control, not as a valid point-in-time encoder.

Raw comparison shards live at `data/shared/embeddings/<provider_id>/`. Their
typed W2 inputs live at
`data/shared/typed_distances/<provider_id>/<representation_id>/wasserstein_w2`.
Paper-specific merge stages consume only the roster declared in `params.yaml`;
the broader exploratory `ablation.grid` does not silently enter publication
tables.

Provider IDs, external model IDs, representation IDs, widths, normalization
transforms, and paper selectors are all part of the reproducibility contract.
Changing one invalidates the DVC stages that declare or consume the affected
artifact and propagates through their downstream dependencies. Paper-specific
selectors keep unrelated branches distinct; they do not stop ordinary DVC
dependency propagation.

## Price-data universe (100 tickers)

Price-dependent stages (`market_data`, `compute_returns`, `compute_covariance`,
and everything downstream of the returns/covariance matrix) run on the same
**100-ticker** universe as the news stages. `market_symbols` is identical to
`symbols`, and `market_symbol_exclusions` is empty. EA and HOLX are served from
recorded market-data archives because live retrieval is no longer available;
both traded throughout the governed evaluation period and remain in the panel.

Both evaluation windows therefore use the same 100 firms, with no
news-to-price or cross-window firm attrition. The windows are declared once in
`params.yaml` as `return_panels` and drive the `compute_covariance` and
`mantel_tests` matrix families: the in-sample matrix is the `@window0` cell under
`data/shared/`, the out-of-sample matrix the `@window1` cell under
`data/shared/oos/`. Each entry also carries the complete-case contract
(`expected_tickers`, `expected_observations`, and the first/last complete date)
that `compute-covariance` enforces as a hard guard.

To change the governed price universe, update `symbols`, `market_symbols`,
`data/universe.csv`, and both `return_panels` complete-case contracts together,
then re-run
`dvc repro --glob 'compute_returns@*' 'compute_covariance@*'`.
