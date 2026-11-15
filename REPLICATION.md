# Reproducing the four papers

The supported projects are `01_continuous_bounds`, `03_distance_implied_mpt`,
`05_spatial_pricing` and `06_information_motion`. The public DVC graph computes
their numerical analyses, renders their tables and figures, certifies the Lean
claims and compiles the manuscripts.

## Environment and inputs

Install Git LFS before cloning so the TeX bundle, figures and publication assets
are materialized. Enter the pinned environment and install the locked packages:

```sh
git lfs install
devenv shell -- just setup
devenv shell -- dvc pull
```

The default DVC remote reads anonymously from the public
[PricingPerspective data bucket](https://huggingface.co/buckets/marcusinthesky/PricingPerspective).
Readers need no cloud account or API credentials. Git stores the source and DVC
graph; the data objects are downloaded separately and checked by their hashes.
The frozen `fixed_inputs` stage groups the released numerical inputs. Its
fail-closed command is never executed by reproduction, including with `--force`.

The input boundary consists of:

- Computed document embeddings with stable identifiers, firm labels, dates,
  dimensions and numeric text-length metadata. Vectors and cohort membership
  are preserved exactly; article text and titles are excluded.
- Text-free cohort manifests and corpus counts, separate anonymous sparse
  TF–IDF fits, co-mention events, and per-firm numeric liquidity controls.
- Adjusted-return panels, public factor data and classification metadata.
- Fixed numeric historical benchmarks used by the four papers.

The pipeline does not scrape news, download encoders or request embeddings.
`validate-public-inputs` rejects changed shard checksums, changed row counts and
unapproved columns before the downstream embedding computations.

## Verification and execution

```sh
devenv shell -- just python::test-all
devenv --profile lean shell -- just lean::check
devenv --profile lean shell -- just lean::certify
devenv shell -- just latex::check
devenv shell -- dvc repro
devenv shell -- just reproduction-check
```

Run each Python package's tests in its own process; the root recipes do this.
The complete tests do not use Testmon selection. The approved numerical settings,
resampling schedules and seeds remain in `params.yaml` and the numeric design
inputs. Computational stages own float64 initialization.

A full recomputation can use `dvc repro --force` after downloading the fixed
inputs. Run one DVC reproduction at a time in a checkout. Ordinary `dvc repro`
reuses unchanged intermediates and resumes at completed stage boundaries.

`just reproduction-check` compares manuscript values at their displayed
precision, literal table values and exact counts with the reviewed baseline.
It also compares the numerical geometry of 19 figures and the decoded pixels of
two heatmaps. The baselines in `data/reference/published_values.json` and
`data/reference/published_results.json` are fixed inputs; PDF timestamps and
code/proof provenance are excluded. Cohort identifiers and counts are additionally
checked by the input and experiment contracts. Numerical differences require
review; do not replace the baselines merely to make a check pass.

## Historical numeric benchmarks

Paper 6's completed information-recovery campaign and historical observation
checks are fixed, checksummed numeric inputs. Current directional calibration,
corporate projection, article-redraw validation and inference audits are computed
by their retained stages. The saved campaign remains read-only; its checkpoints
and captured input hashes are audited before use.

Paper 5's reference-sample bridge recomputes point estimates from the published
returns and interaction matrix, and compares them with the fixed reference
estimate and historical interval endpoints. These references are benchmarks,
not newly generated historical campaigns.

Sanitizing inputs and migrating dependency namespaces can change file hashes and
provenance without changing estimates. The public input manifest records the
released bytes, and certification regenerates the current proof identities.

## Manuscripts and website

`just latex::check` builds the four arXiv papers and retained venue targets
against the vendored offline TeX bundle. The bibliography is generated from
citation metadata under `.context/reference`. Submission bundles contain paper
sources and required assets only.

The website uses the same four paper identities and current results. Its static
export is configured for `/PricingPerspective` on GitHub Pages.
