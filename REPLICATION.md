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
  first three import. `manimize` (the animation package for the dissertation) is
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

`src/python/uv.lock` is locked to exactly the members present here.

## What is not included

- Development history, research notes, correspondence, and agent tooling.
- Test suites, benchmarks, and implementation code outside the runtime closure.
- The DVC-managed object store. Git carries the `.dvc` pointers and the
  `dvc.lock` hashes, not the data itself.
- `.dvc/config` — no DVC remote is configured. An authorized remote must be
  added before `dvc pull` can resolve anything.
- `src/latex/projects/00_dissertation`. `dvc.yaml` still declares the
  `render_dissertation` stage, but its inputs are absent, so that stage cannot
  be reproduced from this snapshot. The three paper chains above are the
  supported terminals.

## Checking out

Figures, PDFs, the Tectonic bundle, and the recorded metric series are stored in
Git LFS. Install it first, or the working tree will contain pointer files
instead of content:

```bash
git lfs install
git clone https://github.com/marcusinthesky/PricingPerspective.git
```

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

Restore the DVC objects from an authorized remote, then drop `--dry` and
`--allow-missing`:

```bash
dvc repro p1_numbers p3_numbers p5_numbers
```

Frozen embedding and market-data stages remain governed by the hashes recorded
in `dvc.lock`.
