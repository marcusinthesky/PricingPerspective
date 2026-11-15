# Pricing Perspective

Four papers on distribution-valued firm information, financial dependence,
portfolio risk and competitive adjustment.

[![License: Apache 2.0](https://img.shields.io/github/license/marcusinthesky/PricingPerspective)](LICENSE)
[![Lean 4](https://img.shields.io/badge/proofs-Lean%204-000)](src/lean)
[![DVC](https://img.shields.io/badge/data-DVC-13ADC7)](dvc.yaml)

| Project | Paper | Preprint |
| --- | --- | --- |
| 01 | Systematic covariance envelopes from Wasserstein geometry | [2410.23447](https://arxiv.org/abs/2410.23447) |
| 03 | Portfolio risk bounds without cross-asset return covariances | [2608.29692](https://arxiv.org/abs/2608.29692) |
| 05 | Wasserstein-barycentric interaction fields for spatial factor models | [2608.29669](https://arxiv.org/abs/2608.29669) |
| 06 | Information Games: Strategic Crowding and Firm Repositioning in Language-Model Space | [2609.37820](https://arxiv.org/abs/2609.37820) |

Marcus Gawronsky and Chun-Sung Huang. Read the papers and companion explanations
on the [project website](https://marcusinthesky.github.io/PricingPerspective/).

The repository contains the numerical pipeline, Lean proofs, LaTeX sources,
website and tests needed to reproduce these papers. Statistical computation uses
the public [Statadic](https://github.com/marcusinthesky/statadic) package; document
projections use public [Insitu](https://github.com/marcusinthesky/insitu).

## Reproduction

Install Nix, devenv and Git LFS, then follow [REPLICATION.md](REPLICATION.md).
The pinned environment provides Python 3.13, Lean 4.35 and the manuscript tools.

```sh
devenv shell -- just setup
devenv shell -- dvc pull
devenv shell -- just python::test-all
devenv --profile lean shell -- just lean::check
devenv shell -- dvc repro
devenv shell -- just reproduction-check
```

Reproduction starts from published embeddings, adjusted returns and text-free
numeric controls. Article text and scraping implementations are not distributed.
The [replication guide](REPLICATION.md) identifies the fixed historical numeric
benchmarks and explains the published-precision comparison.

Run `just --list` for the maintained commands. The repository uses `prek` for
formatting and validation; `just check` runs its configured checks.

## Disclosures

The authors have no conflicts of interest or funding to disclose. Generative
models assisted editorial review, formatting and proof formalization.
