# Pricing Perspective

Research programme exploring distribution-valued firm information, financial dependence,
portfolio risk and competitive adjustment.

[![Site](https://img.shields.io/badge/site-GitHub%20Pages-222?logo=github)](https://marcusinthesky.github.io/statadic/)
[![DOI](https://zenodo.org/badge/1358295759.svg)](https://doi.org/10.5281/zenodo.22689254)  

[![Python 3.13–3.14](https://img.shields.io/badge/python-3.13%E2%80%933.14-3776AB?logo=python&logoColor=white)](src/python/packages/statadic/docs/start/installation.md)
[![Lean 4.35.0](https://img.shields.io/badge/Lean-4.35.0-0E7490?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMjQwMCIgaGVpZ2h0PSI5MDAiIHZpZXdCb3g9IjAgMCAyNDAwIDkwMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMTUxNy40MyA3NDEuMDM2VjIxMy41NTFMMTQwOC44NiA0NTQuOTk3QzE0MDguODQgNDU1LjAzOSAxNDA4LjgzIDQ1NS4wODMgMTQwOC44MSA0NTUuMTI1TDEyNzQuODcgNzQzLjE2N0MxMjczLjA3IDc0Ny4wMjUgMTI2OS4yMSA3NDkuNDk4IDEyNjQuOTYgNzQ5LjUwOUwxMjYwLjQ5IDc0OS41MkMxMjU2LjIxIDc0OS41MyAxMjUyLjMxIDc0Ny4wNDIgMTI1MC41MSA3NDMuMTUzTDExMTcuNDkgNDU1LjE1TDEwMDMuNSAyMTEuNzAyVjczOC41NTZDMTAwMy41IDc0NC42MTEgOTk4LjU4OSA3NDkuNTIgOTkyLjUzNiA3NDkuNTJIMjU2Ljk2MUMyNTAuOTA4IDc0OS41MiAyNDYuMDAxIDc0NC42MTEgMjQ2IDczOC41NTZWMTU5Ljk2NEMyNDYgMTUzLjkwOSAyNTAuOTA3IDE0OSAyNTYuOTYxIDE0OUMyNjMuMDE0IDE0OSAyNjcuOTIxIDE1My45MDkgMjY3LjkyMSAxNTkuOTY0VjcyNy41OTJIOTgxLjU3NlY0NjEuNDY0SDYyNC43NUM2MTguNjk3IDQ2MS40NjQgNjEzLjc5IDQ1Ni41NTUgNjEzLjc5IDQ1MC41QzYxMy43OSA0NDQuNDQ1IDYxOC42OTcgNDM5LjUzNiA2MjQuNzUgNDM5LjUzNkg5ODEuNTc2VjE3My40MTFINjI0Ljc1QzYxOC42OTcgMTczLjQxMSA2MTMuNzkgMTY4LjUwMiA2MTMuNzkgMTYyLjQ0OEM2MTMuNzkgMTU2LjM5MyA2MTguNjk3IDE1MS40ODQgNjI0Ljc1IDE1MS40ODRIOTkyLjUzNkw5OTMuMzI4IDE1MS41MTNDOTk3LjI2NCAxNTEuNzk3IDEwMDAuNzcgMTU0LjE4NSAxMDAyLjQ2IDE1Ny43OTdMMTEzNC4zOCA0MzkuNTM2SDEzOTEuNzhMMTUxOC4zOSAxNTcuOTUxQzE1MjAuMTYgMTU0LjAxNSAxNTI0LjA3IDE1MS40ODQgMTUyOC4zOSAxNTEuNDg0SDE1MzAuODJDMTUzNC4xNSAxNTEuNDg0IDE1MzcuMjkgMTUyLjk5MyAxNTM5LjM3IDE1NS41ODVMMTk4NS4wOCA3MTAuOTA0VjE1OS45NjRDMTk4NS4wOCAxNTMuOTA5IDE5ODkuOTkgMTQ5IDE5OTYuMDQgMTQ5QzIwMDIuMDkgMTQ5IDIwMDcgMTUzLjkwOSAyMDA3IDE1OS45NjRWNzM5LjA0MUMyMDA3IDc0NS4wOTYgMjAwMi4wOSA3NTAuMDA1IDE5OTYuMDQgNzUwLjAwNUgxOTkzLjZDMTk5MC4yOCA3NTAuMDA1IDE5ODcuMTQgNzQ4LjQ5NiAxOTg1LjA2IDc0NS45MDRMMTUzOS4zNSAxOTAuNTgxVjc0MS4wMzZDMTUzOS4zNSA3NDcuMDkxIDE1MzQuNDQgNzUyIDE1MjguMzkgNzUyQzE1MjIuMzMgNzUyIDE1MTcuNDMgNzQ3LjA5MSAxNTE3LjQzIDc0MS4wMzZaTTEyNjIuNzIgNzE3LjI5NkwxMzgxLjY4IDQ2MS40NjRIMTE0NC41NUwxMjYyLjcyIDcxNy4yOTZaIiBmaWxsPSIjMDcwMDAwIi8%2BPHBhdGggZD0iTTIwNTEuMDggMTYzLjk2NVYxNTUuMDkxSDIwOTIuODlWMTYzLjk2NUgyMDc3LjMxVjIwNkgyMDY2LjY3VjE2My45NjVIMjA1MS4wOFoiIGZpbGw9IiNCN0I3QjciLz48cGF0aCBkPSJNMjA5OS43OCAxNTUuMDkxSDIxMTMuMDVMMjEyNy4wNyAxODkuMjk1SDIxMjcuNjdMMjE0MS42OSAxNTUuMDkxSDIxNTQuOTZWMjA2SDIxNDQuNTJWMTcyLjg2NEgyMTQ0LjFMMjEzMC45MiAyMDUuNzUxSDIxMjMuODFMMjExMC42NCAxNzIuNzRIMjExMC4yMlYyMDZIMjA5OS43OFYxNTUuMDkxWiIgZmlsbD0iI0I3QjdCNyIvPjwvc3ZnPg%3D%3D)](https://lean-lang.org/)
[![Nix 2.35.4](https://img.shields.io/badge/built%20with-Nix%202.35.4-5277C3?logo=nixos&logoColor=white)](infra/nix/flake.nix)  

[![License: Apache 2.0](https://img.shields.io/github/license/marcusinthesky/PricingPerspective)](LICENSE)

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
The manuscript targets, PDFs and source archives are restricted to arXiv.

## Reproduction

Install Nix, devenv and Git LFS.
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
The [data guide](data/README.md) identifies the fixed historical numeric
benchmarks and publication comparisons. The [manuscript guide](src/latex/README.md)
describes the four arXiv builds and source bundles.

Run `just --list` for the maintained commands. The repository uses `prek` for
formatting and validation; `just check` runs its configured checks.

## Disclosures

The authors have no conflicts of interest or funding to disclose. Generative
models assisted editorial review, formatting and proof formalization.
