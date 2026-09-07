# Pricing Perspective

**Three papers. One geometric language for financial dependence.** Probability-valued firm information is used to study asset co-movement, construct interaction fields, and certify portfolio diversification.

[![License: Apache 2.0](https://img.shields.io/github/license/marcusinthesky/PricingPerspective)](LICENSE)
[![Site](https://img.shields.io/badge/site-GitHub%20Pages-222?logo=github)](https://marcusinthesky.github.io/PricingPerspective/)
[![Built with Nix](https://img.shields.io/badge/built%20with-nix-5277C3?logo=nixos&logoColor=white)](infra/nix/flake.nix)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![DVC](https://img.shields.io/badge/data-DVC-13ADC7?logo=dvc&logoColor=white)](dvc.yaml)
[![Lean 4](https://img.shields.io/badge/proofs-Lean%204-000?logo=lean&logoColor=white)](src/lean)

## Papers

Separation, reconstruction, dispersion — each paper changes which object is held fixed and which is allowed to vary.

| # | Paper | Class | arXiv |
|---|---|---|---|
| 01 | Systematic Covariance Envelopes from Wasserstein Geometry: Evidence from Language-Model Representations | `q-fin.CP` | [![arXiv](https://img.shields.io/badge/arXiv-2410.23447-b31b1b.svg)](https://arxiv.org/abs/2410.23447) |
| 02 | Wasserstein-Barycentric Interaction Fields for Spatial Factor Models: Evidence from Language-Model Representations | `q-fin.ST` | [![arXiv](https://img.shields.io/badge/arXiv-2608.29669-b31b1b.svg)](https://arxiv.org/abs/2608.29669) |
| 03 | Portfolio Risk Bounds without Cross-Asset Return Covariances: Distributional Fields from Language-Model Representations | `q-fin.ST` | [![arXiv](https://img.shields.io/badge/arXiv-2608.29692-b31b1b.svg)](https://arxiv.org/abs/2608.29692) |

Marcus Gawronsky, Chun-Sung Huang. Abstracts, the narrated video, and the podcast explainer are on the [project site](https://marcusinthesky.github.io/PricingPerspective/).

## Introduction

Read all adjacent and parent `README.md` files, these files offer progressive disclosure of the project's structure and purpose and document key concepts, conventions and dependencies — which must be followed.

Read `ARCHITECTURE.md` files for the "why" behind architectural and tool decisions — comparison tables, trade-offs, and alternatives considered. These are the canonical source of truth for design decisions.

## Pipeline

The DVC stage graph below (foreach/matrix families collapsed) is a static snapshot of `dvc.yaml`.

![DVC pipeline stage graph](pipeline.png)

## Environment

Use `nix develop` when running terminal commands to ensure the correct environment is set up.

## Commands

Run `just --list` to see a list of available commands, at different levels of abstraction across the codebase.

## Code Quality

This repository imposes strict quality gates and automated code-generation using `prek` `.pre-commit-config.yaml` hooks, which must be installed and configured correctly, and passed before any code can be committed.

## Conflicts of Interest

The authors bear no conflicts of interest to disclose.

## Funding

There is no fundings needed to disclose.

## Use of Generative Models

Generative Models have been used in later revisions of this work to aid in editorial review, formatting of final manuscripts and aid in proof formalization.
