# Pricing Perspective monorepo task runner.
# Run `just` (no args) to list all recipes.

set shell := ["bash", "-uc"]

# Dev shell (flake devShell) + nix formatting.
mod nix 'infra/nix/justfile'
# Python uv workspace: apps (pipeline/simulation/crawler) + packages (jcor).
mod python 'src/python/justfile'
# Lean 4: build, mathlib cache, sorry/axiom allowlist gate.
mod lean 'src/lean/justfile'
# Marimo notebook workspace (src/python/notebooks).
mod marimo 'src/python/notebooks/justfile'
# Reference CSL validation, BibTeX export, and citation graph.
mod references 'src/python/scripts/references/justfile'
# Materialized Markdown views: reusable Python core plus repository projectors.
mod insitu 'src/python/apps/insitu-repository/justfile'
# Plan layer: graph, board, ROADMAP map.
mod plan 'src/python/apps/insitu-repository/justfile'
# LaTeX manuscripts: build, fmt, smoke.
mod latex 'src/latex/justfile'
# TypeScript Bun workspace and Turbo task graph.
mod typescript 'src/typescript/justfile'
# OpenTofu/Terragrunt infra: fmt, validate, lint.
mod terraform 'infra/terraform/justfile'

# Default: list available recipes.
default:
    @just --list

# One-time local setup: LaTeX needs nothing here: its bundle subset is
# vendored in the tree (`just latex::bundle-refresh` regenerates it), so every
# compile is already offline.
setup: python::setup typescript::setup

# Install git hooks via prek (drop-in pre-commit replacement).
hooks:
    prek install

# Lint everything.
lint: python::lint latex::lint typescript::lint

# Auto-format all code.
fmt: nix::fmt python::fmt terraform::fmt latex::fmt typescript::fmt

# Static type checking.
typecheck: python::typecheck typescript::typecheck

# Check Tach's module-boundary pilot and uv workspace external dependencies.
tach: python::tach

# Run the test suite.
test: python::test

# Refresh semantic fingerprints, rebuild changed DVC stages, and refresh the tracked DAG image.
build:
    uv run --project src/python --no-sync semflow check
    dvc repro
    just pipeline-graph

# Render DVC's native collapsed Mermaid graph as the tracked PNG overview.
pipeline-graph:
    dvc dag --collapse-foreach-matrix --mermaid | merman-cli render - --format png --background white --raster-unbounded --quiet --output pipeline.png

# Run all configured prek hooks against the whole tree.
check:
    prek run --all-files

# Diff-scoped prek: hooks only on files changed vs BASE (default uncommitted vs HEAD; pass `main` for the PR set) — fast pre-commit gate, no --all-files sweep.
check-diff base="HEAD":
    #!/usr/bin/env bash
    set -euo pipefail
    mapfile -t all < <( { git diff --name-only "{{ base }}"; git ls-files --others --exclude-standard; } | sort -u )
    files=()
    for f in "${all[@]}"; do [ -e "$f" ] && files+=("$f"); done
    if [ ${#files[@]} -eq 0 ]; then echo "check-diff: no changed files vs {{ base }}"; exit 0; fi
    echo "check-diff: ${#files[@]} changed file(s) vs {{ base }}"
    prek run --files "${files[@]}"
