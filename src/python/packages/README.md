---
description: Reusable Python libraries and packages in the uv workspace.
---

# src/python/packages/

Shared Python libraries consumed by pipeline scripts, notebooks, and
(optionally) applications.

## Contents

| Package | Description | Key deps |
|---|---|---|
| `insitu/` | Typed, reactive in-place document materializations | `Pydantic`, `watchfiles` |
| `jcor/` | JAX-based distance correlation and energy statistics | `jax[cuda12]`, `jaxopt` |
| `agentrail/` | Provider-neutral workflow contracts and deterministic orchestration | `pydantic`, `pydantic-graph` |
| `dotell/` | Structured logging and process-resource telemetry | `loguru`, `psutil` |
| `semflow/` | Semantic Python fingerprints and review-first DVC patches | `PyYAML`, `Typer` |

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "src/python/packages"
depth = 1
-->

```text
.
├── dotell
├── insitu # Reactive, typed materializations for Markdown and YAML front matter.
├── jcor
└── semflow # Semantic Python fingerprints and review-first DVC dependency patches that suppress non-executable pipeline churn without taking ownership of the stage graph.
```

<!-- insitu:end -->

Each package follows the `src` layout with `py.typed` markers for PEP 561
compliance, has its own `pyproject.toml`, and is a member of the uv workspace
defined in `src/python/pyproject.toml`.

## Usage

```bash
# Install all workspace packages
uv sync --all-packages

# Install a specific package
uv sync --package jcor

# Run package tests
uv run pytest src/python/packages/jcor/tests/
```

## Adding a package

1. Create `packages/<name>/` with `src/<name>/__init__.py`
2. Add `pyproject.toml` with `[build-system]` and deps
3. Ensure `src/python/pyproject.toml` workspace members includes `packages/*`
4. Run `uv lock` from `src/python/`
