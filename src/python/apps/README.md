---
description: Runnable Python applications in the uv workspace.
---

# src/python/apps/

Runnable products and scientific workflows. The encoder-vintage GPU application
is documented in [EttaX](ettax/README.md); reusable consumer-agnostic libraries
belong under `src/python/packages/`.

`insitu-repository/` is the executable policy adapter for this repository's
Markdown projections; its reusable reconciliation engine is `packages/insitu/`.
`harnessme/` is the flattened DuckDB/dbt application boundary for repository
introspection, retrieval, architecture analysis, and its mart-backed dashboards.

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "src/python/apps"
depth = 1
-->

```text
.
├── ettax
├── insitu-repository # Pricing Perspective projectors and orchestration for the reusable Python Insitu engine.
├── manimize
├── pipeline
└── simulation # GPU-capable JAX Monte-Carlo simulation package for energy/Wasserstein metrics and portfolio optimisation.
```

<!-- insitu:end -->
