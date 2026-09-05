---
description: Application source code — Python, LaTeX, Lean, and TypeScript in a polyglot monorepo.
---

# src/

Application code for the monorepo, split by language ecosystem.

## Contents

| Directory | Language | Purpose |
|---|---|---|
| `python/` | Python | Data pipelines, ML, packages, scripts, notebooks |
| `latex/` | LaTeX | Sole, canonical journal and arXiv manuscript workspace |
| `lean/` | Lean 4 | Formal verification of economic theorems |
| `typescript/` | TypeScript | Bun-managed Turbo workspace for deployable web applications |

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "src"
depth = 1
-->

```text
.
├── latex # Reproducible LaTeX publishing workspace rendered end to end by a single pinned Tectonic engine.
├── lean # Formal verification of asset-pricing theorems in Lean 4 + mathlib — a Lake multi-package workspace.
├── python # Python workspace — data pipelines, ML, packages, scripts, and notebooks.
└── typescript # Bun-managed TypeScript workspace for deployable applications and shared packages.
```

<!-- insitu:end -->

## Cross-layer workflow

```bash
just setup           # uv sync
just lint            # lint all code
just fmt             # auto-format
just typecheck       # static type checking (Python and TypeScript)
just check           # prek hooks on everything
just build           # run DVC pipeline
```

See `ARCHITECTURE.md` for why these languages are co-located, and each
subdirectory's `README.md` for workspace-specific commands.
