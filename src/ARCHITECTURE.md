---
description: Why Python, LaTeX, Lean, and TypeScript are co-located in a polyglot monorepo.
title: Application Layer
---

## src

### Why these languages?

The monorepo covers four distinct application domains, each with the
best-fit language:

| Language | Domain | Why |
|---|---|---|
| **Python** | Data, ML, scripts, notebooks | Ecosystem (NumPy, pandas, JAX); rapid prototyping; marimo notebooks |
| **LaTeX** | Canonical journal and arXiv manuscripts (sole manuscript renderer) | Publisher classes, source portability, TeX publishing ecosystem |
| **Lean** | Formal verification | Proof assistant; formalizes economic theorems |
| **TypeScript** | Deployable web applications | Bun and Turbo provide an isolated, cacheable frontend workspace |

### Why co-locate?

Atomic cross-layer commits. A feature that touches a Python data pipeline, a
LaTeX paper, a Lean proof, and a TypeScript site is a single PR, a single CI
run, a single revert. The alternative — separate repos — requires coordinated
releases and cross-repo dependency management.

### Structure

Each language directory is a self-contained workspace with its own package
manager, lock file, linting config, and pre-commit hooks. They share the
monorepo's quality gates (`just check`, conventional commits) but keep
language-specific tooling isolated via nested `.pre-commit-config.yaml` files.

```text
src/
├── python/       # uv workspace (apps + packages + notebooks + scripts)
├── latex/        # Canonical publishing workspace (Tectonic + TeX Live gates)
├── lean/         # Formal verification
└── typescript/   # Bun-managed Turbo web application workspace
```

See each subdirectory's `README.md` for workspace-specific commands and
`ARCHITECTURE.md` for tool-choice rationale.
