---
description: Understanding-oriented documentation covering design principles and the ARCHITECTURE.md convention.
title: Explanation
---

## Architecture

### What is ARCHITECTURE.md?

An `ARCHITECTURE.md` is a co-located design document that **describes, explains,
and justifies** the architectural and tool-choice decisions for the directory it
lives in. It is the "why" to the README's "what" and "how".

The convention comes from [Matklad (rust-analyzer)](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html):

> The biggest difference between an occasional contributor and a core developer
> lies in the knowledge about the physical architecture of the project.

### How this repo uses ARCHITECTURE.md

This repo follows the [Diátaxis framework](https://diataxis.fr/) for
documentation:

| Quadrant | File | Purpose |
|---|---|---|
| **Explanation** | `ARCHITECTURE.md` | Why decisions were made, trade-offs, alternatives |
| **How-to** | `README.md` | Getting started, commands, conventions |
| **Reference** | Auto-generated API docs | Type signatures, parameters |
| **Tutorials** | `AGENTS.md` chain | Progressive disclosure |

### Where ARCHITECTURE.md files live

| Location | Scope |
|---|---|
| Root `ARCHITECTURE.md` | Overview of the monorepo design |
| `src/ARCHITECTURE.md` | Python + LaTeX + Lean co-location |
| `src/python/ARCHITECTURE.md` | uv workspace design |
| `src/latex/ARCHITECTURE.md` | Canonical arXiv and journal publishing boundary |
| `src/lean/ARCHITECTURE.md` | Lake multi-package workspace |
| `src/typescript/ARCHITECTURE.md` | Bun + Turbo web application workspace |
| `infra/ARCHITECTURE.md` | Reproducible environment, static schemas, and deployment infrastructure |
| `src/python/apps/pipeline/ARCHITECTURE.md` | Python pipeline and DVC stage design |
| `infra/nix/docs/just/ARCHITECTURE.md` | just task runner design |
| `infra/nix/docs/prek/ARCHITECTURE.md` | prek hooks design |

### Design principles

**Co-location over centralization.** Documentation lives next to the code it describes.

**Single source of truth.** Every piece of architectural documentation exists in exactly one place.

**Progressive disclosure.** Documentation is layered:

1. `AGENTS.md` — agent antipatterns (always loaded, under 100 lines)
2. `README.md` — what exists, commands, structure
3. `ARCHITECTURE.md` — why decisions were made
4. `.agents/` — tool configs, skills, MCPs

**Verification over instruction.** Linters, type checkers, test suites, and
pre-commit hooks give immediate, deterministic feedback. Quality gates via `prek`
enforce formatting and linting across every layer.

**Mermaid for illustrations.** Use [Mermaid](https://mermaid.js.org/) for flow,
sequence, and architecture diagrams. Prefer `flowchart TD` (top-down); use `LR`
only when genuinely horizontal.

### Why this monorepo?

This research codebase co-locates Python (data pipelines, ML), LaTeX
(research publishing), Lean (formal proofs), and TypeScript (deployable web
applications) in a single repository. The primary benefit
is atomic cross-layer commits: a paper development, its supporting data pipeline,
and its formal proofs change together in one PR, one CI run, one revert.

The alternative — separate repos for Python, LaTeX, Lean, and TypeScript — would require
coordinated releases and cross-repo dependency management for every feature.

### Why nix + just + prek?

| Tool | Replaces | Why |
|---|---|---|
| **nix** | Docker, pip/system deps | Single flake provides every tool; pinned versions via `flake.lock`; reproducible dev shells |
| **just** | Make, shell scripts, npm scripts | Polyglot command runner; `just --list` for discoverability; `mod` imports for modularity |
| **prek** | pre-commit (Python) | Rust binary; same YAML config; auto-discovers nested configs |

### Why LaTeX as the sole manuscript renderer?

LaTeX (`src/latex`) is the sole, canonical manuscript source following the
2026-07-23 t24.12 cutover, because arXiv and target journals expose their
source templates and production systems in TeX. Tectonic preserves a
watch-style loop; a minimal pinned TeX Live checks publisher compatibility.
Typst was retained frozen through the cutover as a rollback baseline and as
the source for presentation slides and the DVC DAG renderer; it has since
been fully removed from the repository (`src/typst/` deleted) once the
LaTeX port reached parity — never permanent dual canonical sources.

### Why Lake multi-package workspace?

Lake's multi-package workspace pattern keeps proofs modular: each package has
its own `lakefile.lean` and dependencies, while the root `lakefile.lean`
coordinates the build. This is the standard pattern for Lean 4 monorepos
(e.g., mathlib).
