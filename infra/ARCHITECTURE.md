---
description: Explanation and reference about "why" decisions were made.
title: Explanation
---

## infra/

Nix was chosen to provide a single, reproducible dev shell for the repository's
Python, LaTeX, Lean, and operator toolchains with zero per-machine setup beyond
nix + direnv.

### Why a single flake for everything?

Alternative: separate nix files per language, or Docker Compose with multiple
containers. A single flake means `nix develop` gives you `uv`, `tectonic`, TeX
Live, `lean`, and `merman-cli` simultaneously — no switching contexts. The
`flake.lock` pins everything
deterministically.

#### Tool selection rationale

- **uv** over pip/poetry: faster, Rust-based, workspace support, PEP 723 scripts
- **pyrefly** over ty/mypy/pyright: parses jaxtyping shape strings
  (`Float[Array, "n d"]`) natively and understands PEP 723 inline metadata, so
  neither the workspace pass nor the standalone-scripts pass needs suppressions
  (see `src/python/ARCHITECTURE.md`)
- **Tectonic + minimal TeX Live** for LaTeX, the sole, canonical manuscript
  renderer; the DVC DAG uses the native Rust Merman renderer, so no Typst or
  browser runtime remains in that path
- **prek** over pre-commit: Rust binary, same YAML config, no Python dependency
- **dvc-with-remotes** over dvc: includes GCS/S3/Azure backends for remote storage

### Why are document schemas static?

The repository consumes Schematter YAML profiles directly. Keeping that consumed
form under `infra/schemas/documents/` avoids a Python source-to-generated build edge
and makes schema changes reviewable in one representation. See
[`schemas/ARCHITECTURE.md`](schemas/ARCHITECTURE.md) for the boundary and the
owner-local policy for runtime JSON contracts.
