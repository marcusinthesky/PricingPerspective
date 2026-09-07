---
description: Pricing Perspective projectors and orchestration for the reusable Python Insitu engine.
---

# insitu-repository

This workspace app is the repository-specific edge around the reusable
[`insitu`](../../packages/insitu/) package. It supplies typed transforms for
Git-tracked directory trees, the DVC DAG, and the Nix tool
inventory, while plan children and lifecycle backlinks are canonical SQL/Jinja
materializations. It exposes one `check`/`sync`/`watch` lifecycle.

Use the root `just insitu::…` recipes rather than invoking this package directly.
