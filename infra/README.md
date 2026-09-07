---
description: Infrastructure configuration — the reproducible nix dev shell and the OpenTofu/Terragrunt IaC managing GCP, GitHub, and Hetzner.
---

# infra/

Infrastructure configuration for the monorepo: the pinned, reproducible nix
dev shell that provides every development tool, and the OpenTofu + Terragrunt
infrastructure-as-code that manages cloud resources (GCP, GitHub, Hetzner).

## Contents

| Directory | Purpose |
|---|---|
| `nix/` | Nix flake with custom derivations and dev shell packages |
| `schemas/` | Canonical static Schematter profiles for repository documents |
| `terraform/` | OpenTofu + Terragrunt IaC — GCP (pilot: the DVC bucket), GitHub, Hetzner |

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "infra"
depth = 1
-->

```text
.
├── nix # Reproducible nix dev shell with pinned tools for Python, LaTeX, Lean, SQL, and pipeline management.
├── schemas # Canonical static document schemas consumed directly by Schematter and prek without a repository-local compiler.
└── terraform # OpenTofu + Terragrunt infrastructure-as-code — GCP (pilot the DVC bucket), GitHub and Hetzner, split by blast radius with per-component state and prek quality gates.
```

<!-- insitu:end -->

See `nix/README.md` for the tool list, `nix/ARCHITECTURE.md` for why nix over
Docker/containers, and `schemas/ARCHITECTURE.md` for the no-compiler schema
boundary.
