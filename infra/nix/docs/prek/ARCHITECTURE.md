---
description: Why prek as a drop-in pre-commit replacement instead of pre-commit or lefthook.
title: prek
---

The monorepo uses [prek](https://github.com/j178/prek), a Rust drop-in for
`pre-commit`. It is a single binary, uses the same `.pre-commit-config.yaml`
format, runs hooks in parallel, and auto-discovers nested config files.

## Why prek over alternatives?

| Alternative | Problem |
|---|---|
| **pre-commit (Python)** | Requires Python runtime; slow startup; large install footprint |
| **lefthook** | Different config format; smaller community; fewer built-in hooks |
| **husky** | Node dependency; npm scripts coupling |
| **No hooks** | Quality drift; formatting inconsistencies across contributors |

prek gives us the pre-commit ecosystem (hooks, repos, formats) with zero
Python overhead. It auto-discovers nested `.pre-commit-config.yaml` files
in subdirectories like `src/python/` and `src/latex/`.

Current prek workspace discovery uses `ignore::WalkBuilder` together with
`.prekignore` via `add_custom_ignore_filename(".prekignore")`. There is no
hard-coded hidden-directory rejection in prek's workspace layer. The default
ignore walk still skips hidden paths, but root `.prekignore` negations can
re-include exact hidden roots for native project discovery.

That matches local testing on prek 0.4.4: the repo's root `.prekignore`
configuration lets prek discover `.`, `.agents/`, and `.context/` as workspace
projects without owner-dispatch wrapper hooks. The architecture keeps hidden
directory names intact, uses `.prekignore` as the inclusion surface, splits
`.agents/` into provider projects and `.context/` into corpus projects, and
leaves the root config to own cross-tree policy and generators.

Nested filters and cwd handling stay project-relative once a project is
discovered, so the design does not need extra dispatch shims for hidden trees.

## How we use it

- Install hooks: `prek install` (or via `.envrc` on shell entry)
- Run all hooks: `prek run --all-files` (or `just check`)
- Auto-discovered nested configs handle language-specific linting/formatters

## Trade-offs

prek is newer than pre-commit and has a smaller community. Some edge-case
hook features (like `fail_fast`) may behave slightly differently. Hidden-tree
discovery also depends on keeping `.prekignore` negations accurate. For the
hooks used in this repo (ruff, pyrefly, builtin, codespell, conventional-commit),
prek is a drop-in replacement.
