---
description: Why just is the polyglot command runner instead of Make, shell scripts or npm scripts.
title: just
---

The monorepo uses [just](https://github.com/casey/just) as its command runner.
Each language layer exposes the same recipe names — `setup`, `lint`, `fmt`,
`check` — composed via `mod` imports into a single root justfile.

## Why just over alternatives?

| Alternative | Problem |
|---|---|
| **Make** | Syntax is painful; designed for C compilation, not task running |
| **Shell scripts** | No dependency tracking; no `--list`; hard to discover |
| **npm scripts** | Locked to Node ecosystem; `package.json` bloat |
| **Task (go-task)** | Go-centric; heavier install; smaller ecosystem |

just gives us `just setup`, `just lint`, `just fmt`, `just check` — identical
recipe names across all language layers. New developers run
`just --list` and see every available command. Recipes compose via `mod`
imports, keeping each layer's justfile self-contained.

## Conventions

- Every recipe has a doc comment (shown by `just --list`).
- Every justfile starts with `set shell := ["bash", "-uc"]`.
- Standard recipe names: `setup`, `lint`, `fmt`, `typecheck`, `check`.
- Module composition via `mod`, never `import`.
