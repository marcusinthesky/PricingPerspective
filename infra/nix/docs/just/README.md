---
description: just — the polyglot command runner for the monorepo
---

# just

[just](https://github.com/casey/just) is the monorepo's command runner.

## Key commands

```bash
just --list      # list all available commands
just setup       # install workspace deps
just lint        # lint all code
just fmt         # auto-format all code
just typecheck   # static type checking
just check       # run all prek hooks
just build       # rebuild DVC pipeline
just hooks       # install git hooks
```

## Module structure

The root justfile imports sub-modules via `mod`:

```just
mod nix 'infra/nix/justfile'
mod python 'src/python/justfile'
```

Sub-module recipes are available as `nix::fmt`, `python::lint`, etc.
Aggregate recipes like `just fmt` chain layer-specific targets.

See `ARCHITECTURE.md` for why just over alternatives.
