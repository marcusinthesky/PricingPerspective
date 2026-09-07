---
description: prek — Rust drop-in replacement for pre-commit
---

# prek

[prek](https://github.com/j178/prek) is the git hook manager for the monorepo.

## Key commands

```bash
prek install                        # install git hooks
prek run --all-files                # run all hooks against the whole tree
prek run --all-files --config <f>   # run with a specific config file
```

## Auto-discovery

prek auto-discovers nested `.pre-commit-config.yaml` files in subdirectories.
Each language layer has its own hooks scoped to its directory tree:

- `src/python/.pre-commit-config.yaml` — ruff, pyrefly, uv-lock
- `src/latex/.pre-commit-config.yaml` — LaTeX lint/build hooks
- Root `.pre-commit-config.yaml` — workspace-wide hooks plus the shared policy
  and generators for hidden and non-hidden trees

On prek 0.4.4, native workspace discovery is driven by `ignore::WalkBuilder`
plus `.prekignore`. The default walk skips hidden directories, but root
`.prekignore` negations can opt exact hidden roots back in. In this repo that
directly discovers `.`, `.agents/`, and `.context/`.

See `ARCHITECTURE.md` for why prek over alternatives.
