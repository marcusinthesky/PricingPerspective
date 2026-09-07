---
description: Reproducible nix dev shell with pinned tools for Python, LaTeX, Lean, SQL, and pipeline management.
---

# infra/nix/

The nix flake provides every tool needed for the monorepo — pinned to specific
versions via `flake.lock`. No system-level installs required beyond nix + direnv.

## Usage

```bash
nix develop              # enter the dev shell
direnv allow             # or: auto-enter via direnv
just --list              # see available commands
just setup               # uv sync
```

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "infra/nix"
depth = 1
-->

```text
.
├── docs # Usage notes for the vendored binaries pinned in the nix dev shell (just, prek, dotenvx).
├── pkgs # Packages this flake vendors because nixpkgs has no usable attribute for them — one file per derivation.
├── shells # The flake's devShells — an interactive default plus workload-minimal Lean, prek, and replication CI slices.
└── toolchain-provenance # Provenance gate asserting every CLI tool the repository declares is provisioned by the Nix dev shell rather than a machine-local profile.
```

<!-- insitu:end -->

## Tools provided

Generated from `shells/groups/*.nix` — the same list the shell is built from, so
it cannot drift. Tool names are the command you type, not the Nix attribute; run
`<tool> --help` for detail, and see the group file for why a tool is here at all.

<!-- insitu:begin tool_inventory
id = "tool-inventory"
-->

| Group | Provides | Tools |
|---|---|---|
| **shell** | Interactive shell, version control, and the GitHub CLI. | zsh, git, git-lfs, gh |
| **tools** | Operator tooling: secrets, the DuckDB/dbt data plane, DVC, the hook runner, watch loops, data transformation, and rendering. | jq, yq, sd, dotenvx, secret-tool, dvc, prek, duckdb, harnessme-sql-gates, pandoc, watchexec, difft, graphviz, merman-cli, ffmpeg, espeak-ng |
| **languages** | Every language in the repo and its toolchain — runtime, type checker, formatter, linter, LSP — from Python and Lean through LaTeX, Markdown, Nix, JSON, and HCL. | uv, pyrefly, bun, deno, lean, harper-ls, tectonic, tex-fmt, texlab, chktex, pdflatex, latexdiff, just, just-lsp, nixfmt, statix, deadnix, nixd, yaml-language-server, marksman, rumdl, schematter, lychee, frontmatter, tombi, dbt-language-server, jsonschema-cli, vscode-langservers-extracted, bash-language-server, tofu, terragrunt, tflint, terraform-ls, trivy |
| **search** | Discovery: find files, text, and code — exact, then structural, then semantic — and size up what the repository contains. | rg, fd, fzf, ast-grep, semble_rs, tokei |
| **documents** | Native PDF/raster primitives used by the visual-document-analysis skill. | pdf2md, poppler-utils, pdfannots, qpdf, ocrmypdf, tesseract, mupdf, magick, diff-pdf |
| **diagnostics** | Runtime cost: command benchmarking and live Python profiling. | hyperfine, py-spy |
| **cloud** | Cloud provider CLI: ADC credentials for the OpenTofu google provider, and brownfield resource export. | gcloud |

<!-- insitu:end -->

## Flake inputs

| Input | Role |
|---|---|
| `nixpkgs` | The pin every tool resolves from (`nixos-unstable`, frozen by `flake.lock`) |
| `nixpkgs-duckdb` | A second, separately-moving pin used **only** for the DuckDB CLI, so it can track the version `src/python/apps/harnessme/extensions.lock.yml` requires without moving the whole shell |
| `flake-utils` | `eachSystem` over x86_64-linux and aarch64-linux |
| `lean4-nix` | Source-only (`flake = false`); supplies the `fetchBinaryLean` helper |
| `uv2nix`, `pyproject-nix`, `pyproject-build-systems` | Build the hermetic dbt/DuckDB Python env from a `uv.lock` |
| `python-workspace` | The `src/python` uv workspace, consumed as a source input; the SQL environment selects only the `harnessme` member closure |

Vendored third-party sources are pinned by commit and content hash in
`pkgs/*.nix`.

## Layout

| Path | Holds |
|---|---|
| `flake.nix` | Inputs and output wiring, and nothing else |
| `pkgs/` | One file per vendored derivation, each with its own `BUMPING:` runbook |
| `shells/` | `devShells`: per-domain package groups plus the shell environment |
| `docs/` | Usage notes for vendored third-party binaries (just, prek, dotenvx) |

Prose is tiered deliberately. A one-line descriptor sits beside each package; a
`BUMPING:`/`LOCKSTEP:` runbook sits beside the derivation it governs, because
that is read at the moment the code is edited; and the "why this tool and not
the alternative" arguments live in `ARCHITECTURE.md`.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for why nix over Docker, why Tectonic is
the sole renderer yet a second TeX engine still ships, why the Lean toolchain is
a store derivation, and why the profiler set stops where it does.
