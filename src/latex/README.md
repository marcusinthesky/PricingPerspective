---
description: Reproducible LaTeX publishing workspace rendered end to end by a single pinned Tectonic engine.
---

# src/latex/

This workspace is the sole, canonical publishing source for journal and arXiv
manuscripts as of the 2026-07-23 t24.12 cutover, rendered by the DVC `render`
matrix stage. Typst has been fully removed from the project (see
[`ARCHITECTURE.md`](ARCHITECTURE.md)).

## Commands

Run from the repository root:

```bash
just latex::new 01_example              # scaffold a project
just latex::watch 01_example draft      # rebuild on save
just latex::build 01_example arxiv      # the PDF (add `1` to forbid fetching)
just latex::bundle 01_example arxiv     # audit + archive a clean upload
just latex::bundle 01_example quantitative-finance
just latex::build 05_spatial_pricing jfec
just latex::bundle 05_spatial_pricing jfec
just latex::build 00_dissertation uct-thesis # assemble Papers 1, 5, and 3
just latex::check                       # format, lint, boundaries, smoke build
just latex::bundle-refresh              # regenerate the vendored TeX bundle
```

There is no LaTeX provisioning step. `src/latex/bundle/*.zip` is the TeX asset
set — the subset of the pinned upstream bundle these manuscripts resolve, ~7 MB,
tracked in git-LFS — so every compile is offline on a fresh clone. Run
`bundle-refresh` (network required, DVC figures present) only after adding a
`\usepackage` or a font size the subset lacks, then commit the regenerated zip.

`just latex::build` is the only recipe that produces a PDF — interactively, in
the gates, and in the DVC `render` stage alike. Each target's artifacts land in
`projects/<paper>/build/<target>/`: `<target>.pdf` and the venue archive are
tracked in git (LFS) so a checkout alone can read the paper and reproduce an
upload. Quantitative Finance uses `<target>.zip`; arXiv and JFEC use
`<target>.tar.gz`. The `.log`, `.blg`, and intermediates beside them are ignored. Run
`just latex::fmt` explicitly to rewrite sources; gates only check. Compilation
also rejects unresolved citations, references, and multiply-defined labels.

`projects/00_dissertation` is the one aggregate project. Its
`dissertation.toml` fixes essay order; `tools/assemble-dissertation` copies only
each paper's public `body.tex`/`appendix.tex` interface and owned source assets
into a disposable workspace, namespaces labels and generated-value keys, and
then compiles one PDF. Canonical paper sources are neither moved nor committed
twice. The thesis has one citation-selected bibliography and collects the three
paper appendices after the synthesis chapter. It deliberately has no journal
source bundle.

## Layout

| Path | Owns |
|---|---|
| `templates/pp-manuscript.sty` | Stable semantic commands and environments |
| `projects/<paper>/Tectonic.toml` | Pinned bundle and named build targets |
| `projects/<paper>/src/targets/` | Document class, packages, layout, front matter |
| `projects/<paper>/src/body.tex`, `appendix.tex` | Stable public paper composition interfaces |
| `projects/<paper>/src/chapters/` | Content-only manuscript sections |
| `projects/<paper>/src/generated/` | Pipeline-owned values and tables |
| `projects/<paper>/src/metadata.tex` | Title and author data |
| `projects/<paper>/src/refs.bib` | Generated BibTeX catalog |
| `projects/00_dissertation/dissertation.toml` | Aggregate paper identity and order |
| `tools/assemble-dissertation` | Disposable, collision-safe thesis assembly |
| project-local generated class/style files | Nix-materialized, pinned Taylor & Francis and OUP assets |
| `projects/*/src/oup-*` | Nix-materialized, pinned OUP class and author-date style |
| `tools/` | Static boundary enforcement |

Read [`AGENTS.md`](AGENTS.md) before editing manuscripts and
[`ARCHITECTURE.md`](ARCHITECTURE.md) before changing boundaries or dependencies.
