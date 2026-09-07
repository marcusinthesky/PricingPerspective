---
description: Reproducible LaTeX publishing workspace rendered end to end by a single pinned Tectonic engine.
---

# src/latex/

This workspace is the sole, canonical publishing source for the arXiv
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
just latex::build 05_spatial_pricing arxiv
just latex::bundle 05_spatial_pricing arxiv
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
`projects/<paper>/build/<target>/`: `<target>.pdf` and the `<target>.tar.gz`
upload archive are tracked in git (LFS) so a checkout alone can read the paper
and reproduce an upload. The `.log`, `.blg`, and intermediates beside them are
ignored. Run
`just latex::fmt` explicitly to rewrite sources; gates only check. Compilation
also rejects unresolved citations, references, and multiply-defined labels.

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
| `tools/` | Static boundary enforcement |

Read [`AGENTS.md`](AGENTS.md) before editing manuscripts and
[`ARCHITECTURE.md`](ARCHITECTURE.md) before changing boundaries or dependencies.
