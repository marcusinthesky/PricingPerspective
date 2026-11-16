# Reproducing the four papers

The manuscript projects are `01_continuous_bounds`, `03_distance_implied_mpt`,
`05_spatial_pricing`, and `06_information_motion`. Each has one `arxiv` target.
LaTeX is their canonical source; Python produces empirical values, tables and
figures, and Lean produces the formal-verification disclosures.

Run from the repository root:

```bash
just latex::check
just latex::bundle 01_continuous_bounds arxiv
just latex::bundle 03_distance_implied_mpt arxiv
just latex::bundle 05_spatial_pricing arxiv
just latex::bundle 06_information_motion arxiv
```

`just latex::build <paper> arxiv 1` compiles offline for inspection, preserving
an existing certified source archive. `just latex::watch <paper> arxiv` rebuilds
when authored inputs change. `just latex::preview <figure.pgf>` renders a single
pipeline figure. `just latex::fmt` rewrites authored formatting;
`just latex::check` checks formatting and compiles all four papers.

The pinned Tectonic engine reads the versioned bundle in `bundle/`. Builds do
not fetch TeX packages. Each project's `build/arxiv/` publishes `arxiv.pdf` and
`arxiv.tar.gz`. The source archive is audited, unpacked and compiled again
before publication. Preview and release use the same deterministic `main.tex`
compile, so unchanged inputs produce the same PDF bytes.

Generated values and images are pipeline-owned. Reproduce them with DVC rather
than editing their files. Paper 1's reviewed interpretation snapshot binds its
neighbour exhibit to the displayed numbers and selections; the interpretation
checker reports substantive changes and permits only its documented small
floating-point tolerance. Changes to scientific interpretation require an
explicit update of the reviewed snapshot and its manuscript acknowledgment.

Read [AGENTS.md](AGENTS.md) for manuscript boundaries and
[ARCHITECTURE.md](ARCHITECTURE.md) for the build contract.
