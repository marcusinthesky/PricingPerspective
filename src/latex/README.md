# Reproducing the four papers

The four manuscript projects are `01_continuous_bounds`,
`03_distance_implied_mpt`, `05_spatial_pricing`, and `06_information_motion`.
LaTeX is their canonical source. Python produces empirical values, tables and
figures; Lean produces the formal-verification disclosures.

Run from the repository root:

```bash
just latex::check
just latex::bundle 01_continuous_bounds arxiv
just latex::bundle 01_continuous_bounds quantitative-finance
just latex::bundle 03_distance_implied_mpt ejf
just latex::bundle 05_spatial_pricing gfj
just latex::bundle 06_information_motion management-science
just latex::build 06_information_motion wfbs
```

Each project also has an `arxiv` target. The Management Science helper builds
both the manuscript and electronic companion. `just latex::build <paper>
<target> 1` compiles offline. `just latex::preview <figure.pgf>` renders a single
pipeline figure for inspection. `just latex::fmt` rewrites authored formatting;
`just latex::check` only checks it and compiles the publication matrix.

The pinned Tectonic engine reads the versioned bundle in `bundle/`. Publisher
classes and bibliography styles are materialized from pinned environment assets
by the build recipes. Build PDFs and source archives are published under each
project's `build/<target>/` directory. A bundle is audited, unpacked and compiled
again before publication. No submission correspondence belongs in these archives.

Generated values and images are pipeline-owned. Reproduce them with DVC rather
than editing their files. Paper 1's reviewed interpretation snapshot binds its
neighbour exhibit to the displayed numbers and selections; the interpretation
checker reports substantive changes and permits only its documented small
floating-point tolerance. Changes to scientific interpretation require an
explicit update of the reviewed snapshot and its manuscript acknowledgment.

Read [AGENTS.md](AGENTS.md) for manuscript boundaries and
[ARCHITECTURE.md](ARCHITECTURE.md) for the build contract.
