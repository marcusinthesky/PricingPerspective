# Manuscript build architecture

Python and Lean produce manuscript artifacts. Each paper consumes only its own
chapters, generated values, figures, references and metadata. The shared
`templates/pp-manuscript.sty` supplies semantic commands and theorem environments;
each `targets/arxiv.tex` owns layout, packages, front matter and bibliography style.

## Build and release

Tectonic is the only PDF compiler. Every project points to the versioned
`tlextras-2022.0r0-pricing-perspective.zip` bundle, which fixes the TeX asset bytes.
Compiles do not install TeX packages or require network access. `bundle-refresh`
is the separate, network-dependent maintenance command for updating the resolved
bundle after a dependency change.

`just latex::build` stages one paper in a temporary directory and checks its
compile log and recorded source closure. `just latex::bundle` first certifies
Lean claims and publication bindings, audits the staged sources, writes a
manifest, and unpacks and recompiles the archive. Both routes compile the same
`main.tex` with Tectonic's deterministic mode; preview builds preserve certified
archives and manifests. The four arXiv PDFs and source archives are tracked
artifacts. Intermediate logs and staging directories are ignored. The source
manifest records every staged input and the pinned TeX bundle identity.

## Generated scientific content

`generated/` is owned by pipeline renderers. Values are declared once and consumed
through `\ppvalue`, `\ppnum` or `\ppnume`; missing values retain their explicit
semantics. Bibliographies are citation-selected projections from
`.context/reference/`, checked by `just references::bibliography-check`.
Lean claim mappings own verification status and disclosure scope; manuscripts
consume the generated identifier-free projection.

Paper 1's interpretation checker compares the generated neighbour-outcome
contract with an authored reviewed snapshot. The snapshot stays fixed across
accepted roundoff, preventing successive builds from moving the baseline. The
neighbour-outcome protocol permits eight binary64 ULPs in raw nonnegative W2
values while displayed values, ranks, identities, methods, samples and input
hashes must match. Other protocols and unknown fields remain strict.

## Figure transport

Figures remain pipeline-produced PGF assets. `\ppfigure{fraction}{name}` imports
an image at its authored size; the fraction records its intended linewidth share
for print-size validation. Raster sidecars referenced by PGF are part of the
figure closure. `tools/stage-images` stages reachable figures and rejects missing
or escaping paths. The bundle audit also rejects unreferenced staged images,
external absolute inputs, unresolved citations, missing glyphs and shell escape.

`just latex::check` runs formatting and boundary checks and compiles the four
papers offline. `just latex::harness-test` exercises hostile source closures,
numeric contracts, image staging and deterministic PDFs, archives and manifests.
The effective TeX assets derive from the pinned TeX Live 2022 bundle.
