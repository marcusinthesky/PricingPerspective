# Manuscript build architecture

Python and Lean produce manuscript artifacts. Each paper consumes only its own
chapters, generated values, figures, references and metadata. The shared
`templates/pp-manuscript.sty` supplies semantic commands and theorem environments;
target adapters own layout, packages, front matter and bibliography style.

## Build and release

Tectonic is the only PDF compiler. Every project points to the versioned
`tlextras-2022.0r0-pricing-perspective.zip` bundle, which fixes the TeX asset bytes.
The environment pins the compiler and materializes the required Taylor & Francis
and Elsevier classes. Compiles do not install TeX packages or require network
access. `bundle-refresh` is the separate, network-dependent maintenance command
for updating the resolved bundle after a dependency change.

`just latex::build` compiles one target and checks its log. `just latex::bundle`
first certifies Lean claims and publication bindings, stages manuscript inputs,
audits the compiler-recorded dependency closure, writes a source manifest, and
unpacks and recompiles the archive. Publication PDFs and archives are tracked
artifacts; intermediate logs and staging directories are ignored. PDF byte
identity is not a numerical reproduction criterion because compiler metadata may
vary. The source manifest fixes every staged input.

The publication matrix consists of arXiv for all four papers, Quantitative
Finance for Paper 1, European Journal of Finance for Paper 3, Global Finance
Journal for Paper 5, and Management Science and WFBS for Paper 6. Management
Science builds a main/companion pair with generated cross-document label and
counter handoffs. Its source archive includes author metadata for reproduction;
the main and companion PDFs are checked for anonymous front matter.

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

## Figures and validation

Figures remain pipeline-produced PGF assets. `\ppfigure{fraction}{name}` imports
an image at its authored size; the fraction records its intended linewidth share
for print-size validation. Raster sidecars referenced by PGF are part of the
figure closure. `tools/stage-images` stages reachable figures and rejects missing
or escaping paths. The bundle audit also rejects unreferenced staged images,
external absolute inputs, unresolved citations, missing glyphs and shell escape.

`just latex::check` runs formatting and boundary checks and compiles the real
paper targets offline. `just latex::harness-test` exercises hostile source
closures, numeric contracts, image staging and deterministic archive manifests.
The effective TeX assets derive from TeX Live 2022; publisher recompilation may
use a newer release, so target adapters retain conservative package selections.
