---
title: Wasserstein Playground
emoji: 🎯
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
tags:
  - arxiv:2410.23447
short_description: Shape distributions; watch W2 geometry cap how returns co-move.
description: "Browser-native marimo notebook for W2 covariance ceilings and the s-APT barycentric operator."
---

## playground

A deliberately small marimo notebook — one file, three scientific dependencies —
that turns two results from the dissertation into three connected facets you can
drag a slider at. It is exported to WebAssembly and runs entirely in your browser
under Pyodide, so the Space is `sdk: static`: no server, no cold start, no quota.

Shape four firms' characteristic laws in the sidebar. Their quadratic Wasserstein
separation caps how much their returns can co-move (paper 1); the target-anchored
barycentric operator reconstructs the target from its peers and returns the
interaction row `W♭` (paper 5).

### What the three panels show

The outer panels share one visual hierarchy, so the same three roles read the same
way in the densities and in the returns:

| Role | Encoding |
|---|---|
| Target (the anchor) | thick, black, solid |
| `W♭` mixture — what the operator built from the peers | thinner, black, dashed |
| Source firms | thin, dotted, light and slightly transparent |

**Left — characteristic laws.** Each firm is a two-component skew-normal mixture,
plotted as its closed-form density with the `M` sampled "articles" underneath as
dotted rug ticks. Four knobs span every shape: `Δ=0, α=0` is symmetric unimodal,
`Δ=0, α≠0` is skewed, `Δ>σ` is bimodal, and the two combine — and the legend names
whichever shape the current knobs actually produce. The black dashed curve is the
law the barycentric operator built: the target reconstructed from its peers.
Ticking *Overlay sample KDE* adds each cloud's empirical density in dash-dot, grey
for the target so it never reads as `W♭`.

**Middle — pairwise geometry.** The heatmap is the *sharp* correlation ceiling
implied by each pairwise W₂ distance. Its grayscale scale is stretched over the
observed off-diagonal range rather than `[-1, 1]`: comonotone coupling of similar
one-dimensional laws is always strongly correlated, so a full-range scale would
render every cell the same shade. Firm labels use the same colors as their density
and return paths.

**Right — implied returns.** A return panel drawn at a chosen share of the
pairwise ceiling, with the `W♭`-weighted peer average dashed over the target.

The figures use a restrained paper-and-ink palette close to the portfolio site. Color
is reserved for firm identity in the plots; the pairwise value scale and every
surface stay neutral.

### Why the maths collapses to sorting

Both papers are stated for clouds in an embedding space; in one dimension the
optimal transport plan between equally sized empirical clouds is the sort, and
that makes each construction a couple of lines:

| Object | Paper | One-dimensional form |
|---|---|---|
| W₂ distance | 1, 5 | RMS gap between the two sorted clouds |
| Sharp covariance ceiling | 1 | Attained by the comonotone coupling, so `corrcoef` of the sorted clouds |
| Transport excess | 1 | The unused share of that ceiling |
| Aligned peer positions `y` | 5 | Column `m` of the sorted peer clouds |
| Interaction row `W♭` | 5 | Simplex least squares of the target on aligned peers |

Because sorting couples *all* firms to a common uniform, the pairwise ceilings are
attained simultaneously: the ceiling matrix is a genuine PSD correlation matrix,
not a pasting-together of unrelated pairwise bounds, and it can be factored
directly to draw the return panel.

### Reading the output honestly

The `W♭` coefficients are barycentric coordinates on *aligned* peer positions.
They are not probabilities of drawing a peer's article, not a tradable replicating
portfolio, and not a causal influence — the same caveats the paper attaches to the
estimated field. The reconstruction RMSE reported under the panels is the
objective value: how well the aligned convex span represents the target.

The Space is an illustration on synthetic one-dimensional laws. The papers'
empirical fields are built from language-model article embeddings.

### Commands

```bash
just python::notebooks::edit playground   # marimo edit, live in your browser
just python::notebooks::wasm playground   # rebuild dist/ from the notebook
just python::notebooks::lock playground   # refresh notebook.py.lock
just python::notebooks::check playground  # format-check, lint, typecheck
```

Like every notebook here this is a standalone PEP 723 unit, not a workspace
member: `notebook.py` carries its own `# /// script` block and `notebook.py.lock`
pins it, so `edit` and the exports all resolve the same isolated environment —
the mechanism `just python::typecheck-scripts` also uses to type check it.

### Deploying to a Space

`just python::notebooks::wasm playground` writes the whole Space payload to `dist/`:
`index.html`, the `assets/` bundle, and this card copied in as `README.md`, whose
front matter selects Hugging Face's free `sdk: static`. `dist/` is untracked build
output — upload its *contents* to the root of the Space, however you prefer:

```bash
hf upload marcusinthesky/PricingPerspective dist . --repo-type=space
```

`public/` alongside the notebook is copied into `dist/` by the export, so anything
the Space needs to serve as a plain file belongs there.

The Space is the only host. The portfolio site embeds it from
`https://marcusinthesky-pricingperspective.static.hf.space` as a third
interactive companion beside the video and the podcast, behind a click-to-load
button — booting CPython, numpy, scipy and matplotlib under WebAssembly takes
around 30 seconds and saturates a core, which is not a cost to impose on every
visitor to the home page. Nothing is copied into the site, so publishing here is
the only step: there is no second copy to drift.

### Citation

This Space accompanies:

> Marcus Gawronsky and Chun-Sung Huang. *Systematic Covariance Envelopes from
> Wasserstein Geometry: Evidence from Language-Model Representations.*

- Paper page: <https://huggingface.co/papers/2410.23447>
- Preprint: <https://arxiv.org/abs/2410.23447>

```bibtex
@misc{gawronsky2024systematic,
  title  = {Systematic Covariance Envelopes from Wasserstein Geometry:
            Evidence from Language-Model Representations},
  author = {Gawronsky, Marcus and Huang, Chun-Sung},
  year   = {2024},
  eprint = {2410.23447},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.ST},
  url    = {https://arxiv.org/abs/2410.23447}
}
```
