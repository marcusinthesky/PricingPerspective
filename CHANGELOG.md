---
type: changelog
description: Per-PR record of changes to the four papers, numerical reproduction, formal proofs and project website.
---

# CHANGELOG

## [PR #28](https://github.com/marcusinthesky/PricingPerspective/pull/28): Reproducible four-paper release

### Added

- Added reproduction from published embeddings, adjusted returns, anonymous
  sparse TF–IDF inputs, co-mention events and numerical controls, with checksummed
  input validation and preserved experimental settings.
- Added comparisons of manuscript values at their displayed precision, exact
  counts, tables and figures against fixed reference artifacts. Numerical
  identity errors use the certificate's existing tolerance.

### Changed

- Updated the four arXiv manuscripts, citation metadata and
  companion website to reflect the retained analyses.
- Restricted manuscript builds and DVC publication outputs to the four arXiv
  targets, including their PDFs and source archives.
- Pinned the Nix/devenv environment and Python workspace. Statistical computation
  uses public Statadic; documentation projections use public Insitu.
- Reduced Lean dependencies to the required proof closure. Statadic resolves
  from its pinned public GitHub revision; EnergyStatistics and
  WassersteinGeometry retain the bridges needed by the papers.
- Verified 1,660 Python tests and the Lean publication checks for 63 claims with
  no admissions. Full DVC reproduction and publication comparisons passed:
  922 numeric displays, two numerical identity bounds, 42 fragments and
  21 figures.

### Removed

- Removed scraping, encoder training and inference, article text and titles
  from the reproduction code and data handoff.
- Removed packages and experiments outside the four-paper reproduction.
  `.context` retains citation metadata under `reference` only.
- Removed the unused PaperReconstructions dependency, preserving its required
  type aliases in the owning proof module.

### Fixed

- Stabilized inactive-firm ordering in portfolio figures. The evolution heatmap
  compares exact colors by firm and year, preserving the numerical weights.
- Refined the spatial likelihood optimizer with its analytic score, resolving
  roundoff in paired bootstrap contrasts without changing the objective, bounds,
  seeds or resampling schedules.
- Counted barycentric peers using the solver's existing numerical-support
  threshold. Corrected the remaining-peer labels for AMD (94 to 30) and HON
  (75 to 53); weights, estimates, top-five peers and residual mass are unchanged.
- Made arXiv PDFs deterministic across preview and release builds, and
  preserved certified source archives during preview builds.
- Made Paper 6 tests independent of downloaded artifacts and terminal color
  formatting, so the standard CI run works from a clean checkout.

## [PR #1](https://github.com/marcusinthesky/PricingPerspective/pull/1): AnyIO dependency update

### Changed

- Proposed a Dependabot update from AnyIO 4.14.1 to 4.14.2 in
  `src/python/uv.lock`; this dependency proposal remains unmerged.
