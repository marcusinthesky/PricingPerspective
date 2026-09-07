---
type: article-journal
title: Barycenters in the Wasserstein Space
description: Agueh & Carlier (2011), quadratic Wasserstein barycentres and their multimarginal formulation
author:
- family: Agueh
  given: Martial
- family: Carlier
  given: Guillaume
issued:
  date-parts:
  - - 2011
URL: https://doi.org/10.1137/100805741
DOI: 10.1137/100805741
container-title: SIAM Journal on Mathematical Analysis
volume: '43'
issue: '2'
publisher: Society for Industrial and Applied Mathematics
page: 904-924
tags:
- theory
---

# Barycenters in the Wasserstein Space

## Summary

The paper defines Wasserstein barycentres for finitely many probability measures, studies
existence, uniqueness, characterization, and regularity, and relates the quadratic problem
to multimarginal optimal transport.

## Key findings

### F1 — The quadratic barycentre problem has a multimarginal formulation

- Claim: the weighted quadratic Wasserstein barycentre problem is related to a
  multimarginal optimal-transport problem whose pointwise cost is minimized at the weighted
  Euclidean barycentre.
- Evidence: Section 4 introduces the multimarginal problem in `paper.md:769-816`, and
  Proposition 4.2 states the barycentre--multimarginal connection at `paper.md:913-1001`.
- Scope: the packaged PDF is the authors' 2010 pre-publication version; Proposition 4.2 uses
  the Euclidean regularity assumptions stated there.

### F2 — Existence and regularity are stronger questions than equality of values

- Claim: existence, uniqueness, characterization, and regularity of barycentres are treated
  separately from the multimarginal representation.
- Evidence: the paper's roadmap separates these questions by section in `paper.md:60-68`.
- Scope: citing the multimarginal identity does not by itself license an optimizer-existence
  or uniqueness claim outside the paper's assumptions.

## Assessment

This is the primary antecedent for Paper 5's displayed barycentre--multimarginal identity.
The repository's Lean theorem deliberately proves only equality of the two infimum values
in a separable Hilbert space, without importing the optimizer or regularity conclusions.

## Detailed analysis

The source is especially useful for separating the standard mathematical identity from the
paper-specific application: Paper 5 uses the identity to express characteristic and
exposure dispersion through a common barycentre objective, while its carrier and spatial
attenuation inequalities are new downstream steps.
