---
type: article-journal
title: Empirical Process Results for Exchangeable Arrays
description: Davezies, D'Haultfœuille & Guyonvarch (2021), empirical-process and bootstrap results for exchangeable arrays
author:
- family: Davezies
  given: Laurent
- family: D'Haultfœuille
  given: Xavier
- family: Guyonvarch
  given: Yannick
issued:
  date-parts:
  - - 2021
URL: https://doi.org/10.1214/20-AOS1981
DOI: 10.1214/20-AOS1981
container-title: The Annals of Statistics
volume: '49'
issue: '2'
publisher: Institute of Mathematical Statistics
page: 845-862
---

# Exchangeable-Array Bootstrap

## Summary

The paper develops laws of large numbers, central limit theorems, and
bootstrap-process convergence results for dissociated exchangeable arrays,
including dyadic observations that can depend when they share a sampled unit.

## Key findings

### F1 — Node resampling induces tuple-product bootstrap weights

- Claim: Sampling nodes with replacement assigns each node a multinomial
  multiplicity, and a sampled tuple receives the product of its endpoint
  multiplicities.
- Evidence: Section 2.3 gives the resampling construction.
- Scope: For a dyad $(i,j)$, the resulting tuple weight is $w_iw_j$.

### F2 — Bootstrap convergence is conditional and asymptotic

- Claim: The bootstrap-process result is asymptotic rather than an unconditional
  finite-sample guarantee.
- Evidence: Theorem 2.2 states the result used here.
- Scope: It depends on the paper's exchangeability, dissociation, moment, and
  entropy conditions and supplies no causal interpretation.

## Assessment

The node-level resampling construction is directly useful for exchangeable
dyadic arrays, provided its asymptotic and conditional scope is retained.

## Detailed analysis

The packaged PDF is the 18-page published *Annals* version, volume 49, issue 2,
pages 845–862, DOI 10.1214/20-AOS1981. Because a local PDF fallback dropped
display mathematics, `paper.md` was regenerated from the authors' matching May
2020 v4 LaTeX source rather than retaining that broken conversion.
