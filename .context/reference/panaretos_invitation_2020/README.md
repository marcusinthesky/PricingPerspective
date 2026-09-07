---
type: book
title: An Invitation to Statistics in Wasserstein Space
description: Panaretos & Zemel (2020), statistics in Wasserstein space (book)
author:
- family: Panaretos
  given: Victor M.
- family: Zemel
  given: Yoav
issued:
  date-parts:
  - - 2020
URL: https://link.springer.com/book/10.1007/978-3-030-38438-8
DOI: 10.1007/978-3-030-38438-8
ISBN: 978-3-030-38438-8
container-title: SpringerBriefs in Probability and Mathematical Statistics
publisher: Springer
---

# An Invitation to Statistics in Wasserstein Space

## Summary

The book develops optimal-transport geometry and statistical methods for
probability measures in Wasserstein space, including geodesics, tangent spaces,
and Fréchet means.

## Key findings

### F1 — Wasserstein space has a usable geodesic and tangent-space geometry

- Claim: McCann interpolation gives constant-speed geodesics, while tangent
  vectors are represented through an $L_2(\mu)$ closure of gradients.
- Evidence: Chapter 2 records both constructions in
  `chapters/02_wasserstein_space.md`.
- Scope: The tangent-space statement depends on the chapter's regularity and
  absolute-continuity qualifications; it is not a global linearization.

### F2 — Fréchet means extend averaging to probability measures

- Claim: A Fréchet mean minimizes a sum-of-squared-Wasserstein-distances
  functional, with existence and uniqueness requiring additional conditions.
- Evidence: Section 3.1 defines the functional and develops the relevant
  conditions in `chapters/03_frechet_means_w2.md`.
- Scope: The result concerns Wasserstein barycentres, not an arithmetic average
  of distribution parameters.

## Assessment

Use Chapters 2–3 when a project claim requires geometric structure or a
distribution-valued mean. Do not cite the book as support for a particular
asset-pricing model without an independent pricing argument.

## Detailed analysis

The repository packages Chapters 1–5 as separate Markdown files. Chapter 5
further gives an iterative construction based on pairwise transport problems;
Corollary 5.3.2 states Wasserstein convergence under its stated uniqueness and
regularity conditions.
