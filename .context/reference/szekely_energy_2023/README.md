---
type: book
title: The Energy of Data and Distance Correlation
description: Székely & Rizzo (2023), energy of data and distance correlation (book)
author:
- family: Székely
  given: Gábor J.
- family: Rizzo
  given: Maria L.
issued:
  date-parts:
  - - 2023
URL: https://www.routledge.com/The-Energy-of-Data-and-Distance-Correlation/Szekely-Rizzo/p/book/9781482242744
DOI: 10.1201/9780429157158
ISBN: 978-0-429-15715-8
container-title: Monographs on Statistics and Applied Probability
publisher: CRC Press
---

# The Energy of Data and Distance Correlation

## Summary

The book consolidates energy distance, distance covariance, and distance
correlation, together with their sample statistics, hypothesis tests, and
extensions beyond Euclidean observations.

## Key findings

### F1 — Energy distance characterizes equality of distributions

- Claim: For distributions with finite first moments, energy distance is
  nonnegative and equals zero exactly when the distributions are identical.
- Evidence: Chapter 4 states this property immediately after equation (4.1) in
  `chapters/04_introduction_to_energy_inference.md`.
- Scope: The statement uses independent random variables and the chapter's
  finite-first-moment assumptions.

### F2 — Metric-space extensions require negative-type structure

- Claim: Distance-based energy inference extends to metric spaces whose metric
  is of negative type, including Euclidean and Hilbert spaces.
- Evidence: Chapter 10 states the condition in
  `chapters/10_energy_in_metric_spaces_and_other_distances.md`.
- Scope: Equality characterization needs the stronger negative-type conditions
  discussed there; an arbitrary metric is insufficient.

### F3 — The two-sample U-statistic is unbiased

- Claim: The two-sample U-statistic estimates population energy distance without
  finite-sample bias.
- Evidence: Chapter 16 states this for equation (16.10) in
  `chapters/16_u_statistics_and_unbiased_dcov.md`.
- Scope: The samples must satisfy the independence and moment conditions used by
  the chapter.

## Assessment

This is a useful primary reference for energy-statistics definitions and
sampling results. Keep energy distance between distributions distinct from
distance correlation between coordinates of a joint random vector.

## Detailed analysis

All 23 chapters are packaged under `chapters/`, so future claims should cite a
chapter, numbered equation, theorem, or remark instead of treating the book as
one undifferentiated source. The former covariance-style identity in this note
was removed because it was not supported by an exact locator in the package.
