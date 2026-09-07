---
type: report
title: The Network of Firms Implied by the News
description: Schwenkler and Zheng (2020), news co-mention firm network, contagion
  predictability, and aggregate risk measurement
author:
- family: Schwenkler
  given: Gustavo
- family: Zheng
  given: Hannan
issued:
  date-parts:
  - - 2020
URL: https://hdl.handle.net/10419/244260
DOI: 10.2849/250952
container-title: ESRB Working Paper Series
number: '108'
publisher: European Systemic Risk Board
ISBN: 978-92-9472-129-7
---

# The network of firms implied by the news

## Summary

The paper extracts a firm-to-firm network from sentence-level co-mentions in
over 100,000 Reuters articles published between 2006 and 2013, then shows that
the resulting links carry contagion information that predicts firm-level
returns and credit downgrades, and that network connectivity measures predict
aggregate volatility, credit spreads, default rates, and output.

## Key findings

### F1 — News co-mention is used as a bivariate link signal, not a sentiment score

- Claim: the identifying assumption is that two firms sharing a business
  connection are mentioned in the same sentence, with link strength increasing
  in the frequency of such reports.
- Evidence: `paper.md:57`; the paper contrasts this explicitly with the
  univariate sentiment-extraction literature.
- Scope: the output is a discrete, externally supplied relation matrix over
  roughly 3,000 CRSP/Compustat firms and over 20,000 distinct links
  (`paper.md:59`).

### F2 — Reporting is selective, so the network is not an unbiased link sample

- Claim: the news is more likely to report a link when the *less popular*
  linked firm is in distress; link likelihood does not rise when the more
  popular firm is distressed.
- Evidence: `paper.md:61,63`; logit regressions with time, firm, or link fixed
  effects, robust to restricting to the 500 largest firms and to excluding the
  financial crisis.
- Scope: this is a statement about the measurement channel, and the authors
  frame it as extending Scherbina and Schlusche (2015) by showing the news is
  not an unbiased source of information about firm links.

### F3 — Connectivity measures predict aggregate outcomes

- Claim: shocks to average degree raise the VIX and credit spreads for up to
  three months; shocks to second-order interconnectivity raise the aggregate
  default rate and lower GDP growth for 12 or more months.
- Evidence: `paper.md:65,67`; monthly VAR with impulse response functions.
- Scope: aggregate time-series prediction, not a cross-sectional pricing
  restriction.

## Assessment

Relevant to this repository as the measurement precursor to `ge_news-implied_2023`,
which builds news co-mention links into a spatial factor model. The estimand
differs from ours in the same way Ge's does: this paper supplies a discrete
relation matrix exogenously and studies contagion and prediction, whereas the
present work derives a pairwise distance from distribution-valued firm
characteristics and restricts systematic covariance. F2 is the load-bearing
caution — a co-mention network is a selected sample of links, so it is evidence
about reporting behaviour as much as about economic linkage.

## Detailed analysis

The methodology is a discrete-relation construction: named-entity recognition
over article text, sentence-level co-mention as the link indicator, and link
weight increasing in report frequency. Nothing in the construction treats a
firm's article collection as a distribution over a characteristic space, so the
paper offers no comparison object for distributional shape. Its usefulness here
is bibliographic and cautionary rather than theoretical, and `ge_news-implied_2023`
covers the same claim inside an asset-pricing model.
