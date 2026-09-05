---
type: report
title: Expected Returns and Large Language Models
description: Chen, Kelly & Xiu (2026), LLM embeddings of financial news to predict cross-sectional expected returns
author:
- family: Chen
  given: Leland Bybee
- family: Kelly
  given: Bryan T.
- family: Xiu
  given: Dacheng
issued:
  date-parts:
  - - 2026
URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687
archive: SSRN
archive_location: '4416687'
publisher: SSRN
number: '4416687'
---

# Expected Returns and Large Language Models

## Summary

Chen, Kelly, and Xiu use LLM embeddings from financial news to predict
cross-sectional expected returns, demonstrating that contextual text
representations contain pricing-relevant information beyond traditional
sentiment measures.

## Key findings

### F1 — LLM embeddings contain cross-sectional return predictability

- Claim: High-dimensional text embeddings from financial news predict
  cross-sectional expected returns, capturing information beyond traditional
  sentiment scores and firm characteristics.
- Evidence: The paper trains predictive models on LLM embeddings and reports
  out-of-sample return predictability relative to standard factor benchmarks.
- Scope: Embeddings are used as predictive features for individual-stock
  expected returns, not as firm-level distributions whose pairwise distances
  restrict covariance.

## Assessment

Cite as evidence that language-model representations carry pricing-relevant
information. The distinction from Paper 1 is that Chen, Kelly, and Xiu map
embeddings to expected-return point predictions, whereas our application
compares distributions of embeddings across firms and derives covariance
restrictions from their Wasserstein distance.
