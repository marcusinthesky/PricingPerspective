---
type: article-journal
title: From Word Embeddings to Document Distances
description: Kusner et al. (2015), Word Mover's Distance as optimal transport between empirical word-embedding distributions
author:
- family: Kusner
  given: Matt J.
- family: Sun
  given: Yu
- family: Kolkin
  given: Nicholas I.
- family: Weinberger
  given: Kilian Q.
issued:
  date-parts:
  - - 2015
URL: https://proceedings.mlr.press/v37/kusnerb15.html
container-title: Proceedings of the 32nd International Conference on Machine Learning
collection-title: Proceedings of Machine Learning Research
volume: '37'
publisher: PMLR
page: 957-966
---

# From Word Embeddings to Document Distances

## Summary

Kusner et al. introduce Word Mover's Distance, which treats the words in two
documents as empirical distributions in an embedding space and measures the
minimum transport cost between them.

## Key findings

### F1 — Word Mover's Distance compares embedded word distributions by transport

- Claim: A document is represented as a weighted point cloud of embedded words,
  and the distance between documents is the minimum cumulative travel needed to
  match the two clouds.
- Evidence: `paper.md:108-128` defines the representation and identifies the
  optimization as an Earth Mover's Distance transportation problem.
- Scope: The paper studies word-level document comparison, not firm-level
  distributions of article embeddings or financial outcomes.

## Assessment

Use this paper as the NLP precedent for comparing text through distributions of
representations rather than a single pooled score. It does not support the
paper's characteristic-to-exposure map or covariance result.
