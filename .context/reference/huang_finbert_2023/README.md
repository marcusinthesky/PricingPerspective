---
type: article-journal
title: '{FinBERT}---A Large Language Model for Extracting Information from Financial Text'
description: Huang, Wang & Yang (2023), FinBERT for financial sentiment and ESG classification
author:
- family: Huang
  given: Allen H.
- family: Wang
  given: Hui
- family: Yang
  given: Yi
issued:
  date-parts:
  - - 2023
URL: https://doi.org/10.1111/1911-3846.12832
DOI: 10.1111/1911-3846.12832
container-title: Contemporary Accounting Research
volume: '40'
issue: '2'
page: 806-841
---

# FinBERT---A Large Language Model for Extracting Information from Financial Text

## Summary

Huang, Wang, and Yang develop FinBERT, a domain-specific BERT model
pre-trained on financial communication text, and demonstrate its value for
financial sentiment analysis and ESG classification tasks.

## Key findings

### F1 — Domain-specific pre-training improves financial text tasks

- Claim: A BERT model further pre-trained on financial communications
  outperforms general-purpose models on sentiment analysis and ESG
  classification from corporate disclosures and analyst reports.
- Evidence: The paper reports improved F1 scores on financial sentiment
  benchmarks relative to generic BERT and dictionary-based methods.
- Scope: The application maps individual documents to sentiment labels or
  topic classifications, not to distributional distances across firms.

## Assessment

Cite to establish that finance-specific language models exist and are
effective for extracting information from financial text. The contrast with
Paper 1 is that FinBERT produces first-order textual signals (per-document
sentiment or classification), whereas our application treats the entire
distribution of a firm's article representations as the object of interest.
