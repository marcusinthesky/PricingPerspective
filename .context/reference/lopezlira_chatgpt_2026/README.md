---
type: article-journal
title: Can {ChatGPT} Forecast Stock Price Movements? Return Predictability and Large Language Models
description: Lopez-Lira & Tang (2026), GPT-4 headline assessments for return predictability
author:
- family: Lopez-Lira
  given: Alejandro
- family: Tang
  given: Yuehua
issued:
  date-parts:
  - - 2026
URL: https://doi.org/10.1016/j.jfineco.2026.103990
DOI: 10.1016/j.jfineco.2026.103990
container-title: Journal of Financial Economics
volume: '168'
page: '103990'
---

# Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models

## Summary

Lopez-Lira and Tang use GPT-4 assessments of news headlines to study
immediate price responses and subsequent return predictability, finding that
LLM-based sentiment scores predict next-day returns.

## Key findings

### F1 — LLM headline assessments predict short-horizon returns

- Claim: GPT-4 sentiment classifications of individual news headlines
  predict next-day stock returns beyond traditional sentiment dictionaries.
- Evidence: The paper documents significant return predictability from
  positive/negative/neutral headline scores across a broad stock universe.
- Scope: Each headline is mapped to a scalar sentiment assessment; the
  application does not compare distributions of representations across firms.

## Assessment

Cite as a prominent example of LLMs applied to return prediction through
first-order textual signals. The contrast with Paper 1 is that Lopez-Lira
and Tang use per-headline point assessments, whereas our application treats
the full distribution of firm-level article embeddings as the characteristic
object and asks what its geometry implies for systematic covariance.
