# data/raw — raw corpus provenance

## nasdaq.jsonlines

**Origin:** NASDAQ News API scrape via the Scrapy spider at
`src/python/packages/crawler/src/crawler/spiders/nasdaq.py` plus body enrichment.
Endpoint: `api.nasdaq.com/api/news/topic/articlebysymbol`.

**Collection window:** 2022-12 to 2023-01 (file mtime 2023-01-03; per-record
`date_accessed` field). The governed normalization spans October 2009 through
January 2023 and retains 214,480 unique records for the current 100-firm roster.

**Recovery:** Thought lost after the 2023 rewrite. Recovered 2026-07-08 from
`~/Downloads/nasdaq.jsonlines` (origin: original scrape machine).

**Statistics:**

| Property | Value |
|---|---|
| Byte size | 1,815,644,085 |
| Line count | 397,041 |
| MD5 | `a30d7fe43928f7d99fee118a5cf66b77` |
| DVC pointer | `data/raw/nasdaq.jsonlines.dvc` |
| Governed normalized roster | 100/100 configured tickers; minimum 449 deduplicated records |
| Body coverage | >200 chars in ~95% of records (p50 ~3,279 chars) |

**Immutability rule:** This file is NEVER modified by any pipeline stage. All
cleaning, deduplication, and date parsing happen downstream in `normalize_corpus`.
The raw file is byte-identical to the original scrape.

## HUMAN ACTIONS REQUIRED

Run these commands in order **before** any `dvc repro` that overwrites existing outputs:

```sh
# R6 safety snapshot BEFORE any dvc repro
git tag pre-recovery-2026-07-08

# Push DVC-tracked raw file to remote
dvc push data/raw/nasdaq.jsonlines.dvc
```

After staging:

```sh
# Track the DVC pointer and provenance README in git
git add data/raw/nasdaq.jsonlines.dvc data/raw/README.md .gitignore
```
