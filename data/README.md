# Reproduction data

DVC tracks the text-free numerical inputs and generated analyses for the four
papers. Use `dvc pull` to retrieve the fixed inputs from the configured anonymous
remote, then `dvc repro` to compute the retained analyses and manuscripts.

Document vectors retain stable identifiers, dates, firm labels and numeric
metadata. The publication input manifest checks their schemas, row counts and
SHA-256 hashes. Article text, titles and scraping implementations are excluded.

The frozen `fixed_inputs` stage groups the released inputs and their checksums.
`reference/published_values.json` and `reference/published_results.json` are the
reviewed baselines for manuscript values, exact counts and figures used by
`just reproduction-check`. Historical Paper 5 and Paper 6 numerical benchmarks
remain fixed inputs identified by their DVC hashes.
