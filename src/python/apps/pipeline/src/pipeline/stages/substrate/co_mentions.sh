#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
    printf 'usage: %s CORPUS UNIVERSE OUTPUT_DIR START_DATE END_DATE MIN_YEARS\n' "$0" >&2
    exit 2
fi

corpus_path=$1
universe_path=$2
output_dir=$3
start_date=$4
end_date=$5
min_persistence_years=$6

mkdir -p "$output_dir"

duckdb <<SQL
CREATE OR REPLACE TEMP TABLE universe AS
SELECT UPPER(Symbol) AS firm
FROM read_csv_auto('${universe_path}');

CREATE OR REPLACE TEMP TABLE tagged AS
SELECT DISTINCT
    CAST(a.id AS VARCHAR) AS article_id,
    a.created_date,
    YEAR(a.created_date)::INTEGER AS publication_year,
    UPPER(TRIM(a.symbol)) AS source_symbol,
    a.primarytopic,
    UPPER(TRIM(SPLIT_PART(u.raw_firm, '|', 1))) AS firm
FROM read_parquet('${corpus_path}') AS a,
     UNNEST(a.related_symbols) AS u(raw_firm)
INNER JOIN universe AS v
    ON UPPER(TRIM(SPLIT_PART(u.raw_firm, '|', 1))) = v.firm
WHERE a.created_date BETWEEN CAST('${start_date}' AS DATE)
                         AND CAST('${end_date}' AS DATE);

CREATE OR REPLACE TEMP TABLE article_firm_counts AS
SELECT article_id, COUNT(DISTINCT firm) AS n_firms
FROM tagged
GROUP BY article_id;

CREATE OR REPLACE TEMP TABLE events AS
SELECT
    left_firm.article_id,
    left_firm.created_date,
    left_firm.publication_year,
    left_firm.firm AS firm_i,
    right_firm.firm AS firm_j,
    left_firm.source_symbol,
    left_firm.primarytopic
FROM tagged AS left_firm
INNER JOIN tagged AS right_firm
    ON left_firm.article_id = right_firm.article_id
   AND left_firm.firm < right_firm.firm
INNER JOIN article_firm_counts AS counts
    ON left_firm.article_id = counts.article_id
   AND counts.n_firms = 2
ORDER BY left_firm.created_date, left_firm.article_id,
         left_firm.firm, right_firm.firm;

CREATE OR REPLACE TEMP TABLE pair_year AS
SELECT
    firm_i,
    firm_j,
    publication_year,
    COUNT(*)::BIGINT AS co_mention_count
FROM events
GROUP BY firm_i, firm_j, publication_year
ORDER BY firm_i, firm_j, publication_year;

CREATE OR REPLACE TEMP TABLE adjacency AS
SELECT
    firm_i,
    firm_j,
    SUM(co_mention_count)::BIGINT AS weight,
    COUNT(*)::INTEGER AS n_years,
    MIN(publication_year)::INTEGER AS first_year,
    MAX(publication_year)::INTEGER AS last_year
FROM pair_year
GROUP BY firm_i, firm_j
HAVING COUNT(*) >= ${min_persistence_years}
ORDER BY firm_i, firm_j;

COPY events TO '${output_dir}/events.parquet'
    (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE);
COPY pair_year TO '${output_dir}/pair_year.parquet'
    (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE);
COPY adjacency TO '${output_dir}/adjacency.parquet'
    (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE);
COPY (
    SELECT
        'news_co_mentions' AS artifact_type,
        '${start_date}' AS start_date,
        '${end_date}' AS end_date,
        ${min_persistence_years}::INTEGER AS min_persistence_years,
        (SELECT COUNT(*) FROM universe)::INTEGER AS n_nodes,
        (SELECT COUNT(*) FROM events)::BIGINT AS exact_two_articles,
        (SELECT COUNT(DISTINCT firm_i || '|' || firm_j)
         FROM events)::INTEGER AS observed_pairs,
        (SELECT COALESCE(SUM(co_mention_count), 0)
         FROM pair_year)::BIGINT AS pair_mentions,
        (SELECT COUNT(*) FROM adjacency)::INTEGER AS persistent_pairs
) TO '${output_dir}/summary.json'
    (FORMAT JSON, ARRAY FALSE, OVERWRITE_OR_IGNORE TRUE);
SQL
