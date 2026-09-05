# DuckDB CLI, pinned via the separate `nixpkgs-duckdb` input (see its
# comment in the flake's `inputs`) so it can track a newer duckdb than the
# frozen main `nixpkgs` pin without moving anything else in the shell.
{ nixpkgs-duckdb, system }:
(import nixpkgs-duckdb { inherit system; }).duckdb
