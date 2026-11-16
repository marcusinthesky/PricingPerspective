set shell := ["bash", "-uc"]

mod nix 'infra/nix/justfile'
mod python 'src/python/justfile'
mod pipeline 'src/python/apps/pipeline/justfile'
mod lean 'src/lean/justfile'
mod latex 'src/latex/justfile'
mod site 'src/typescript/justfile'
mod typescript 'src/typescript/justfile'
mod references 'src/python/scripts/references/justfile'
mod insitu 'src/python/apps/insitu-repository/lifecycle.justfile'

default:
    @just --list

# Install the locked analysis and website environments.
setup: python::setup site::setup

hooks:
    prek install

lint: python::lint nix::lint lean::lint

typecheck: python::typecheck site::typecheck

test: python::test-fresh

semflow-check:
    uv run --locked --project src/python semflow check

# Recompute every retained paper stage from the public input boundary.
build: semflow-check
    dvc repro

# Verify the sanitized inputs and published values after reproduction.
reproduction-check:
    uv run --locked --script src/python/scripts/replication_release/script.py

check:
    prek run --all-files
