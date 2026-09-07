## Introduction

Read all adjacent and parent `README.md` files, these files offer progressive disclosure of the project's structure and purpose and document key concepts, conventions and dependencies — which must be followed.

Read `ARCHITECTURE.md` files for the "why" behind architectural and tool decisions — comparison tables, trade-offs, and alternatives considered. These are the canonical source of truth for design decisions.

## Pipeline

The DVC stage graph below (foreach/matrix families collapsed) is a static snapshot of `dvc.yaml`.

![DVC pipeline stage graph](pipeline.png)

## Environment

Use `nix develop` when running terminal commands to ensure the correct environment is set up.

## Commands

Run `just --list` to see a list of available commands, at different levels of abstraction across the codebase.

## Code Quality

This repository imposes strict quality gates and automated code-generation using `prek` `.pre-commit-config.yaml` hooks, which must be installed and configured correctly, and passed before any code can be committed.

## Conflicts of Interest

The authors bear no conflicts of interest to disclose.

## Funding

There is no fundings needed to disclose.

## Use of Generative Models

Generative Models have been used in later revisions of this work to aid in editorial review, formatting of final manuscripts and aid in proof formalization.
