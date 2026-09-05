---
description: "Why the TypeScript plane is a Bun workspace orchestrated by Turbo."
title: TypeScript workspace
---

## Why a nested workspace?

The TypeScript plane is a bounded ecosystem inside the polyglot repository. Its
runtime, package graph, lockfile, and build cache are independent from Python,
LaTeX, and Lean, while the outer repository still provides shared Nix and just
entrypoints. This keeps cross-language CI explicit and prevents JavaScript
dependencies from becoming ambient repository dependencies.

## Why Bun plus Turbo?

Bun is the package-manager authority because the incoming application already
uses Bun scripts and exact dependency pins. Turbo is layered on top of Bun's
workspace graph: Bun resolves and links packages, while Turbo schedules,
hashes, and caches package scripts. This is the standard division of
responsibility for an existing multi-package repository.

The root is metadata-only apart from the pinned `turbo` development dependency.
Applications and libraries own their runtime and development dependencies in
their own manifests. The `apps/*` and `packages/*` patterns make the next
project discoverable without changing the orchestration contract.

## Task boundaries

`turbo.json` is the single task contract. `build` produces Next and static
export artifacts; `build-storybook` produces Storybook output; verification and
audit tasks depend on a build; and `dev`/Lighthouse tasks are not cached. The
`NEXT_PUBLIC_SITE_URL` value is declared as a task environment input so a
deployment-origin change cannot reuse a stale generated site.

## Nix boundary

The repository's Nix shell supplies Bun itself. Because the locked nixpkgs
attribute currently trails the workspace's Bun 1.4.0 contract, the shell uses a
small platform-pinned derivation in `infra/nix/pkgs/bun.nix`. CI uses the same
version through `oven-sh/setup-bun`.
