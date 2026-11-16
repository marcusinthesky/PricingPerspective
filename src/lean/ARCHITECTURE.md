---
description: Dependency boundaries, pinned libraries and publication certification in the Lean workspace.
title: Lean workspace
---

# Lean workspace

The root Lake package requires three local packages and exposes `Main` as the
default build target. Package roots enumerate their modules so compilation and
linting cover the retained source tree. Definitions and lemmas are grouped by
mathematical concept; larger concepts use `Defs` and `Basic` leaves with stable
roll-up imports.

## Dependency direction

`EnergyStatistics` and `WassersteinGeometry` are foundations
and cannot import `PricingPerspective`. Its `Operating` domain is independent of
`Exposure`, `Spatial`, `Transmission` and `Connections`. Those three core domains
cannot import `Operating` or downstream `Connections`. The compiled import graph
check enforces these boundaries in addition to Lean's cycle check.

Public Statadic supplies `StatadicFormal.Geometry.NegativeType.Defs` and `Basic`.
The definition leaf contains the finite zero-sum negative-type predicate; the
basic leaf supplies the Hilbert-space result and Gaussian integration lemmas used
by EnergyStatistics' mean embedding. Strict and strong negative type retain their
separate finite-support and measure-separation contracts in EnergyStatistics.
The public dependency is selected through Lake package `proofs` from the Statadic
repository's `src/lean` directory. Its immutable source pin is independent of the
Python dependency pin; neither engine is copied into this repository.

The retained sources are the import closure of the approved publication claims
and the exposure energy embedding bridge. Package roots enumerate this closure. Exposure geometry defines its two finite
real-vector aliases directly.

## Toolchain and resolution

Every first-party Lake configuration pins mathlib `v4.35.0`; `lean-toolchain` and
the Nix toolchain manifest select the matching Lean compiler. The Nix derivation
provides the compiler without an imperative elan installation. Lake manifests
record the resolved revisions.

The explicit mathlib requirement remains last. Lake resolves shared transitive
revisions by requirement order. The dependency audit checks the public Statadic URL,
its `src/lean` subdirectory, all coordinated revisions and toolchain files.

## Integrity and publication

The integrity gate discovers every retained first-party module independently of
default targets, rejects kernel-disabling debug options before building, and
inspects compiled declarations and their transitive axioms. Admissions and
vacuous `True` contracts must match the explicit inventory, currently empty. Custom axioms cannot
be allowlisted. Reviewed exceptions disclose their scope without certifying them
as completed mathematics.

LeanArchitect annotations live beside original project statements and proofs.
The downstream `Publication` assembly adds the external mathlib annotation.
Native library facets export labels, informal statements and inferred proof
references. The claim adapter independently checks elaborated statement types,
trust bases and nonvacuity before joining those exports to publication ownership,
assumptions and disclosures. Content hashes make statement changes visible.

Manuscripts consume identifier-free status projections. The Blueprint companion
uses native declaration links and inferred dependencies. Cheap manuscript checks
compare labels and exact generated projections; full certification builds current
sources and rechecks every mapped statement. Numerical evidence records its own
conditional scope and does not become Lean-verified merely by referring to a
formal result.

## Quality checks

Lake, Batteries and Mathlib own lint rules. The orchestration builds before
inspecting compiled imports, checks complete package indexes, runs declaration
linters and applies Mathlib's text-style checks through a small path adapter.
Named lint exceptions preserve reviewed interfaces while exposing existing debt.
Warnings-as-errors and independent kernel replay remain stricter optional checks;
the maintained integrity policy enforces the empty exception inventory.
