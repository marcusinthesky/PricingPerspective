---
description: Lean 4 proofs and publication claim verification for the four papers.
---

# Lean proofs

This Lake workspace contains the formal results supporting papers 1, 3, 5 and 6.
Run commands from the repository's devenv shell; the pinned compiler is Lean 4.35.0.

```bash
just lean::cache
just lean::publication-check
just lean::build
just lean::lint
just lean::sorry-test
just lean::dependency-audit
just lean::certify
just lean::claims-publication-check
```

The recipes locate the workspace automatically. A direct `lake build` also works
from `src/lean/`. See [the architecture](ARCHITECTURE.md) for dependency direction
and [the quality guide](tools/quality/README.md) for the lint and import checks.

## Packages

| Package | Purpose |
|---|---|
| `PricingPerspective` | Exposure geometry, transmission bounds, spatial pricing, operating games, response and valuation. |
| `EnergyStatistics` | Energy distance, mean embeddings and empirical V-statistics. |
| `WassersteinGeometry` | Probability-law transport, geodesics, finite gluing and multimarginal barycentres. |

The two foundations are independent of `PricingPerspective`. Public
[Statadic](https://github.com/marcusinthesky/statadic) supplies negative-type
geometry and Gaussian projection lemmas to EnergyStatistics. Its Lake package is
`proofs`, selected from `src/lean` at commit
`213a32399705af3463a3c1d8835409f284025ca1`; the modules use `StatadicFormal`.
This source revision is pinned independently of the Python runtime version.

Mathlib is pinned to `v4.35.0`, and LeanArchitect to
`9f3bb53916b9d6fa05b7777389e1844f82b1c9a6`. The dependency audit checks coordinated
source requirements, manifests and toolchains.

## Publication claims

Native `@[blueprint]` annotations beside the original declarations associate
manuscript labels with mathematical statements. The `Publication` library assembles
those modules and annotates the external mathlib polarization identity. The
[claim adapter](tools/claim-manifest/README.md) checks exact elaborated statements,
transitive axioms and the reviewed assumptions and disclosures before generating
publication status files.

There are 9 approved mappings for paper 1, 9 for paper 3, 18 for paper 5 and 27 for
paper 6. A mapping verifies its displayed formal statement under its hypotheses;
it does not certify empirical identification or the implementation of a numerical
method. Paper 6's numerical benchmark references retain their distinct conditional
scope.

`just lean::certify` builds current sources, checks the global integrity policy,
exports native metadata and regenerates manifests and disclosures.
`just lean::claims-publication-check` compares the committed projections and
manuscript labels without invoking Lean. The [Blueprint companion](blueprint/README.md)
provides linked proof explanations and declaration checks.

## Trust policy

Economic and statistical assumptions are explicit hypotheses. Custom axioms are
forbidden; the permitted trust base is `propext`, `Classical.choice` and `Quot.sound`.
The [compiled integrity gate](tools/sorry-audit/README.md) also inventories known
admissions and vacuous `True` contracts, including private helper declarations.
The retained proof closure has no reviewed admissions or vacuous contracts;
`sorry-allowlist.txt` is empty apart from the policy comments. New exceptions fail
the integrity gate.

## Structure

<!-- insitu:begin tree
id = "gittree"
depth = 1
directories_only = true
-->

```text
lean/
├── blueprint/
├── packages/
├── Publication/
├── scripts/
└── tools/
```

<!-- insitu:end -->
