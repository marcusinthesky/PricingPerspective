# PaperReconstructions — TODO

Implementation queue for the `PaperReconstructions` package. The build fix is
specified in `.context/plan/.archive/2026-07-07_lean-build-repair/`; the strengthening items
below support the parked "machine-checked APT" paper.
This file tracks package-local state. Verified against the working tree 2026-07-07.

## Current state

> **Build status.** **Green** (2026-07-07, 8,599 jobs, 0 errors). The
> `OneFactorModel` migration is complete.

| Module | Status | Notes |
|--------|--------|-------|
| `Ross1976/Model.lean` | 1 `sorry` | `ftap_one_period_vector_iff` forward direction (NoArb ⟹ ∃ EMM) at :286; backward direction proven (:287-295) |
| `Ross1976/Ross1976.lean` | sorry-free | `ross_apt_representation` (:65), `ross_expected_return_equals_mu_eps` (:110) — hypothesis-conditional on `NoArbitrage` |
| `Ross1976/Moments.lean` | complete | `var`/`cov` + 7 proven covariance-algebra lemmas |
| `ChamberlainRothschild1983.lean` | sorry-free | No spectral hypothesis; `ApproximateFactorStructure` (:59) is dead code |
| `Ingersoll1984.lean` | sorry-free | `boundedFactorLoadings := True` placeholder (:99-100); no residual-correlation content despite theorem names |
| `Reisman1992.lean` | sorry-free | Migrated to bundled `OneFactorModel`; `hvar_g : var g ≠ 0` hypothesis required by mathlib v4.31.0 `field_simp` |
| `Shanken1992.lean` | sorry-free | Migrated to bundled `OneFactorModel`; `hvar_g : var g ≠ 0` hypothesis added; `shanken_apt_bounds_tautology` concludes `True` (:130) |
| `MiddletonSatchell2000.lean` | 1 `sorry` | Prop 2 at :227; `NoAsymptoticArbitrage := True` (:127-129), `h_bound_e : True` (:222); Prop 3 (:246) proves a positive conclusion where the source proposition is negative — restate or rename |
| `Connor1984.lean` | sorry-free | `connor_capm_equivalence` migrated to bundled `OneFactorModel`; delegates to `reisman_reference_beta`; elaborates (unused-variable warnings only) |
| `Dinunno.lean` | sorry-free | Finite-space only (`[Fintype Ω]`, :72); `doob_meyer_decomposition` (:326) is one `condExp_of_stronglyMeasurable` application — do not oversell |

**Shared-proof disclosure:** the sorry-free no-arbitrage theorems (Ross, CR ×2,
Ingersoll ×2, MS Props 1/3) are one argument — residual means equal ⟹ expected
returns equal ⟹ witness `(r_f = μ_ε, α = 0)`. Paper-specific bounds (CR's δ²,
Ingersoll's `v′Σ⁻¹v`, MS's rates) are not formalized.

## Queue

- [X] ~~**Migrate `Shanken1992.lean` + `Connor1984.lean`** to bundled `OneFactorModel`~~
      Both done 2026-07-07 (build green).
- [X] ~~**Prove `cov_ne_zero_imp_var_ne_zero`**~~ **DONE 2026-07-07**: resolved via
      explicit `hvar_g : var g ≠ 0` hypothesis; sound L² version proven in
      `Ross1976/Moments.lean:163-195`.
- [X] ~~`Shanken1992.lean`: migrate to bundled `OneFactorModel`~~ **DONE 2026-07-07** (build-repair plan).
- [ ] `MiddletonSatchell2000.lean`: complete Proposition 2 (`k > m`, sorry at :227);
      replace `NoAsymptoticArbitrage := True` with a variance-vanishing-sequence
      definition; replace `h_bound_e : True` with `∃ σ, E[e_T²] ≤ σ²`.
- [ ] `MiddletonSatchell2000.lean`: restate or rename `middleton_satchell_fewer_proxies`
      (:246) so it does not claim Prop 3's negative content while proving the positive
      relation.
- [ ] `Shanken1992.lean`: replace `shanken_apt_bounds_tautology` (`True`/`trivial`)
      with a real statement of the finite-economy tautology.
- [ ] `Ingersoll1984.lean`: `boundedFactorLoadings := ∃ M, ∀ i j, |B i j| ≤ M`.
- [ ] `Ross1976/Model.lean`: FTAP forward direction for finite `Ω` (softplus
      construction — feasible in finite dimensions; would let
      `connor_unified_beta_pricing`'s SDF hypothesis be constructed).
- [ ] Realize or remove decorative hypotheses (`h_linindep`, `h_diag_D`, `h_rank_D`,
      `h_no_arb_asymp`, true-factor-side hypotheses).

## Conventions

- Canonical factor-model definitions live in `Ross1976/` (Types, Model, Moments).
- Do not depend on `PricingPerspective` — the dependency points the other way.
- Economic assumptions are theorem hypotheses, never `axiom` declarations.
- Every public `def`/`theorem` cites the source paper in its docstring.
