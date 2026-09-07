import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import PricingPerspective.Transmission.Floor

set_option linter.style.longLine false

namespace PricingPerspective.ContinuousAPT

/-! ### Energy kernel embedding structure -/

/--
**Energy Kernel Embedding.**

A structure encoding the assumption that the squared energy distance between
risk characteristic distributions decomposes exactly as

  `D_E²(β_i, β_j) = sys_var β_i + sys_var β_j − 2 * sys_cov β_i β_j`

where:

* `sys_var β` = systematic variance `σ²(β)`
* `sys_cov β_i β_j` = systematic covariance `Cov_sys(β_i, β_j)`
* `energy_sq β_i β_j` = squared energy distance `D_E²(β_i, β_j)`

The `identity` field is the single axiom of the model: it asserts that the
energy kernel generates the covariance inner product. No additional assumptions
are required for the theorems proved in this module; all further hypotheses
are passed as explicit arguments to each theorem.

#### Axioms

| Axiom | Content |
|-------|---------|
| `identity` | `energy_sq β_i β_j = sys_var β_i + sys_var β_j - 2 * sys_cov β_i β_j` |
-/
structure EnergyKernelEmbedding E where
  sys_var : E → ℝ
  sys_cov : E → E → ℝ
  energy_sq : E → E → ℝ
  identity : ∀ (β_i β_j : E),
    energy_sq β_i β_j = sys_var β_i + sys_var β_j - 2 * sys_cov β_i β_j

/-!
# Energy Distance Bounds on Return Co-Covariance

This module provides the main theorems for the continuous-space asset pricing
formalization. We prove that the systematic covariance (and, when exposures are
nonzero, the systematic correlation) of two assets is bounded below by a function
of the energy distance between their risk characteristic distributions.

## Design philosophy

The results are stated as **theorems with explicit hypotheses**, not as
unchecked `axiom`s. The economic content lives entirely in two hypotheses that
the caller must supply for each pair of assets:

1. **Common technology**: the exposures are the image of the characteristic
   distributions under a common operator, `β_i = T C_i` and `β_j = T C_j`.
   This is a *definition* of `β_i, β_j`, so it is passed as the pair of values
   `β_i, β_j` together with the Lipschitz bound below rather than as an axiom.
2. **Lipschitz information transmission**: the exposure distance is controlled
   by the energy distance, `‖β_i - β_j‖ ≤ L * 𝒟`, with `L ≥ 0` and `𝒟 ≥ 0`.

No global lower bound on exposure norms is required. This avoids the
economically fragile assumption that every asset carries systematic risk bounded
away from zero (which forbids zero-beta, hedged, and market-neutral assets and
contradicts CAPM/APT). Normalized statements are derived as corollaries under the
weak pairwise condition `β_i ≠ 0` and `β_j ≠ 0`.

## Main results

* `systematic_covariance_lower_bound`: bounds the inner product of exposures below
  by half of `‖β_i‖² + ‖β_j‖² - (L * 𝒟)²`. The core, scale-aware result.
* `systematic_correlation_lower_bound`: when both exposures are nonzero, gives a
  lower bound on the systematic correlation `⟪β̂_i, β̂_j⟫`.
* `total_return_correlation_lower_bound`: multiplying by `√(α_i * α_j)` gives
  the total-return correlation bound.

## Notation

Throughout, `𝒟` denotes the (population) energy distance between risk
characteristic distributions. The theorems below treat `𝒟` abstractly as a
nonnegative real-valued function on pairs, so the proofs depend only on the
Lipschitz hypothesis and Hilbert-space geometry.
-/


variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### Exact energy-distance to covariance identities

These theorems are direct consequences of the `EnergyKernelEmbedding` structure
defined in `Model.lean`. They do not require the Lipschitz hypothesis.
-/

/-- Given an `EnergyKernelEmbedding`, the squared energy distance decomposes
exactly as the systematic variance minus twice the systematic covariance.

  `D_E² = σ_i² + σ_j² − 2·Cov_sys`
-/
theorem energy_to_covariance_exact_identity
    (ke : EnergyKernelEmbedding E) (β_i β_j : E) :
    ke.energy_sq β_i β_j =
      ke.sys_var β_i + ke.sys_var β_j - 2 * ke.sys_cov β_i β_j :=
  ke.identity β_i β_j

/-! ### Canonical inhabitant

`EnergyKernelEmbedding` is a hypothesis carrier: the identities above hold for
*whatever* embedding the caller supplies, and their proofs are projections of
its `identity` field. Without an exhibited inhabitant that layer is conditional
on a structure this development never builds, which is a disclosure defect
rather than a soundness one. The declarations below discharge it minimally.
-/

/--
**Canonical Hilbert-geometry embedding.**

The exposure geometry is itself an `EnergyKernelEmbedding`: take
`sys_var β = ‖β‖²`, `sys_cov = ⟪·, ·⟫_ℝ`, and `energy_sq β_i β_j = ‖β_i − β_j‖²`.
The `identity` field is then the real polarization identity (`norm_sub_sq_real`),
so it holds by computation rather than by assumption.

**Scope.** This exhibits an inhabitant, so the identity layer is non-vacuous,
and it shows the identity holds exactly at the Hilbert geometry the manuscript's
proof sketch uses. It does *not* establish the paper's modelling claim, which
differs in two respects: that `energy_sq` may be replaced by the (scaled) energy
distance between characteristic distributions — the energy-to-MMD step, carried
in the manuscript as a cited hypothesis, not proved here — and that `sys_cov` is
the systematic covariance of returns, which is economic and is formalized
nowhere in this development. In particular, instantiating at this embedding does
not license reading `energy_sq` as an energy distance.
-/
noncomputable def hilbertEnergyKernelEmbedding : EnergyKernelEmbedding E where
  sys_var β := ‖β‖ ^ 2
  sys_cov β_i β_j := inner ℝ β_i β_j
  energy_sq β_i β_j := ‖β_i - β_j‖ ^ 2
  identity β_i β_j := by
    have h := norm_sub_sq_real β_i β_j
    linarith

/--
The identity layer exercised at the canonical inhabitant: applying
`energy_to_covariance_exact_identity` to `hilbertEnergyKernelEmbedding` yields
the polarization identity for exposures. The point is not the statement — which
is `norm_sub_sq_real` rearranged — but that the identity layer is *applied* here
rather than only quantified over. -/
theorem energy_to_covariance_exact_identity_hilbert (β_i β_j : E) :
    ‖β_i - β_j‖ ^ 2 = ‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - 2 * inner ℝ β_i β_j :=
  energy_to_covariance_exact_identity hilbertEnergyKernelEmbedding β_i β_j

/-- The exact correlation formula in terms of energy distance.

When both systematic variances are positive:

  `ρ = (σ_i² + σ_j² − D_E²) / (2·√(σ_i²·σ_j²))`

This follows algebraically from the energy-kernel identity. -/
theorem energy_correlation_exact
    (ke : EnergyKernelEmbedding E) (β_i β_j : E)
    (hσi : 0 < ke.sys_var β_i) (hσj : 0 < ke.sys_var β_j) :
    ke.sys_cov β_i β_j / Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) =
      (ke.sys_var β_i + ke.sys_var β_j - ke.energy_sq β_i β_j) /
        (2 * Real.sqrt (ke.sys_var β_i * ke.sys_var β_j)) := by
  rw [energy_to_covariance_exact_identity ke β_i β_j]
  field_simp [ne_of_gt hσi, ne_of_gt hσj]
  ring

/-- When systematic correlation equals 1 (perfect correlation), the squared
energy distance reduces to the squared difference of standard deviations.

  `D_E² = (σ_i − σ_j)²`

This is the "energy distance = variance distance" condition for perfect
correlation. -/
theorem energy_max_distance_rule
    (ke : EnergyKernelEmbedding E) (β_i β_j : E)
    (hρ : ke.sys_cov β_i β_j / Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) = 1)
    (hσi : 0 < ke.sys_var β_i) (hσj : 0 < ke.sys_var β_j) :
    ke.energy_sq β_i β_j =
      (Real.sqrt (ke.sys_var β_i) - Real.sqrt (ke.sys_var β_j)) ^ 2 := by
  rw [energy_to_covariance_exact_identity ke β_i β_j]
  have hσij_pos : 0 < Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) := by
    apply Real.sqrt_pos.mpr
    exact mul_pos hσi hσj
  have hCov : ke.sys_cov β_i β_j = Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) :=
    eq_of_div_eq_one hρ
  rw [hCov]
  have hsqrt_prod : Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) =
      Real.sqrt (ke.sys_var β_i) * Real.sqrt (ke.sys_var β_j) :=
    Real.sqrt_mul (le_of_lt hσi) (ke.sys_var β_j)
  rw [hsqrt_prod]
  have hσi_nn : 0 ≤ ke.sys_var β_i := le_of_lt hσi
  have hσj_nn : 0 ≤ ke.sys_var β_j := le_of_lt hσj
  linear_combination (Real.sq_sqrt hσi_nn).symm + (Real.sq_sqrt hσj_nn).symm

/-- Diversification boundary: when energy distance dominates the sum of
systematic variances, systematic correlation is non-positive.

  `D_E² ≥ σ_i² + σ_j² ⟹ ρ ≤ 0`

If the risk profiles are too dissimilar for any common factor structure to
explain, the systematic correlation is forced to zero or below. -/
theorem energy_diversification_boundary
    (ke : EnergyKernelEmbedding E) (β_i β_j : E)
    (hDE : ke.sys_var β_i + ke.sys_var β_j ≤ ke.energy_sq β_i β_j)
    (hσi : 0 < ke.sys_var β_i) (hσj : 0 < ke.sys_var β_j) :
    ke.sys_cov β_i β_j / Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) ≤ 0 := by
  rw [energy_to_covariance_exact_identity ke β_i β_j] at hDE
  have hCov_nonpos : ke.sys_cov β_i β_j ≤ 0 := by linarith
  have hσij_pos : 0 < Real.sqrt (ke.sys_var β_i * ke.sys_var β_j) := by
    apply Real.sqrt_pos.mpr
    exact mul_pos hσi hσj
  exact (div_le_iff₀ hσij_pos).mpr (by simpa using hCov_nonpos)

/-! ### Main theorem: systematic covariance lower bound

`systematic_covariance_lower_bound` and
`systematic_covariance_lower_bound_of_approximate_map` moved to
`PricingPerspective.Transmission.Floor` on 2026-08-21. They are imported above and share
this namespace, so every use below still resolves unqualified and no proof changed.

Neither statement ever mentioned energy distance: both take a bare nonnegative real, with
energy distance supplied by the caller. Declaring them here misread that generality as an
energy result — see the `Transmission.Floor` module docstring. The corollaries below, which
*do* fix energy distance, stay.
-/

/-! ### Corollary: systematic correlation lower bound -/

/-!
**Corollary (Systematic Correlation Lower Bound).**

When both exposures are nonzero (so the normalized exposures
`β̂_i = β_i / ‖β_i‖`, `β̂_j = β_j / ‖β_j‖` are well defined), dividing the
covariance bound by the positive factor `‖β_i‖ * ‖β_j‖` gives a lower bound
on the systematic correlation
`ρ^{sys}_{ij} = ⟪β̂_i, β̂_j⟫ = ⟪β_i, β_j⟫ / (‖β_i‖ * ‖β_j‖)`:

  `ρ^{sys}_{ij} ≥ (‖β_i‖² + ‖β_j‖² - (L * 𝒟)²) / (2 * ‖β_i‖ * ‖β_j‖)`.

Only a pairwise nondegeneracy condition is needed, not a uniform lower bound on
all assets' exposure norms, so zero-beta or market-neutral assets are excluded
only from this normalized statement and not from the model.
-/

/--
**Corollary (Systematic Correlation Lower Bound).**

When both exposures are nonzero, dividing the covariance bound by
`‖β_i‖ * ‖β_j‖` yields a lower bound on the systematic correlation
`ρ^{sys}_{ij} = ⟪β̂_i, β̂_j⟫`. Only a pairwise nondegeneracy condition is needed,
not a uniform lower bound on all assets' exposure norms.
-/
theorem systematic_correlation_lower_bound
    (β_i β_j : E) (L D_E : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E) (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hβ_i : β_i ≠ 0) (hβ_j : β_j ≠ 0) :
    (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) / (2 * ‖β_i‖ * ‖β_j‖) ≤
      @Inner.inner ℝ E _ β_i β_j / (‖β_i‖ * ‖β_j‖) := by
  have h := systematic_covariance_lower_bound β_i β_j L D_E hL hD hLip
  have hni : 0 < ‖β_i‖ := norm_pos_iff.mpr hβ_i
  have hnj : 0 < ‖β_j‖ := norm_pos_iff.mpr hβ_j
  have hPne : ‖β_i‖ * ‖β_j‖ ≠ 0 := mul_ne_zero (ne_of_gt hni) (ne_of_gt hnj)
  field_simp [hPne]
  linarith [h]

omit [InnerProductSpace ℝ E] in
/--
**Theorem (Correlation-Floor Numerator Is Feasible).**

The transmission bound forces the numerator of the systematic-correlation
floor to be at most its denominator:

`‖β_i‖² + ‖β_j‖² - (L * D_E)² ≤ 2 * ‖β_i‖ * ‖β_j‖`.

This is the reverse triangle inequality followed by the squared transmission
bound.  It is the division-free form of the Paper 1 claim that the proposed
correlation floor cannot exceed one.
-/
theorem systematic_correlation_floor_numerator_le
    (β_i β_j : E) (L D_E : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E)
    (hLip : ‖β_i - β_j‖ ≤ L * D_E) :
    ‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2 ≤
      2 * ‖β_i‖ * ‖β_j‖ := by
  have hreverse : |‖β_i‖ - ‖β_j‖| ≤ ‖β_i - β_j‖ :=
    abs_norm_sub_norm_le β_i β_j
  have hLD : 0 ≤ L * D_E := mul_nonneg hL hD
  have hsq : |‖β_i‖ - ‖β_j‖| ^ 2 ≤ (L * D_E) ^ 2 :=
    (sq_le_sq₀ (abs_nonneg _) hLD).mpr (hreverse.trans hLip)
  rw [sq_abs] at hsq
  nlinarith

omit [InnerProductSpace ℝ E] in
/--
**Corollary (Systematic Correlation Floor Is at Most One).**

For nonzero exposures, dividing
`systematic_correlation_floor_numerator_le` by the positive normalization
`2 * ‖β_i‖ * ‖β_j‖` proves that the lower bound stated for systematic
correlation lies in the admissible correlation range on its upper side.
-/
theorem systematic_correlation_floor_le_one
    (β_i β_j : E) (L D_E : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E)
    (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hβ_i : β_i ≠ 0) (hβ_j : β_j ≠ 0) :
    (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) /
        (2 * ‖β_i‖ * ‖β_j‖) ≤ 1 := by
  have hden : 0 < 2 * ‖β_i‖ * ‖β_j‖ := by
    positivity
  rw [div_le_one hden]
  exact systematic_correlation_floor_numerator_le β_i β_j L D_E hL hD hLip

/-! ### Corollary: total-return correlation lower bound -/

/-!
**Corollary (Total-Return Correlation Lower Bound).**

Let `α_i = ‖β_i‖² / (‖β_i‖² + σ²_{ε_i}) ∈ [0, 1]` be the share of asset
`i`'s return variance attributable to systematic risk. Assuming idiosyncratic
shocks are cross-sectionally uncorrelated
(`Cov(ε_i, ε_j) = 0` for `i ≠ j`) and orthogonal to the factor field, the
total-return correlation factorizes as
`ρ_{ij} = √(α_i * α_j) * ρ^{sys}_{ij}`. Multiplying the systematic correlation
bound by the nonnegative factor `√(α_i * α_j)` yields the total-return bound.
-/

/--
**Corollary (Total-Return Correlation Lower Bound).**

Multiplying the systematic correlation bound by the nonnegative factor
`√(α_i * α_j)` (the geometric mean of the systematic variance shares) gives
the total-return correlation bound, assuming idiosyncratic shocks are
cross-sectionally uncorrelated.
-/
theorem total_return_correlation_lower_bound
    (β_i β_j : E) (L D_E α_i α_j : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E) (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hβ_i : β_i ≠ 0) (hβ_j : β_j ≠ 0) (_hα_i : 0 ≤ α_i) (_hα_j : 0 ≤ α_j) :
    Real.sqrt (α_i * α_j) *
        ((‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) / (2 * ‖β_i‖ * ‖β_j‖)) ≤
      Real.sqrt (α_i * α_j) *
        (@Inner.inner ℝ E _ β_i β_j / (‖β_i‖ * ‖β_j‖)) := by
  have h := systematic_correlation_lower_bound β_i β_j L D_E hL hD hLip hβ_i hβ_j
  exact mul_le_mul_of_nonneg_left h (Real.sqrt_nonneg (α_i * α_j))

/-! ### Admissibility of the divergence: separation forced by Lipschitz transmission (2026-07-25)

The two results below constrain **Assumption 2 (Market Efficiency)** of Paper 1 — the
transmission bound `‖T C_i − T C_j‖ ≤ L · 𝒟(C_i, C_j)` — and not the covariance bound
itself. `systematic_covariance_lower_bound` receives `L * D_E` as a bare nonnegative real
and is silent on which divergence `𝒟` is; these results say that the choice of `𝒟` is
nevertheless not free, because a divergence that fails to separate two distinct
distributions makes the transmission bound unsatisfiable for every injective operator `T`
and every constant `L`.

The risk space `P` carries no structure: only the inequality is used, so `d` is not
assumed to be symmetric, nonnegative, or a metric, and `L` is not assumed nonnegative.
-/

/--
**Lemma (Transmission Collapses on Unseparated Inputs).**

Let `T : P → E` send risk characteristic distributions to systematic exposures in the
covariance Hilbert space `E`, and let `d` be any divergence on `P` satisfying the
transmission bound of Assumption 2 (Market Efficiency), `‖T p − T q‖ ≤ L * d p q`. If
`d p q = 0`, then `T p = T q`.

Nothing beyond that single inequality is used: `P` is a bare type, `d` is not assumed to
be a metric (neither symmetric nor nonnegative), and `L` is not assumed nonnegative. The
proof pairs the bound at the pair `p, q` with `0 ≤ ‖T p − T q‖`.

**Economic reading.** The divergence fixes the model's resolution. If `𝒟` reads two
distinct risk characteristic distributions as identical, market efficiency in the sense of
Assumption 2 forces the market to assign them identical systematic exposures; no
calibration of the transmission constant `L` recovers detail that `𝒟` has already
discarded.

This constrains Assumption 2, not `systematic_covariance_lower_bound`, which takes the
product `L * D_E` over a bare nonnegative real.
-/
lemma exposure_eq_of_divergence_eq_zero {P : Type*}
    (T : P → E) (d : P → P → ℝ) (L : ℝ) (p q : P)
    (hLip : ∀ r s : P, ‖T r - T s‖ ≤ L * d r s) (hd : d p q = 0) :
    T p = T q := by
  have h := hLip p q
  rw [hd, mul_zero] at h
  exact norm_sub_eq_zero_iff.mp (le_antisymm h (norm_nonneg _))

/--
**Proposition (Admissible Divergences Separate Points).**

Immediate consequence of `exposure_eq_of_divergence_eq_zero` under injectivity of `T`, and
the direction Paper 1 uses. If the transmission operator `T` is injective — distinct risk
characteristic distributions carry distinct systematic exposures, an identification
requirement over and above Assumption 1 (Common Technology), which only makes exposures a
function of characteristics — then any divergence `d` admissible for Assumption 2 (Market
Efficiency) must itself separate points: `d p q = 0 → p = q`. Read contrapositively: if
`p ≠ q` while `d p q = 0`, then no injective `T` satisfies the transmission bound, at any
`L`.

**Economic reading.** This is an admissibility criterion for the choice of divergence, and
it is the theoretical form of the reply to the objection that the exercise could be run
with any convenient distance between characteristic distributions. A statistic that
assigns distance zero to a pair of distinct distributions supports the transmission bound
for no injective `T` and no constant `L`, so the model is left without an operator rather
than with a poorly calibrated one. Which candidate divergences pass or fail the criterion
— the degeneracy of mean-based distances, or the separation property of the energy
distance over a ground space of strong negative type — is argued in the paper and is not
formalized here.

As above, this constrains Assumption 2, not `systematic_covariance_lower_bound`. Both
statements are about the population divergence `d`; neither speaks to a sample estimator,
and neither compares the empirical performance of competing divergences.
-/
theorem divergence_separates_of_injective_transmission {P : Type*}
    (T : P → E) (d : P → P → ℝ) (L : ℝ) (p q : P)
    (hLip : ∀ r s : P, ‖T r - T s‖ ≤ L * d r s) (hT : Function.Injective T)
    (hd : d p q = 0) :
    p = q :=
  hT (exposure_eq_of_divergence_eq_zero T d L p q hLip hd)

end PricingPerspective.ContinuousAPT
