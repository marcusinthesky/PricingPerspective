import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Positivity

set_option linter.style.longLine false

/-!
# Transmission covariance floor

The model-independent half of the transmission layer: a Lipschitz bound from *any*
nonnegative characteristic dissimilarity to exposure distance forces a floor on
systematic covariance, by polarization alone.

## Why this file exists

These two theorems were previously declared in `Continuous.Energy.Core`, which made them
read as results *about energy distance*. They are not. Neither statement mentions a
characteristic distance at all: both take a bare nonnegative real `D`, and energy distance
is one instantiation among many. Blueprint `chap:energy-specialization` records the
correction — "any nonnegative dissimilarity satisfying the transmission bound serves" —
and names `cor:approximate-map-bound` as one of the two nodes to rehome into the trunk.
This file is that rehoming; `Continuous.Energy.Core` now imports it rather than owning it.

Contrast with `Transmission.Envelope`, which is *not* dissimilarity-agnostic: the sharp
covariance envelope needs quadratic transport specifically, because minimizing
`𝔼_π ‖Z − Y‖²` over couplings is the same optimization polarization converts into an inner
product. The floor below needs no optimization over couplings, and correspondingly places
no demand on which distance is used.

## Namespace

Declarations stay in `PricingPerspective.ContinuousAPT` for now. Renaming them into
`PricingPerspective.Transmission` would move every mapped claim's `declaration` field and
force a disclosure re-review; the measure-theoretic port re-states these theorems anyway,
so the rename is deferred to that step and paid once.

## References

* Gawronsky & Huang, transmission covariance floor (`thm:energy-bound`).
* `.context/chat/2026-08-16_random_functions/figures/hilbert_model.md`, turn 4, which is
  where the distance-agnosticism of the floor was argued explicitly.
-/

namespace PricingPerspective.ContinuousAPT

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### The floor under an exact common map -/

/--
**Theorem (Systematic Covariance Lower Bound).**

If the exposure distance is controlled by the characteristic dissimilarity,
`‖β_i - β_j‖ ≤ L * 𝒟`, with `L ≥ 0` and `𝒟 ≥ 0`, then

  `⟪β_i, β_j⟫ ≥ (1/2) * (‖β_i‖² + ‖β_j‖² - (L * 𝒟)²)`.

This is the systematic covariance lower bound and is the core, scale-aware
result; it degenerates to a non-positive bound (trivially true by the Lipschitz
hypothesis) when an exposure vanishes.

It is obtained by combining the real polarization identity
`‖x - y‖² = ‖x‖² - 2 * ⟪x, y⟫ + ‖y‖²` with the Lipschitz bound squared. It is
scale-aware: it never divides by an exposure norm, so it remains valid (and
non-vacuous) for assets with arbitrarily small systematic risk, including
zero-beta assets. In the limit `𝒟 → 0` the Lipschitz hypothesis forces
`β_i → β_j`, so the bound forces the systematic correlation toward one.

**Economic interpretation.** `⟪β_i, β_j⟫` is the systematic covariance of the
two assets. The bound says that similar risk characteristic distributions
(small `𝒟`) force a high floor on systematic covariance, with the floor set by
the assets' own systematic risk scales `‖β_i‖`, `‖β_j‖` rather than by an
exogenous constant.
-/
theorem systematic_covariance_lower_bound
    (β_i β_j : E) (L D_E : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E)
    (hLip : ‖β_i - β_j‖ ≤ L * D_E) :
    (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) ≤
      @Inner.inner ℝ E _ β_i β_j := by
  -- Real polarization identity: ‖β_i - β_j‖² = ‖β_i‖² - 2 * ⟪β_i, β_j⟫ + ‖β_j‖²
  have hpol : ‖β_i - β_j‖ ^ 2 = ‖β_i‖ ^ 2 - 2 * @Inner.inner ℝ E _ β_i β_j + ‖β_j‖ ^ 2 :=
    norm_sub_sq_real β_i β_j
  -- Both sides of the Lipschitz bound are nonnegative, so it can be squared.
  have hLD : 0 ≤ L * D_E := by positivity
  have hnn : 0 ≤ ‖β_i - β_j‖ := norm_nonneg _
  have hsq : ‖β_i - β_j‖ ^ 2 ≤ (L * D_E) ^ 2 :=
    (sq_le_sq₀ hnn hLD).mpr hLip
  linarith [hpol, hsq]

/-! ### Covariance floor under an approximate common map -/

/--
**Proposition (Covariance Floor Under an Approximate Common Map).**

`systematic_covariance_lower_bound` inherits the paper's exact premise
`β_i = T(C_i)`: the common map must reproduce every firm's exposure without
error.  This weakens that.  Write `T_i` for the common map's value at firm `i`
and let the realized exposure deviate from it by at most `τ_i`:

  `‖β_i - T_i‖ ≤ τ_i`,   `‖T_i - T_j‖ ≤ L * d`.

Then `‖β_i - β_j‖ ≤ L * d + τ_i + τ_j` by the triangle inequality, and the floor
follows by instantiating `systematic_covariance_lower_bound` at transmission
constant `1` with distance argument `B = L * d + τ_i + τ_j`.

The slack terms absorb firm-specific mapping error — omitted private
information, balance-sheet and hedging effects, media selection — so the
conclusion no longer requires the map to be exact on any firm.  Exact
transmission is the special case `τ_i = τ_j = 0`, which reproduces
`systematic_covariance_lower_bound` verbatim.

`d` is any nonnegative characteristic dissimilarity; nothing here is specific to
energy distance.

This is the slack-inflated form the random-exposure model needs, which is why
blueprint `cor:approximate-map-bound` keeps its manifest binding through the
retirement of the deterministic model around it.
-/
theorem systematic_covariance_lower_bound_of_approximate_map
    (β_i β_j T_i T_j : E) (L d τ_i τ_j : ℝ)
    (hL : 0 ≤ L) (hd : 0 ≤ d) (hτ_i : 0 ≤ τ_i) (hτ_j : 0 ≤ τ_j)
    (hT : ‖T_i - T_j‖ ≤ L * d)
    (hδ_i : ‖β_i - T_i‖ ≤ τ_i) (hδ_j : ‖β_j - T_j‖ ≤ τ_j) :
    (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * d + τ_i + τ_j) ^ 2) ≤
      @Inner.inner ℝ E _ β_i β_j := by
  have hLd : 0 ≤ L * d := mul_nonneg hL hd
  have hBnn : 0 ≤ L * d + τ_i + τ_j := by linarith
  have h4 : ‖β_i - β_j‖ ≤ ‖β_i - T_i‖ + ‖T_i - T_j‖ + ‖T_j - β_j‖ := by
    simpa [dist_eq_norm] using dist_triangle4 β_i T_i T_j β_j
  rw [norm_sub_rev T_j β_j] at h4
  have hchain : ‖β_i - β_j‖ ≤ 1 * (L * d + τ_i + τ_j) := by linarith
  simpa using
    systematic_covariance_lower_bound β_i β_j 1 (L * d + τ_i + τ_j) zero_le_one hBnn hchain

end PricingPerspective.ContinuousAPT
