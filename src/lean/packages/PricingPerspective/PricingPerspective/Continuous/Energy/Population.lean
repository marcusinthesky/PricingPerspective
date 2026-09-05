import PricingPerspective.Continuous.Energy.Core

/-!
# Population return decomposition

This module realizes the Paper 1 population reading of returns as systematic
exposures plus orthogonal idiosyncratic residuals in a real inner-product space.
-/

namespace PricingPerspective.ContinuousAPT

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

open scoped InnerProductSpace

/--
**Theorem (Total Variance Decomposition).**

Pythagoras for the return/exposure/residual decomposition: if the systematic
exposure `β` and idiosyncratic residual `ε` are orthogonal, total variance
splits additively as `‖β + ε‖² = ‖β‖² + ‖ε‖²`.
-/
theorem total_variance_decomposition (β ε : E) (horth : ⟪β, ε⟫_ℝ = 0) :
    ‖β + ε‖ ^ 2 = ‖β‖ ^ 2 + ‖ε‖ ^ 2 := by
  have h := norm_add_sq_real β ε
  rw [horth] at h
  linarith

/--
**Theorem (Total Covariance Reduces to Systematic Covariance).**

If idiosyncratic shocks are cross-sectionally uncorrelated (`⟪ε_i, ε_j⟫ = 0`)
and orthogonal to the other asset's factor exposure (`⟪β_i, ε_j⟫ = 0`,
`⟪β_j, ε_i⟫ = 0`), the total-return covariance equals the systematic
covariance: `⟪β_i + ε_i, β_j + ε_j⟫ = ⟪β_i, β_j⟫`.
-/
theorem total_covariance_eq_systematic
    (β_i β_j ε_i ε_j : E)
    (hij : ⟪β_i, ε_j⟫_ℝ = 0) (hji : ⟪β_j, ε_i⟫_ℝ = 0) (hee : ⟪ε_i, ε_j⟫_ℝ = 0) :
    ⟪β_i + ε_i, β_j + ε_j⟫_ℝ = ⟪β_i, β_j⟫_ℝ := by
  have hjih : ⟪ε_i, β_j⟫_ℝ = 0 := by rw [real_inner_comm]; exact hji
  rw [inner_add_left, inner_add_right, inner_add_right, hij, hjih, hee]
  ring

/--
**Theorem (Variance Share Lies in `[0,1]`).**

The systematic variance share `α = ‖β‖² / ‖β + ε‖²` is *derived*, not assumed:
under orthogonality it always lies in `[0, 1]`, realizing the paper's
`α_i ∈ [0,1]` claim as a consequence rather than a hypothesis.
-/
theorem variance_share_mem_Icc (β ε : E) (horth : ⟪β, ε⟫_ℝ = 0) (hr : β + ε ≠ 0) :
    ‖β‖ ^ 2 / ‖β + ε‖ ^ 2 ∈ Set.Icc (0 : ℝ) 1 := by
  have hdecomp := total_variance_decomposition β ε horth
  have hc_pos : 0 < ‖β + ε‖ ^ 2 := by
    have hn : 0 < ‖β + ε‖ := norm_pos_iff.mpr hr
    positivity
  rw [Set.mem_Icc]
  refine ⟨div_nonneg (sq_nonneg _) hc_pos.le, ?_⟩
  rw [div_le_one hc_pos, hdecomp]
  nlinarith [sq_nonneg ‖ε‖]

/--
**Theorem (Correlation Factorization).**

The total-return correlation factorizes as `ρ_{ij} = √(α_i · α_j) · ρ^{sys}_{ij}`
with the variance shares `α_i, α_j` *derived* (`variance_share_mem_Icc`) rather
than assumed. Positivity of `‖β_i + ε_i‖`, `‖β_j + ε_j‖` is derived from
`hβ_i, hβ_j` via `total_variance_decomposition` — not an added hypothesis.
-/
theorem correlation_factorization
    (β_i β_j ε_i ε_j : E)
    (hii : ⟪β_i, ε_i⟫_ℝ = 0) (hjj : ⟪β_j, ε_j⟫_ℝ = 0)
    (hij : ⟪β_i, ε_j⟫_ℝ = 0) (hji : ⟪β_j, ε_i⟫_ℝ = 0) (hee : ⟪ε_i, ε_j⟫_ℝ = 0)
    (hβ_i : β_i ≠ 0) (hβ_j : β_j ≠ 0) :
    ⟪β_i + ε_i, β_j + ε_j⟫_ℝ / (‖β_i + ε_i‖ * ‖β_j + ε_j‖) =
      Real.sqrt ((‖β_i‖ ^ 2 / ‖β_i + ε_i‖ ^ 2) * (‖β_j‖ ^ 2 / ‖β_j + ε_j‖ ^ 2)) *
        (⟪β_i, β_j⟫_ℝ / (‖β_i‖ * ‖β_j‖)) := by
  have ha_pos : 0 < ‖β_i‖ := norm_pos_iff.mpr hβ_i
  have hb_pos : 0 < ‖β_j‖ := norm_pos_iff.mpr hβ_j
  have ha_ne : ‖β_i‖ ≠ 0 := ha_pos.ne'
  have hb_ne : ‖β_j‖ ≠ 0 := hb_pos.ne'
  have hc2 : 0 < ‖β_i + ε_i‖ ^ 2 := by
    rw [total_variance_decomposition β_i ε_i hii]
    have hb2 : 0 < ‖β_i‖ ^ 2 := pow_pos ha_pos 2
    nlinarith [sq_nonneg ‖ε_i‖]
  have hd2 : 0 < ‖β_j + ε_j‖ ^ 2 := by
    rw [total_variance_decomposition β_j ε_j hjj]
    have hb2 : 0 < ‖β_j‖ ^ 2 := pow_pos hb_pos 2
    nlinarith [sq_nonneg ‖ε_j‖]
  have hc_ne : ‖β_i + ε_i‖ ≠ 0 := by
    intro h; rw [h] at hc2; norm_num at hc2
  have hd_ne : ‖β_j + ε_j‖ ≠ 0 := by
    intro h; rw [h] at hd2; norm_num at hd2
  have hc_pos : 0 < ‖β_i + ε_i‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hc_ne)
  have hd_pos : 0 < ‖β_j + ε_j‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hd_ne)
  have hnum := total_covariance_eq_systematic β_i β_j ε_i ε_j hij hji hee
  have hsqrt :
      Real.sqrt ((‖β_i‖ ^ 2 / ‖β_i + ε_i‖ ^ 2) * (‖β_j‖ ^ 2 / ‖β_j + ε_j‖ ^ 2)) =
        (‖β_i‖ * ‖β_j‖) / (‖β_i + ε_i‖ * ‖β_j + ε_j‖) := by
    rw [show (‖β_i‖ ^ 2 / ‖β_i + ε_i‖ ^ 2) * (‖β_j‖ ^ 2 / ‖β_j + ε_j‖ ^ 2) =
        ((‖β_i‖ * ‖β_j‖) / (‖β_i + ε_i‖ * ‖β_j + ε_j‖)) ^ 2 by
      field_simp]
    exact Real.sqrt_sq (by positivity)
  rw [hnum, hsqrt]
  field_simp

/--
**Theorem (Population Total-Return Correlation Lower Bound).**

The population realization of `total_return_correlation_lower_bound`: the
variance-share factor and cross-sectional orthogonality hypotheses are made
explicit and the `√(α_i · α_j)` factor is derived from `correlation_factorization`
rather than assumed, consuming the previously-unused `_hα` shape. The original
`total_return_correlation_lower_bound` is left untouched.
-/
theorem total_return_correlation_lower_bound'
    (β_i β_j ε_i ε_j : E) (L D_E : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E) (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hβ_i : β_i ≠ 0) (hβ_j : β_j ≠ 0)
    (hii : ⟪β_i, ε_i⟫_ℝ = 0) (hjj : ⟪β_j, ε_j⟫_ℝ = 0)
    (hij : ⟪β_i, ε_j⟫_ℝ = 0) (hji : ⟪β_j, ε_i⟫_ℝ = 0) (hee : ⟪ε_i, ε_j⟫_ℝ = 0) :
    Real.sqrt ((‖β_i‖ ^ 2 / ‖β_i + ε_i‖ ^ 2) * (‖β_j‖ ^ 2 / ‖β_j + ε_j‖ ^ 2)) *
        ((‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) / (2 * ‖β_i‖ * ‖β_j‖)) ≤
      ⟪β_i + ε_i, β_j + ε_j⟫_ℝ / (‖β_i + ε_i‖ * ‖β_j + ε_j‖) := by
  rw [correlation_factorization β_i β_j ε_i ε_j hii hjj hij hji hee hβ_i hβ_j]
  have hsys := systematic_correlation_lower_bound β_i β_j L D_E hL hD hLip hβ_i hβ_j
  have hsqrt_nonneg :
      0 ≤ Real.sqrt ((‖β_i‖ ^ 2 / ‖β_i + ε_i‖ ^ 2) * (‖β_j‖ ^ 2 / ‖β_j + ε_j‖ ^ 2)) :=
    Real.sqrt_nonneg _
  exact mul_le_mul_of_nonneg_left hsys hsqrt_nonneg

/-! ### Bounded residual dependence -/

/--
**Theorem (Total Covariance Floor Under Bounded Residual Dependence).**

`total_covariance_eq_systematic` and hence `total_return_correlation_lower_bound'`
assume cross-sectional residual orthogonality *exactly* (`⟪ε_i, ε_j⟫ = 0`).  This
replaces that equality by the one-sided restriction `-γ ≤ ⟪ε_i, ε_j⟫`: residual
covariance may be negative, but not arbitrarily so.  The total-return covariance
then inherits the systematic floor less `γ`:

  `⟪β_i + ε_i, β_j + ε_j⟫ ≥ (1/2)(‖β_i‖² + ‖β_j‖² - (L·D_E)²) - γ`.

Exact orthogonality is the special case `γ = 0`.  Note this is a *sibling* of the
primed corollary, not a weakening of it: the `√(α_i·α_j)·ρ^sys` factorization does
not survive, because `γ` does not factor through the variance shares.

Residual orthogonality to the *other* asset's exposure (`hij`, `hji`) is retained;
only cross-sectional residual orthogonality is relaxed.
-/
theorem total_covariance_lower_bound_of_bounded_residual
    (β_i β_j ε_i ε_j : E) (L D_E γ : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E) (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hij : ⟪β_i, ε_j⟫_ℝ = 0) (hji : ⟪β_j, ε_i⟫_ℝ = 0)
    (hee : -γ ≤ ⟪ε_i, ε_j⟫_ℝ) :
    (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) - γ ≤
      ⟪β_i + ε_i, β_j + ε_j⟫_ℝ := by
  have hjih : ⟪ε_i, β_j⟫_ℝ = 0 := by rw [real_inner_comm]; exact hji
  have hexp : ⟪β_i + ε_i, β_j + ε_j⟫_ℝ = ⟪β_i, β_j⟫_ℝ + ⟪ε_i, ε_j⟫_ℝ := by
    rw [inner_add_left, inner_add_right, inner_add_right, hij, hjih]
    ring
  have hsys := systematic_covariance_lower_bound β_i β_j L D_E hL hD hLip
  rw [hexp]
  linarith

/--
**Corollary (Total-Return Correlation Floor Under Bounded Residual Dependence).**

Divide `total_covariance_lower_bound_of_bounded_residual` by the positive
normalization `‖β_i + ε_i‖ * ‖β_j + ε_j‖`.  The denominator is the product of
total-return standard deviations, so the right-hand side is the total-return
correlation and the left-hand side is the paper's systematic floor, net of the
residual-dependence allowance `γ`, in the same units.
-/
theorem total_return_correlation_lower_bound_of_bounded_residual
    (β_i β_j ε_i ε_j : E) (L D_E γ : ℝ)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E) (hLip : ‖β_i - β_j‖ ≤ L * D_E)
    (hij : ⟪β_i, ε_j⟫_ℝ = 0) (hji : ⟪β_j, ε_i⟫_ℝ = 0)
    (hee : -γ ≤ ⟪ε_i, ε_j⟫_ℝ)
    (hr_i : β_i + ε_i ≠ 0) (hr_j : β_j + ε_j ≠ 0) :
    ((1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * D_E) ^ 2) - γ) /
        (‖β_i + ε_i‖ * ‖β_j + ε_j‖) ≤
      ⟪β_i + ε_i, β_j + ε_j⟫_ℝ / (‖β_i + ε_i‖ * ‖β_j + ε_j‖) := by
  have h1 : 0 < ‖β_i + ε_i‖ := norm_pos_iff.mpr hr_i
  have h2 : 0 < ‖β_j + ε_j‖ := norm_pos_iff.mpr hr_j
  have hpos : 0 < ‖β_i + ε_i‖ * ‖β_j + ε_j‖ := by positivity
  have h :=
    total_covariance_lower_bound_of_bounded_residual β_i β_j ε_i ε_j L D_E γ hL hD hLip
      hij hji hee
  exact (div_le_div_iff₀ hpos hpos).mpr (mul_le_mul_of_nonneg_right h (le_of_lt hpos))

/--
**Theorem (Covariance Transport Under a Universal Isometry).**

A linear isometry `T : E →ₗᵢ[ℝ] F` (the "universal operator" of the
population reading) preserves the covariance inner product: named corollary
of `LinearIsometry.inner_map_map`.
-/
theorem covariance_inner_transport {F : Type*} [NormedAddCommGroup F]
    [InnerProductSpace ℝ F] (T : E →ₗᵢ[ℝ] F) (β_i β_j : E) :
    ⟪T β_i, T β_j⟫_ℝ = ⟪β_i, β_j⟫_ℝ :=
  LinearIsometry.inner_map_map T β_i β_j

end PricingPerspective.ContinuousAPT
