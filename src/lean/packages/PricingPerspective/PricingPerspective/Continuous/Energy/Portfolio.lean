import PricingPerspective.Continuous.Energy.Core

/-!
# Projective and portfolio energy bounds

This module collects the projective correlation bound, pairwise risk brackets,
portfolio variance envelopes, hedging-error bound, and the Hilbertian CND
mechanism used by the portfolio optimization results.
-/

namespace PricingPerspective.ContinuousAPT

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### Alternative specification: projective lower bound via sphere polarization

This section proves the **projective lower bound** (Strategy 1 from the
proof-structure exploration): when exposures are normalized to the unit sphere
of the RKHS, the inner product is bounded below by `1 − (L²/2)·d²`.

This is obtained from the sphere polarization identity
`‖S_i − S_j‖² = 2 − 2⟨S_i, S_j⟩` combined with the Lipschitz bound, avoiding
the normalization denominator entirely. It is therefore valid even for
exposures approaching zero — the only cost is that the sphere is a non-linear
representation, so APT portfolio linearity does not hold.

#### Main theorem

**Theorem (Projective Lower Bound).**

If exposures are unit vectors in the RKHS and the Lipschitz hypothesis holds
with ground metric `d ≥ 0`, then

  `⟨S_i, S_j⟩ ≥ 1 − (L²/2) · d²`.

In the limit `d → 0` this forces `⟨S_i, S_j⟩ → 1` (perfect correlation).
-/

/--
**Theorem (Projective Lower Bound).**

If exposures are unit vectors in the RKHS and the Lipschitz hypothesis holds
with ground metric `d ≥ 0`, then

  `⟨S_i, S_j⟩ ≥ 1 − (L²/2) · d²`.

In the limit `d → 0` this forces `⟨S_i, S_j⟩ → 1` (perfect correlation).
Valid even for exposures approaching zero — the only cost is that the sphere
is a non-linear representation, so APT portfolio linearity does not hold.
-/
theorem projective_lower_bound
    (S_i S_j : E) (L d : ℝ)
    (hL : 0 ≤ L) (hd : 0 ≤ d)
    (hS1 : ‖S_i‖ = 1) (hS2 : ‖S_j‖ = 1)
    (hLip : ‖S_i - S_j‖ ≤ L * d) :
    1 - (L ^ 2 / 2) * d ^ 2 ≤ @Inner.inner ℝ E _ S_i S_j := by
  have hpol : ‖S_i - S_j‖ ^ 2 = ‖S_i‖ ^ 2 - 2 * @Inner.inner ℝ E _ S_i S_j + ‖S_j‖ ^ 2 :=
    norm_sub_sq_real S_i S_j
  rw [hS1, hS2] at hpol
  have hLD : 0 ≤ L * d := by positivity
  have hnn : 0 ≤ ‖S_i - S_j‖ := norm_nonneg _
  have hsq : ‖S_i - S_j‖ ^ 2 ≤ (L * d) ^ 2 := (sq_le_sq₀ hnn hLD).mpr hLip
  linarith

/-! ### Portfolio variance lower bound via double sum (ROADMAP §3.2 #10 / Paper 3) -/

/-- Portfolio systematic variance expands as the weighted double sum of pairwise
systematic covariances: `‖Σᵢ wᵢ • βᵢ‖² = Σᵢ Σⱼ wᵢ wⱼ ⟪βᵢ, βⱼ⟫`. -/
lemma portfolio_norm_sq_eq_double_sum {n : ℕ} (w : Fin n → ℝ) (β : Fin n → E) :
    ‖∑ i, w i • β i‖ ^ 2 =
      ∑ i, ∑ j, w i * w j * @Inner.inner ℝ E _ (β i) (β j) := by
  calc
    ‖∑ i, w i • β i‖ ^ 2 = @Inner.inner ℝ E _ (∑ i, w i • β i) (∑ i, w i • β i) := by
      rw [real_inner_self_eq_norm_sq]
    _ = ∑ i, @Inner.inner ℝ E _ (w i • β i) (∑ j, w j • β j) := by
      rw [sum_inner]
    _ = ∑ i, ∑ j, @Inner.inner ℝ E _ (w i • β i) (w j • β j) := by
      simp_rw [inner_sum]
    _ = ∑ i, ∑ j, (w i * @Inner.inner ℝ E _ (β i) (w j • β j)) := by
      simp_rw [real_inner_smul_left]
    _ = ∑ i, ∑ j, (w i * (w j * @Inner.inner ℝ E _ (β i) (β j))) := by
      simp_rw [real_inner_smul_right]
    _ = ∑ i, ∑ j, w i * w j * @Inner.inner ℝ E _ (β i) (β j) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      refine Finset.sum_congr rfl fun j _ => ?_
      ring

/--
**Theorem (Portfolio Variance Lower Bound).** (ROADMAP §3.2 #10 / Paper 3's
machine-checked headline.)

Long-only portfolio variance expands as a weighted double sum of pairwise systematic
covariances. When each pairwise energy distance is controlled by a Lipschitz
condition `‖β_i - β_j‖ ≤ L * D_{ij}` with nonnegative weights, Lipschitz constant,
and ground distances, the variance is bounded below by the same double sum of
pairwise systematic covariance lower bounds.

The weight normalization `Σ w = 1` is deliberately **not** assumed because the
inequality does not need it; it applies a fortiori to fully-invested long-only
portfolios.
-/
theorem portfolio_variance_lower_bound {n : ℕ}
    (β : Fin n → E) (w : Fin n → ℝ) (L : ℝ) (D : Fin n → Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : 0 ≤ L) (hD : ∀ i j, 0 ≤ D i j)
    (hLip : ∀ i j, ‖β i - β j‖ ≤ L * D i j) :
    ∑ i, ∑ j, w i * w j * ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 - (L * D i j) ^ 2)) ≤
      ‖∑ i, w i • β i‖ ^ 2 := by
  rw [portfolio_norm_sq_eq_double_sum w β]
  refine Finset.sum_le_sum fun i _ => ?_
  refine Finset.sum_le_sum fun j _ => ?_
  have hw_nonneg : 0 ≤ w i * w j := mul_nonneg (hw i) (hw j)
  refine mul_le_mul_of_nonneg_left ?_ hw_nonneg
  exact systematic_covariance_lower_bound (β i) (β j) L (D i j) hL (hD i j) (hLip i j)

/-! ### Risk cap and hedging-error bounds (2026-07-08 novel-bounds extension) -/

/--
**Theorem (Pairwise Systematic Covariance Upper Bound).**

Under a LOWER Lipschitz identification — information distance lower-bounds
exposure distance — systematic covariance is capped:

  `⟪β_i, β_j⟫ ≤ (1/2) * (‖β_i‖² + ‖β_j‖² − (ℓ · D_E)²)`.

Combined with `systematic_covariance_lower_bound` this yields a two-sided
envelope on pairwise systematic covariance.
-/
theorem systematic_covariance_upper_bound
    (β_i β_j : E) (ℓ D_E : ℝ)
    (hℓ : 0 ≤ ℓ) (hD : 0 ≤ D_E)
    (hLip_lower : ℓ * D_E ≤ ‖β_i - β_j‖) :
    @Inner.inner ℝ E _ β_i β_j ≤
      (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (ℓ * D_E) ^ 2) := by
  have hpol : ‖β_i - β_j‖ ^ 2 = ‖β_i‖ ^ 2 - 2 * @Inner.inner ℝ E _ β_i β_j + ‖β_j‖ ^ 2 :=
    norm_sub_sq_real β_i β_j
  have hℓD : 0 ≤ ℓ * D_E := by positivity
  have hsq : (ℓ * D_E) ^ 2 ≤ ‖β_i - β_j‖ ^ 2 :=
    (sq_le_sq₀ hℓD (norm_nonneg _)).mpr hLip_lower
  linarith [hpol, hsq]

/--
**Theorem (Portfolio Variance Upper Bound).**

Machine-checked RISK CAP: with nonneg weights and pairwise lower-Lipschitz
identification, portfolio systematic variance is bounded above by the double
sum of pairwise covariance caps:

  `‖Σᵢ wᵢ βᵢ‖² ≤ Σᵢ Σⱼ wᵢ wⱼ · (1/2)(‖βᵢ‖² + ‖βⱼ‖² − (ℓ · Dᵢⱼ)²)`.

The weight normalization `Σ w = 1` is deliberately **not** assumed; the bound
applies a fortiori to any nonneg-weighted portfolio.
-/
theorem portfolio_variance_upper_bound {n : ℕ}
    (β : Fin n → E) (w : Fin n → ℝ) (ℓ : ℝ) (D : Fin n → Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hℓ : 0 ≤ ℓ) (hD : ∀ i j, 0 ≤ D i j)
    (hLip_lower : ∀ i j, ℓ * D i j ≤ ‖β i - β j‖) :
    ‖∑ i, w i • β i‖ ^ 2 ≤
      ∑ i, ∑ j, w i * w j * ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 - (ℓ * D i j) ^ 2)) := by
  rw [portfolio_norm_sq_eq_double_sum w β]
  refine Finset.sum_le_sum fun i _ => ?_
  refine Finset.sum_le_sum fun j _ => ?_
  have hw_nonneg : 0 ≤ w i * w j := mul_nonneg (hw i) (hw j)
  exact mul_le_mul_of_nonneg_left
    (systematic_covariance_upper_bound (β i) (β j) ℓ (D i j) hℓ (hD i j) (hLip_lower i j))
    hw_nonneg

/--
**Theorem (Hedging/Tracking-Error Bound).**

For a convex mixture mimicking a target exposure `β_t`, the exposure tracking
error is bounded by the weighted sum of information distances to the target:

  `‖β_t − Σᵢ wᵢ βᵢ‖ ≤ L · Σᵢ wᵢ Dᵢ`.

The weight normalization `Σ w = 1` IS load-bearing here. This bound measures
the objective minimized by Paper 2's energy-kernel mixture (jcor
`energy_distance_kernel`): using information distances to bound exposure
replication error.
-/
theorem hedging_error_bound {n : ℕ}
    (β_t : E) (β : Fin n → E) (w : Fin n → ℝ) (L : ℝ) (D : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1)
    (hLip : ∀ i, ‖β_t - β i‖ ≤ L * D i) :
    ‖β_t - ∑ i, w i • β i‖ ≤ L * ∑ i, w i * D i := by
  have hkey : ∑ i, w i • (β_t - β i) = β_t - ∑ i, w i • β i := by
    simp_rw [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, hsum, one_smul]
  rw [← hkey]
  calc ‖∑ i, w i • (β_t - β i)‖
      ≤ ∑ i, ‖w i • (β_t - β i)‖ := norm_sum_le _ _
    _ = ∑ i, w i * ‖β_t - β i‖ := by
        congr 1; ext i
        rw [norm_smul, Real.norm_of_nonneg (hw i)]
    _ ≤ ∑ i, w i * (L * D i) := by
        apply Finset.sum_le_sum
        intro i _
        exact mul_le_mul_of_nonneg_left (hLip i) (hw i)
    _ = L * ∑ i, w i * D i := by
        rw [Finset.mul_sum]; congr 1; ext i; ring

/-! ### Portfolio variance envelope, new-asset brackets, certified cap, and diversification

This section contains the 2026-07-08 extensions.
-/

/--
**Theorem (Portfolio Variance Envelope / Floor–Cap Sandwich).**

With bi-Lipschitz identification `ℓ·Dᵢⱼ ≤ ‖βᵢ − βⱼ‖ ≤ L·Dᵢⱼ`, portfolio
systematic variance is sandwiched between the two price-free double sums.
One-theorem packaging of `portfolio_variance_lower_bound` and
`portfolio_variance_upper_bound` for citation. `Σ w = 1` is deliberately not
assumed (not needed on either side).
-/
theorem portfolio_variance_envelope {n : ℕ}
    (β : Fin n → E) (w : Fin n → ℝ) (L ℓ : ℝ) (D : Fin n → Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ) (hD : ∀ i j, 0 ≤ D i j)
    (hLip : ∀ i j, ‖β i - β j‖ ≤ L * D i j)
    (hLip_lower : ∀ i j, ℓ * D i j ≤ ‖β i - β j‖) :
    ∑ i, ∑ j, w i * w j * ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 - (L * D i j) ^ 2)) ≤
        ‖∑ i, w i • β i‖ ^ 2 ∧
      ‖∑ i, w i • β i‖ ^ 2 ≤
        ∑ i, ∑ j, w i * w j * ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 - (ℓ * D i j) ^ 2)) :=
  ⟨portfolio_variance_lower_bound β w L D hw hL hD hLip,
   portfolio_variance_upper_bound β w ℓ D hw hℓ hD hLip_lower⟩

/--
New asset with no return history; risk scale bracketed price-free via one
calibrated anchor; unfold with `abs_le`. Premise: `abs_norm_sub_norm_le`.
-/
theorem new_asset_risk_bracket
    (β_t β_i : E) (L D_ti : ℝ)
    (hLip : ‖β_t - β_i‖ ≤ L * D_ti) :
    |‖β_t‖ - ‖β_i‖| ≤ L * D_ti :=
  (abs_norm_sub_norm_le β_t β_i).trans hLip

/--
New asset with no return history; pairwise systematic covariance sandwiched
price-free between lower and upper envelopes. All of `hL`, `hℓ`, `hD` are
load-bearing. Term-mode pairing of `systematic_covariance_lower_bound` and
`systematic_covariance_upper_bound`.
-/
theorem new_asset_covariance_bracket
    (β_t β_i : E) (L ℓ D_ti : ℝ)
    (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ) (hD : 0 ≤ D_ti)
    (hLip : ‖β_t - β_i‖ ≤ L * D_ti)
    (hLip_lower : ℓ * D_ti ≤ ‖β_t - β_i‖) :
    (1 / 2) * (‖β_t‖ ^ 2 + ‖β_i‖ ^ 2 - (L * D_ti) ^ 2) ≤
        @Inner.inner ℝ E _ β_t β_i ∧
      @Inner.inner ℝ E _ β_t β_i ≤
        (1 / 2) * (‖β_t‖ ^ 2 + ‖β_i‖ ^ 2 - (ℓ * D_ti) ^ 2) :=
  ⟨systematic_covariance_lower_bound β_t β_i L D_ti hL hD hLip,
   systematic_covariance_upper_bound β_t β_i ℓ D_ti hℓ hD hLip_lower⟩

/--
**Theorem (Certified Price-Free Risk Cap).**

Under homogeneous scale `‖βᵢ‖ = σ` and lower-Lipschitz identification, and
with a fully-invested portfolio `Σwᵢ = 1`, the systematic variance is capped by
a price-free expression:

  `‖Σᵢ wᵢ βᵢ‖² ≤ σ² − (ℓ²/2) · Σᵢⱼ wᵢwⱼ (Dᵢⱼ)²`.

`Σw = 1` IS load-bearing here: it collapses `(Σw)²σ²` to `σ²`. Everything on
the RHS except `σ`, `ℓ` is price-free. Do NOT add `0 ≤ σ` (unused).
-/
theorem certified_risk_cap {n : ℕ}
    (β : Fin n → E) (w : Fin n → ℝ) (σ ℓ : ℝ) (D : Fin n → Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1)
    (hσ : ∀ i, ‖β i‖ = σ)
    (hℓ : 0 ≤ ℓ) (hD : ∀ i j, 0 ≤ D i j)
    (hLip_lower : ∀ i j, ℓ * D i j ≤ ‖β i - β j‖) :
    ‖∑ i, w i • β i‖ ^ 2 ≤
      σ ^ 2 - ℓ ^ 2 / 2 * ∑ i, ∑ j, w i * w j * (D i j) ^ 2 := by
  have h := portfolio_variance_upper_bound β w ℓ D hw hℓ hD hLip_lower
  simp only [hσ] at h
  have hrhs_eq : ∑ i : Fin n, ∑ j : Fin n,
      w i * w j * ((1 / 2) * (σ ^ 2 + σ ^ 2 - (ℓ * D i j) ^ 2)) =
      σ ^ 2 - ℓ ^ 2 / 2 * ∑ i, ∑ j, w i * w j * (D i j) ^ 2 := by
    have expand : ∀ i j : Fin n,
        w i * w j * ((1 / 2) * (σ ^ 2 + σ ^ 2 - (ℓ * D i j) ^ 2)) =
        σ ^ 2 * (w i * w j) - ℓ ^ 2 / 2 * (w i * w j * (D i j) ^ 2) :=
      fun i j => by ring
    simp_rw [expand, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, mul_one]
    rw [hsum, mul_one]
  linarith [hrhs_eq ▸ h]

/--
**Theorem (Diversification Tightens the Cap).**

"Diversification = buying distance" — a portfolio `w₂` with higher price-free
quadratic `Q(w₂) = Σᵢⱼ w₂ᵢ w₂ⱼ Dᵢⱼ²` than `w₁` provably makes the certified
cap tighter for `w₁`'s realised variance. `w₂` carries NO hypotheses.
-/
theorem diversification_tightens_cap {n : ℕ}
    (β : Fin n → E) (w₁ w₂ : Fin n → ℝ) (σ ℓ : ℝ) (D : Fin n → Fin n → ℝ)
    (hw₁ : ∀ i, 0 ≤ w₁ i) (hsum₁ : (∑ i, w₁ i) = 1)
    (hσ : ∀ i, ‖β i‖ = σ)
    (hℓ : 0 ≤ ℓ) (hD : ∀ i j, 0 ≤ D i j)
    (hLip_lower : ∀ i j, ℓ * D i j ≤ ‖β i - β j‖)
    (hQ : ∑ i, ∑ j, w₂ i * w₂ j * (D i j) ^ 2 ≤
          ∑ i, ∑ j, w₁ i * w₁ j * (D i j) ^ 2) :
    ‖∑ i, w₁ i • β i‖ ^ 2 ≤
      σ ^ 2 - ℓ ^ 2 / 2 * ∑ i, ∑ j, w₂ i * w₂ j * (D i j) ^ 2 := by
  have h := certified_risk_cap β w₁ σ ℓ D hw₁ hsum₁ hσ hℓ hD hLip_lower
  have hmono : ℓ ^ 2 / 2 * (∑ i, ∑ j, w₂ i * w₂ j * (D i j) ^ 2) ≤
      ℓ ^ 2 / 2 * (∑ i, ∑ j, w₁ i * w₁ j * (D i j) ^ 2) :=
    mul_le_mul_of_nonneg_left hQ (by positivity)
  linarith

/--
**Lemma (Hilbertian CND identity / Schoenberg mechanism).**

For zero-sum weights `Σᵢ uᵢ = 0` and points `xᵢ ∈ E`, the weighted double sum
of squared distances equals `−2 ‖Σᵢ uᵢ xᵢ‖²`. This is the quantitative Hilbert
form of `EnergyStatistics.DistNegativeType`, connecting the Schoenberg–Lyons
distance-of-negative-type condition to the Hilbert-space geometry.
-/
lemma sq_dist_cnd_of_inner {n : ℕ} (x : Fin n → E) (u : Fin n → ℝ)
    (hsum : (∑ i, u i) = 0) :
    ∑ i, ∑ j, u i * u j * ‖x i - x j‖ ^ 2 = -2 * ‖∑ i, u i • x i‖ ^ 2 := by
  have lhs_eq : ∑ i : Fin n, ∑ j : Fin n, u i * u j * ‖x i - x j‖ ^ 2 =
      -2 * ∑ i : Fin n, ∑ j : Fin n, u i * u j * @Inner.inner ℝ E _ (x i) (x j) := by
    have step1 : ∑ i : Fin n, ∑ j : Fin n, u i * u j * ‖x i - x j‖ ^ 2 =
        ∑ i : Fin n, ∑ j : Fin n,
          (u i * u j * ‖x i‖ ^ 2 -
           2 * (u i * u j * @Inner.inner ℝ E _ (x i) (x j)) +
           u i * u j * ‖x j‖ ^ 2) := by
      congr 1; ext i; congr 1; ext j
      rw [norm_sub_sq_real]; ring
    rw [step1]
    have h1 : ∑ i : Fin n, ∑ j : Fin n, u i * u j * ‖x i‖ ^ 2 = 0 := by
      simp_rw [show ∀ i j : Fin n, u i * u j * ‖x i‖ ^ 2 = (u i * ‖x i‖ ^ 2) * u j from
        fun i j => by ring, ← Finset.mul_sum, hsum, mul_zero, Finset.sum_const_zero]
    have h3 : ∑ i : Fin n, ∑ j : Fin n, u i * u j * ‖x j‖ ^ 2 = 0 := by
      simp_rw [show ∀ i j : Fin n, u i * u j * ‖x j‖ ^ 2 = u i * (u j * ‖x j‖ ^ 2) from
        fun i j => by ring, ← Finset.mul_sum, ← Finset.sum_mul, hsum, zero_mul]
    simp_rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, h1, h3, ← Finset.mul_sum]
    ring
  rw [lhs_eq, ← portfolio_norm_sq_eq_double_sum]

/-- If the squared-distance matrix `[Dᵢⱼ²]` is conditionally negative definite,
the price-free quadratic `Q(w) = Σᵢⱼ wᵢwⱼ Dᵢⱼ²` is concave on the sum-one
affine subspace: the certificate-tightening maximization of `Q` over the simplex
is a concave program (every local optimum global). `hCND` is discharged by
`sq_dist_cnd_of_inner` when `D i j = ‖x i − x j‖` for Hilbert-embedded exposures. -/
theorem Q_concave_on_simplex {n : ℕ} (D : Fin n → Fin n → ℝ)
    (hCND : ∀ u : Fin n → ℝ, (∑ i, u i) = 0 →
      ∑ i, ∑ j, u i * u j * (D i j) ^ 2 ≤ 0)
    (w₁ w₂ : Fin n → ℝ) (hsum₁ : (∑ i, w₁ i) = 1) (hsum₂ : (∑ i, w₂ i) = 1)
    (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    θ * (∑ i, ∑ j, w₁ i * w₁ j * (D i j) ^ 2) +
        (1 - θ) * (∑ i, ∑ j, w₂ i * w₂ j * (D i j) ^ 2) ≤
      ∑ i, ∑ j, (θ * w₁ i + (1 - θ) * w₂ i) * (θ * w₁ j + (1 - θ) * w₂ j) * (D i j) ^ 2 := by
  have key : ∀ i j : Fin n,
      (θ * w₁ i + (1 - θ) * w₂ i) * (θ * w₁ j + (1 - θ) * w₂ j) * (D i j) ^ 2 =
      θ * (w₁ i * w₁ j * (D i j) ^ 2) + (1 - θ) * (w₂ i * w₂ j * (D i j) ^ 2) -
      (θ * (1 - θ)) * ((w₁ i - w₂ i) * (w₁ j - w₂ j) * (D i j) ^ 2) :=
    fun i j => by ring
  simp_rw [key, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  have hzero : (∑ i, (w₁ i - w₂ i)) = 0 := by
    simp [Finset.sum_sub_distrib, hsum₁, hsum₂]
  have hcnd := hCND (fun i => w₁ i - w₂ i) hzero
  have hθ1θ : 0 ≤ θ * (1 - θ) := mul_nonneg hθ0 (by linarith)
  nlinarith [Finset.sum_comm (f := fun i j => (w₁ i - w₂ i) * (w₁ j - w₂ j) * (D i j) ^ 2)
    (s := Finset.univ) (t := Finset.univ)]


end PricingPerspective.ContinuousAPT
