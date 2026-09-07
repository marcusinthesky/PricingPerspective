import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Matrix.Normed
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# Spatial asset pricing (s-CAPM / s-APT): structural spine

Machine-checked finite-dimensional core of the spatial-lag asset-pricing literature, for the
Paper 5 program (energy-distance-implied spatial weights).

Sources (`.context/reference/` keys):
* `kou_asset_2018` — Kou, Peng & Zhong (2018), *Asset Pricing with Spatial Interaction*,
  Management Science 64(5): spatial-lag model (Eq. 6), spatial covariance (Eq. 7), S-CAPM
  (Theorem 1) and the zero-intercept restriction (Eq. 15).
* `fernandez_spatial_2011` — Fernandez (2011), *Spatial linkages in international financial
  markets*, Quantitative Finance 11(2): spatial-lag VaR variance decomposition (Eq. 8a).
* `gao2025high` — Gao, Tu & Tsay (2025), arXiv:2511.01271: heterogeneous spatial interactions;
  their amplification remark is machine-checked here as `one_le_leontief_entrywise`.
* `ge_news-implied_2023` — Ge, Li & Linton (2023), J. Econometrics 235(2): Leontief-inverse
  ("weak dependence") structure of the spatial component.

Contents:
* Generic complete-normed-ring lemmas: geometric-series norm bound on `Ring.inverse (1 - x)`
  (`norm_inverse_one_sub_le`) and a resolvent perturbation bound
  (`norm_inverse_one_sub_sub_le`) — the new rung feeding the energy-robustness ambiguity brackets.
* `leontief ρ W = (1 - ρ • W)⁻¹`: invertibility under `‖ρ • W‖ < 1` (Neumann), two-sided
  inverse identities, and the geometric-series form `leontief_eq_tsum`.
* `reduced_form_of_spatial_lag`: per-realization reduced form `r = (1 - ρW)⁻¹ x` of the
  spatial-lag model `r = ρ W r + x` (Kou Eq. 6).
* `spatialCov ρ W V = leontief ρ W * V * (leontief ρ W)ᵀ` (Kou Eq. 7): positive
  semidefiniteness (congruence), the one-factor Fernandez form (`spatialCov_factor_model`),
  and the portfolio-variance split (`portfolio_variance_decomposition`).
* S-CAPM: `beta_pricing_of_tangency` (algebraic core of Kou Theorem 1) and
  `spatial_intercept_eq_zero` (Kou Eq. 15 — the empirically tested restriction ᾱ = 0).
* Amplification: entrywise `1 ≤ leontief ρ W` for entrywise-nonnegative `W` and `ρ ≥ 0`, and
  amplification of effective factor loadings (`le_leontief_mulVec_of_nonneg`).
* Perturbation brackets: `leontief_sub_norm_le`, `spatialCov_sub_norm_le`,
  `spatialCov_entry_bracket` — certified entrywise covariance brackets under
  spatial-parameter ambiguity (new; the spatial ↔ energy-robustness interface).

All matrix norms below are the `L∞` operator norm (maximum absolute row sum), activated via
`open scoped Matrix.Norms.Operator`. The probabilistic identification `spatialCov = Cov(r)`
(Kou Eq. 7) is definitional here; the per-realization reduced form is machine-checked, the
lift to second moments is prose (same discipline as the `EnergyKernelEmbedding` layer).
-/

namespace PricingPerspective.Discrete.Spatial

open Matrix
open scoped Matrix.Norms.Operator

/-! ## Generic complete-normed-ring lemmas -/

section NormedRing

variable {R : Type*} [NormedRing R] [NormOneClass R] [CompleteSpace R]

/-- Geometric-series bound on the resolvent: if `‖x‖ < 1` then
`‖(1 - x)⁻¹‖ ≤ (1 - ‖x‖)⁻¹`, stated via `Ring.inverse`. -/
theorem norm_inverse_one_sub_le {x : R} (hx : ‖x‖ < 1) :
    ‖Ring.inverse (1 - x)‖ ≤ (1 - ‖x‖)⁻¹ := by
  rw [← geom_series_eq_inverse x hx]
  refine le_trans (norm_tsum_le_tsum_norm ?_) ?_
  · -- Summable (fun i => ‖x ^ i‖)
    refine Summable.of_nonneg_of_le (fun i => norm_nonneg _) (fun i => norm_pow_le x i) ?_
    exact summable_geometric_of_lt_one (norm_nonneg _) hx
  · -- ∑' i, ‖x ^ i‖ ≤ ∑' i, ‖x‖ ^ i = (1 - ‖x‖)⁻¹
    refine le_trans (Summable.tsum_le_tsum (fun i => norm_pow_le x i) ?_ ?_) ?_
    · exact Summable.of_nonneg_of_le (fun i => norm_nonneg _) (fun i => norm_pow_le x i)
        (summable_geometric_of_lt_one (norm_nonneg _) hx)
    · exact summable_geometric_of_lt_one (norm_nonneg _) hx
    · rw [tsum_geometric_of_lt_one (norm_nonneg _) hx]

/-- Resolvent perturbation bound (new rung for the Paper 5 energy-robustness T-ladder): for `‖x‖ < 1`,
`‖y‖ < 1`,
`‖(1 - x)⁻¹ - (1 - y)⁻¹‖ ≤ ‖x - y‖ / ((1 - ‖x‖) (1 - ‖y‖))`.
Proof route: resolvent identity `u⁻¹ - v⁻¹ = u⁻¹ (v - u) v⁻¹` plus
`norm_inverse_one_sub_le`. -/
theorem norm_inverse_one_sub_sub_le {x y : R} (hx : ‖x‖ < 1) (hy : ‖y‖ < 1) :
    ‖Ring.inverse (1 - x) - Ring.inverse (1 - y)‖ ≤ ‖x - y‖ / ((1 - ‖x‖) * (1 - ‖y‖)) := by
  have hux : IsUnit (1 - x) := isUnit_one_sub_of_norm_lt_one hx
  have huy : IsUnit (1 - y) := isUnit_one_sub_of_norm_lt_one hy
  have hpos_x : 0 < 1 - ‖x‖ := sub_pos.mpr hx
  have hpos_y : 0 < 1 - ‖y‖ := sub_pos.mpr hy
  calc
    ‖Ring.inverse (1 - x) - Ring.inverse (1 - y)‖
        = ‖Ring.inverse (1 - x) * ((1 - y) - (1 - x)) * Ring.inverse (1 - y)‖ := by
      have h_eq : Ring.inverse (1 - x) - Ring.inverse (1 - y) =
          Ring.inverse (1 - x) * ((1 - y) - (1 - x)) * Ring.inverse (1 - y) := by
        calc
          Ring.inverse (1 - x) - Ring.inverse (1 - y)
              = Ring.inverse (1 - x) * 1 - 1 * Ring.inverse (1 - y) := by
            simp
          _ = Ring.inverse (1 - x) * ((1 - y) * Ring.inverse (1 - y))
              - (Ring.inverse (1 - x) * (1 - x)) * Ring.inverse (1 - y) := by
            rw [Ring.mul_inverse_cancel (1 - y) huy, Ring.inverse_mul_cancel (1 - x) hux]
          _ = Ring.inverse (1 - x) * ((1 - y) - (1 - x)) * Ring.inverse (1 - y) := by
            simp [mul_sub, sub_mul, mul_assoc]
      rw [h_eq]
    _ ≤ ‖Ring.inverse (1 - x)‖ * ‖(1 - y) - (1 - x)‖ * ‖Ring.inverse (1 - y)‖ := by
      calc
        ‖Ring.inverse (1 - x) * ((1 - y) - (1 - x)) * Ring.inverse (1 - y)‖
            ≤ ‖Ring.inverse (1 - x) * ((1 - y) - (1 - x))‖ * ‖Ring.inverse (1 - y)‖ :=
          norm_mul_le _ _
        _ ≤ (‖Ring.inverse (1 - x)‖ * ‖(1 - y) - (1 - x)‖) * ‖Ring.inverse (1 - y)‖ := by
          gcongr; exact norm_mul_le _ _
        _ = ‖Ring.inverse (1 - x)‖ * ‖(1 - y) - (1 - x)‖ * ‖Ring.inverse (1 - y)‖ := by ring
    _ = ‖Ring.inverse (1 - x)‖ * ‖x - y‖ * ‖Ring.inverse (1 - y)‖ := by
      simp [sub_sub_sub_cancel_left]
    _ ≤ (1 - ‖x‖)⁻¹ * ‖x - y‖ * (1 - ‖y‖)⁻¹ := by
      gcongr
      · exact norm_inverse_one_sub_le hx
      · exact norm_inverse_one_sub_le hy
    _ = ‖x - y‖ * ((1 - ‖x‖)⁻¹ * (1 - ‖y‖)⁻¹) := by ring
    _ = ‖x - y‖ * (((1 - ‖x‖) * (1 - ‖y‖))⁻¹) := by rw [mul_inv]
    _ = ‖x - y‖ / ((1 - ‖x‖) * (1 - ‖y‖)) := by rw [div_eq_mul_inv]

end NormedRing

/-! ## The Leontief inverse and the spatial-lag model -/

section SpatialModel

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The Leontief inverse `(1 - ρ • W)⁻¹` of a spatial weight matrix `W` at spatial
autocorrelation `ρ` (Kou Eq. 7; Ge–Li–Linton's `G(Ψ)`). Uses `Matrix.inv`, so it is junk
unless `1 - ρ • W` is invertible; all substantive results below assume `‖ρ • W‖ < 1`. -/
noncomputable def leontief (ρ : ℝ) (W : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  (1 - ρ • W)⁻¹

variable {ρ ρ' : ℝ} {W V : Matrix ι ι ℝ}

/-- Invertibility of `1 - ρ • W` under the spectral-domination condition `‖ρ • W‖ < 1`
(the sufficient form of Kou's "ρ⁻¹ is not an eigenvalue of W"). -/
theorem isUnit_one_sub_smul (h : ‖ρ • W‖ < 1) : IsUnit (1 - ρ • W) :=
  isUnit_one_sub_of_norm_lt_one h

/-- `(1 - ρ • W) * leontief ρ W = 1` under `‖ρ • W‖ < 1`. -/
theorem one_sub_smul_mul_leontief (h : ‖ρ • W‖ < 1) :
    (1 - ρ • W) * leontief ρ W = 1 := by
  have h_unit : IsUnit ((1 - ρ • W : Matrix ι ι ℝ)) := isUnit_one_sub_smul h
  have h_det : IsUnit ((1 - ρ • W).det) := (Matrix.isUnit_iff_isUnit_det _).mp h_unit
  unfold leontief
  exact Matrix.mul_nonsing_inv _ h_det

/-- `leontief ρ W * (1 - ρ • W) = 1` under `‖ρ • W‖ < 1`. -/
theorem leontief_mul_one_sub_smul (h : ‖ρ • W‖ < 1) :
    leontief ρ W * (1 - ρ • W) = 1 := by
  have h_unit : IsUnit ((1 - ρ • W : Matrix ι ι ℝ)) := isUnit_one_sub_smul h
  have h_det : IsUnit ((1 - ρ • W).det) := (Matrix.isUnit_iff_isUnit_det _).mp h_unit
  unfold leontief
  exact Matrix.nonsing_inv_mul _ h_det

/-- Neumann-series form of the Leontief inverse: `(1 - ρ • W)⁻¹ = ∑ₖ (ρ • W)ᵏ`.
This is the structural lemma behind every spatial "network multiplier" interpretation. -/
theorem leontief_eq_tsum (h : ‖ρ • W‖ < 1) :
    leontief ρ W = ∑' k : ℕ, (ρ • W) ^ k := by
  unfold leontief
  rw [Matrix.nonsing_inv_eq_ringInverse]
  apply (geom_series_eq_inverse (ρ • W) h).symm

/-- Per-realization reduced form of the spatial-lag model (Kou Eq. 6): if
`r = ρ • (W *ᵥ r) + x` then `r = (1 - ρ • W)⁻¹ *ᵥ x`. -/
theorem reduced_form_of_spatial_lag (h : ‖ρ • W‖ < 1) {r x : ι → ℝ}
    (hr : r = ρ • (W *ᵥ r) + x) : r = leontief ρ W *ᵥ x := by
  have h_mul : leontief ρ W * (1 - ρ • W) = 1 := leontief_mul_one_sub_smul h
  have h_key : (1 - ρ • W) *ᵥ r = x := by
    calc
      (1 - ρ • W) *ᵥ r = (1 : Matrix ι ι ℝ) *ᵥ r - (ρ • W) *ᵥ r := by rw [Matrix.sub_mulVec]
      _ = r - (ρ • W) *ᵥ r := by rw [Matrix.one_mulVec]
      _ = r - ρ • (W *ᵥ r) := by rw [Matrix.smul_mulVec]
      _ = x := sub_eq_of_eq_add (hr.trans (add_comm _ _))
  calc
    r = (1 : Matrix ι ι ℝ) *ᵥ r := by simp
    _ = (leontief ρ W * (1 - ρ • W)) *ᵥ r := by rw [h_mul]
    _ = leontief ρ W *ᵥ ((1 - ρ • W) *ᵥ r) := by rw [Matrix.mulVec_mulVec]
    _ = leontief ρ W *ᵥ x := by rw [h_key]

/-! ## Spatial covariance (Kou Eq. 7) -/

/-- The spatial covariance implied by the spatial-lag model with innovation covariance `V`:
`Σ = (1 - ρW)⁻¹ V ((1 - ρW)⁻¹)ᵀ` (Kou Eq. 7; Fernandez Eq. 8a). -/
noncomputable def spatialCov (ρ : ℝ) (W V : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  leontief ρ W * V * (leontief ρ W)ᵀ

/-- Congruence preserves positive semidefiniteness: `spatialCov ρ W V` is PSD whenever the
innovation covariance `V` is. No invertibility hypothesis is needed. -/
theorem spatialCov_posSemidef (hV : V.PosSemidef) : (spatialCov ρ W V).PosSemidef := by
  unfold spatialCov
  have h := hV.mul_mul_conjTranspose_same (leontief ρ W)
  rwa [conjTranspose_eq_transpose_of_trivial] at h

omit [DecidableEq ι] in
/-- Sandwich identity for rank-one blocks: `A (u vᵀ) Bᵀ = (A u)(B v)ᵀ`. Proved by transposing
to reuse `Matrix.mul_vecMulVec` on both sides. -/
private theorem mul_vecMulVec_mul_transpose (A B : Matrix ι ι ℝ) (u v : ι → ℝ) :
    A * vecMulVec u v * Bᵀ = vecMulVec (A *ᵥ u) (B *ᵥ v) := by
  rw [Matrix.mul_vecMulVec]
  refine Matrix.transpose_injective ?_
  rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_vecMulVec,
    Matrix.mul_vecMulVec, Matrix.transpose_vecMulVec]

/-- Fernandez Eq. (8a): for a one-factor innovation covariance
`V = σm² β βᵀ + σu² I`, the spatial covariance splits into a spatially transformed factor
term and a spatially amplified idiosyncratic term. -/
theorem spatialCov_factor_model (b : ι → ℝ) (σm σu : ℝ) :
    spatialCov ρ W (σm ^ 2 • vecMulVec b b + σu ^ 2 • (1 : Matrix ι ι ℝ)) =
      σm ^ 2 • vecMulVec (leontief ρ W *ᵥ b) (leontief ρ W *ᵥ b) +
        σu ^ 2 • (leontief ρ W * (leontief ρ W)ᵀ) := by
  unfold spatialCov
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one]
  rw [mul_vecMulVec_mul_transpose]

/-- Portfolio-variance split under a one-factor spatial covariance (Fernandez Eq. 8a inside
her VaR formula): for any transformation `A` (instantiate `A := leontief ρ W`),
`ω' (σm² (Ab)(Ab)ᵀ + σu² A Aᵀ) ω = σm² ⟨ω, Ab⟩² + σu² ‖Aᵀ ω‖²`. -/
theorem portfolio_variance_decomposition (A : Matrix ι ι ℝ) (ω b : ι → ℝ) (σm σu : ℝ) :
    ω ⬝ᵥ ((σm ^ 2 • vecMulVec (A *ᵥ b) (A *ᵥ b) + σu ^ 2 • (A * Aᵀ)) *ᵥ ω) =
      σm ^ 2 * (ω ⬝ᵥ (A *ᵥ b)) ^ 2 + σu ^ 2 * ((Aᵀ *ᵥ ω) ⬝ᵥ (Aᵀ *ᵥ ω)) := by
  have _hdec : DecidableEq ι := ‹DecidableEq ι›
  have ha : ω ⬝ᵥ (vecMulVec (A *ᵥ b) (A *ᵥ b) *ᵥ ω) = (ω ⬝ᵥ (A *ᵥ b)) ^ 2 := by
    rw [Matrix.vecMulVec_mulVec]
    simp only [op_smul_eq_smul, dotProduct_smul, smul_eq_mul]
    rw [dotProduct_comm (A *ᵥ b) ω]
    ring
  have hb : ω ⬝ᵥ ((A * Aᵀ) *ᵥ ω) = (Aᵀ *ᵥ ω) ⬝ᵥ (Aᵀ *ᵥ ω) := by
    rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose]
  rw [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.smul_mulVec, dotProduct_add,
    dotProduct_smul, dotProduct_smul, smul_eq_mul, smul_eq_mul, ha, hb]

/-! ## S-CAPM (Kou Theorem 1 and Eq. 15) -/

omit [DecidableEq ι] in
/-- Algebraic core of the S-CAPM (Kou Theorem 1): if the market portfolio `wM` is
mean–variance efficient — first-order condition `S *ᵥ wM = γ • (μ - rf)` for some `γ ≠ 0` —
its weights sum to one, and market variance is nonzero, then every asset's excess return is
its beta times the market excess return. The spatial content enters through
`S = spatialCov ρ W V`. -/
theorem beta_pricing_of_tangency (S : Matrix ι ι ℝ) (μ wM : ι → ℝ) (rf γ : ℝ)
    (hFOC : S *ᵥ wM = γ • (fun i ↦ μ i - rf)) (hγ : γ ≠ 0)
    (hvar : wM ⬝ᵥ (S *ᵥ wM) ≠ 0) (hsum : ∑ i, wM i = 1) (i : ι) :
    μ i - rf = (S *ᵥ wM) i / (wM ⬝ᵥ (S *ᵥ wM)) * ((∑ j, wM j * μ j) - rf) := by
  -- Step a: entrywise FOC
  have h_entry : (S *ᵥ wM) i = γ * (μ i - rf) := by
    calc
      (S *ᵥ wM) i = (γ • (fun i ↦ μ i - rf)) i := by rw [hFOC]
      _ = γ * (μ i - rf) := by rw [Pi.smul_apply, smul_eq_mul]
  -- Step b: market variance
  have h_var : wM ⬝ᵥ (S *ᵥ wM) = γ * ((∑ j, wM j * μ j) - rf) := by
    rw [hFOC]
    rw [dotProduct_smul, smul_eq_mul]
    congr 1
    calc
      wM ⬝ᵥ (fun i ↦ μ i - rf) = ∑ i, wM i * (μ i - rf) := rfl
      _ = ∑ i, (wM i * μ i - wM i * rf) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [mul_sub]
      _ = (∑ i, wM i * μ i) - (∑ i, wM i * rf) := by rw [Finset.sum_sub_distrib]
      _ = (∑ i, wM i * μ i) - rf * (∑ i, wM i) := by
        rw [← Finset.sum_mul, mul_comm]
      _ = (∑ i, wM i * μ i) - rf * 1 := by rw [hsum]
      _ = (∑ i, wM i * μ i) - rf := by ring
      _ = (∑ j, wM j * μ j) - rf := by simp
  -- Step c: nonvanishing
  have hM : (∑ j, wM j * μ j) - rf ≠ 0 := by
    intro h0
    apply hvar
    rw [h_var, h0, mul_zero]
  -- Step d: conclude
  calc
    μ i - rf = (γ * (μ i - rf)) / γ := by field_simp [hγ]
    _ = (S *ᵥ wM) i / γ := by rw [h_entry]
    _ = (S *ᵥ wM) i / (γ * ((∑ j, wM j * μ j) - rf)) * ((∑ j, wM j * μ j) - rf) := by
      field_simp [hM]
    _ = (S *ᵥ wM) i / (wM ⬝ᵥ (S *ᵥ wM)) * ((∑ j, wM j * μ j) - rf) := by rw [h_var]

/-- Kou Eq. (15): under the tangency condition, the intercept of the *spatial* excess-return
regression vanishes: `(1 - ρW) *ᵥ ᾱ = 0` where `ᾱ` is the vector of CAPM pricing errors.
This is the restriction tested empirically in the S-CAPM literature. -/
theorem spatial_intercept_eq_zero (S : Matrix ι ι ℝ) (μ wM : ι → ℝ) (rf γ : ℝ)
    (hFOC : S *ᵥ wM = γ • (fun i ↦ μ i - rf)) (hγ : γ ≠ 0)
    (hvar : wM ⬝ᵥ (S *ᵥ wM) ≠ 0) (hsum : ∑ i, wM i = 1) :
    (1 - ρ • W) *ᵥ
        (fun i ↦ (μ i - rf) - (S *ᵥ wM) i / (wM ⬝ᵥ (S *ᵥ wM)) * ((∑ j, wM j * μ j) - rf)) =
      0 := by
  have hz : (fun i ↦ (μ i - rf) - (S *ᵥ wM) i / (wM ⬝ᵥ (S *ᵥ wM)) * ((∑ j, wM j * μ j) - rf)) = (0 : ι → ℝ) := by
    ext i
    exact sub_eq_zero.mpr (beta_pricing_of_tangency S μ wM rf γ hFOC hγ hvar hsum i)
  rw [hz, Matrix.mulVec_zero]

/-! ## Spatial amplification (Kou §5 remark; Gao–Tu–Tsay §2) -/

/-- Amplification, matrix form: for entrywise-nonnegative `W` and `ρ ≥ 0` with
`‖ρ • W‖ < 1`, the Leontief inverse dominates the identity entrywise
(`(1 - ρW)⁻¹ = 1 + ρW + (ρW)² + ⋯ ≥ 1`). -/
private lemma pow_apply_nonneg (hW : ∀ i j, 0 ≤ W i j) (hρ : 0 ≤ ρ) (k : ℕ) (i j : ι) :
    0 ≤ ((ρ • W) ^ k) i j := by
  induction k generalizing i j with
  | zero =>
    rw [pow_zero, Matrix.one_apply]
    split <;> norm_num
  | succ k ih =>
    rw [pow_succ, Matrix.mul_apply]
    refine Finset.sum_nonneg fun l _ => ?_
    rw [Matrix.smul_apply]
    have h_mul : 0 ≤ ρ * W l j := mul_nonneg hρ (hW l j)
    have h_pow : 0 ≤ ((ρ • W) ^ k) i l := ih i l
    exact mul_nonneg h_pow h_mul

theorem one_le_leontief_entrywise (h : ‖ρ • W‖ < 1) (hW : ∀ i j, 0 ≤ W i j) (hρ : 0 ≤ ρ)
    (i j : ι) : (1 : Matrix ι ι ℝ) i j ≤ leontief ρ W i j := by
  rw [leontief_eq_tsum h]
  -- Goal: (1 : Matrix ι ι ℝ) i j ≤ (∑' k : ℕ, (ρ • W) ^ k) i j
  let φ : (Matrix ι ι ℝ) →L[ℝ] ℝ :=
    { Matrix.entryLinearMap ℝ ℝ i j with
      cont := LinearMap.continuous_of_finiteDimensional _ }
  have h_summable : Summable (fun k : ℕ => (ρ • W) ^ k) :=
    summable_geometric_of_norm_lt_one h
  have h_summable_entry : Summable (fun k : ℕ => ((ρ • W) ^ k) i j) :=
    ContinuousLinearMap.summable φ h_summable
  have h_tsum_entry : (∑' k : ℕ, (ρ • W) ^ k) i j = ∑' k : ℕ, ((ρ • W) ^ k) i j := by
    calc
      (∑' k : ℕ, (ρ • W) ^ k) i j = φ (∑' k : ℕ, (ρ • W) ^ k) := rfl
      _ = ∑' k : ℕ, φ ((ρ • W) ^ k) := ContinuousLinearMap.map_tsum φ h_summable
      _ = ∑' k : ℕ, ((ρ • W) ^ k) i j := rfl
  rw [h_tsum_entry]
  have h_one_le_pow0 : (1 : Matrix ι ι ℝ) i j ≤ ((ρ • W) ^ 0) i j := by
    rw [pow_zero]
  have h_le_tsum : ((ρ • W) ^ 0) i j ≤ ∑' k : ℕ, ((ρ • W) ^ k) i j :=
    h_summable_entry.le_tsum 0 (fun k hk => pow_apply_nonneg hW hρ k i j)
  exact le_trans h_one_le_pow0 h_le_tsum

/-- Amplification of effective factor loadings: for nonnegative loadings `b`, the effective
(spatially propagated) loadings `(1 - ρW)⁻¹ b` dominate `b` entrywise. Machine-checks the
"spatial proximity amplifies systematic risk" remark of Kou et al. and Gao–Tu–Tsay. -/
theorem le_leontief_mulVec_of_nonneg (h : ‖ρ • W‖ < 1) (hW : ∀ i j, 0 ≤ W i j) (hρ : 0 ≤ ρ)
    {b : ι → ℝ} (hb : ∀ i, 0 ≤ b i) (i : ι) : b i ≤ (leontief ρ W *ᵥ b) i := by
  have h_one_mulVec : b i = ((1 : Matrix ι ι ℝ) *ᵥ b) i := by
    rw [Matrix.one_mulVec]
  rw [h_one_mulVec]
  simp [Matrix.mulVec]
  refine Finset.sum_le_sum fun j _ => ?_
  have h_entry : (1 : Matrix ι ι ℝ) i j ≤ leontief ρ W i j :=
    one_le_leontief_entrywise h hW hρ i j
  exact mul_le_mul_of_nonneg_right h_entry (hb j)

/-! ## Perturbation brackets (spatial ↔ energy-robustness interface) -/

/-- Entry bound for the `L∞` operator norm: `|M i j| ≤ ‖M‖`
(each entry is dominated by its row's absolute sum). -/
theorem abs_entry_le_norm (M : Matrix ι ι ℝ) (i j : ι) : |M i j| ≤ ‖M‖ := by
  have hnn : ‖M i j‖₊ ≤ ‖M‖₊ := by
    rw [linfty_opNNNorm_def]
    calc ‖M i j‖₊ ≤ ∑ k, ‖M i k‖₊ :=
          Finset.single_le_sum (f := fun k ↦ ‖M i k‖₊) (fun k _ ↦ zero_le) (Finset.mem_univ j)
      _ ≤ Finset.univ.sup fun i ↦ ∑ k, ‖M i k‖₊ :=
          Finset.le_sup (f := fun i ↦ ∑ k, ‖M i k‖₊) (Finset.mem_univ i)
  calc |M i j| = ‖M i j‖ := (Real.norm_eq_abs _).symm
    _ = (‖M i j‖₊ : ℝ) := (coe_nnnorm _).symm
    _ ≤ (‖M‖₊ : ℝ) := NNReal.coe_le_coe.mpr hnn
    _ = ‖M‖ := coe_nnnorm _

/-- Perturbation of the Leontief inverse in the spatial parameter `ρ`:
`‖(1 - ρ'W)⁻¹ - (1 - ρW)⁻¹‖ ≤ |ρ' - ρ| ‖W‖ / ((1 - ‖ρ'W‖)(1 - ‖ρW‖))`.
Instance of `norm_inverse_one_sub_sub_le`. -/
theorem leontief_sub_norm_le (h : ‖ρ • W‖ < 1) (h' : ‖ρ' • W‖ < 1) :
    ‖leontief ρ' W - leontief ρ W‖ ≤
      |ρ' - ρ| * ‖W‖ / ((1 - ‖ρ' • W‖) * (1 - ‖ρ • W‖)) := by
  rcases isEmpty_or_nonempty ι with hι | hι
  · haveI := hι
    have hle : ‖leontief ρ' W - leontief ρ W‖ = 0 := by
      rw [linfty_opNorm_def]; simp
    rw [hle]
    apply div_nonneg (mul_nonneg (abs_nonneg _) (norm_nonneg _))
    exact mul_nonneg (by linarith) (by linarith)
  · haveI := hι
    have hnum : ‖ρ' • W - ρ • W‖ = |ρ' - ρ| * ‖W‖ := by
      rw [← sub_smul, norm_smul, Real.norm_eq_abs]
    have key := norm_inverse_one_sub_sub_le h' h
    rw [hnum] at key
    unfold leontief
    simp only [nonsing_inv_eq_ringInverse]
    exact key

/-- Norm bound on the spatial-covariance perturbation, via
`A' V A'ᵀ - A V Aᵀ = (A' - A) V A'ᵀ + A V (A' - A)ᵀ`. Purely algebraic — no invertibility
hypothesis; combine with `leontief_sub_norm_le` to make the right-hand side explicit in
`|ρ' - ρ|`. -/
theorem spatialCov_sub_norm_le :
    ‖spatialCov ρ' W V - spatialCov ρ W V‖ ≤
      ‖leontief ρ' W - leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W)ᵀ‖ +
        ‖leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W)ᵀ - (leontief ρ W)ᵀ‖ := by
  unfold spatialCov
  set A' := leontief ρ' W
  set A := leontief ρ W
  have hid : A' * V * A'ᵀ - A * V * Aᵀ
      = (A' - A) * V * A'ᵀ + A * V * (A'ᵀ - Aᵀ) := by
    simp only [sub_mul, mul_sub]
    abel
  rw [hid]
  refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
  · refine (linfty_opNorm_mul _ _).trans ?_
    gcongr
    exact linfty_opNorm_mul _ _
  · refine (linfty_opNorm_mul _ _).trans ?_
    gcongr
    exact linfty_opNorm_mul _ _

/-- Certified entrywise covariance bracket under spatial-parameter ambiguity: each entry of
`spatialCov ρ' W V` deviates from `spatialCov ρ W V` by at most the norm bound of
`spatialCov_sub_norm_le`. This is the interface into the energy-robustness box ∩ PSD robust
minimum-variance machinery. -/
theorem spatialCov_entry_bracket (i j : ι) :
    |spatialCov ρ' W V i j - spatialCov ρ W V i j| ≤
      ‖leontief ρ' W - leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W)ᵀ‖ +
        ‖leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W)ᵀ - (leontief ρ W)ᵀ‖ := by
  calc |spatialCov ρ' W V i j - spatialCov ρ W V i j|
      = |(spatialCov ρ' W V - spatialCov ρ W V) i j| := by rw [Matrix.sub_apply]
    _ ≤ ‖spatialCov ρ' W V - spatialCov ρ W V‖ := abs_entry_le_norm _ i j
    _ ≤ _ := spatialCov_sub_norm_le

end SpatialModel

end PricingPerspective.Discrete.Spatial
