import PricingPerspective.Connections.SpatialStability
import Mathlib.Analysis.Convex.DoublyStochasticMatrix
import Mathlib.Analysis.Normed.Lp.PiLp

/-!
# Cross-sectional dispersion under a normalized spatial multiplier

Paper 5's spatial-dispersion bridge.  For a finite field of risk-space vectors, this module
defines uniform cross-sectional squared dispersion and proves two layers:

* an abstract multiplier whose centering action commutes and whose mean-zero `L²` norm has
  factors `lower` and `upper` brackets squared cross-sectional dispersion by those factors;
* for a doubly stochastic `W` and `0 ≤ ρ < 1`, the normalized SAR multiplier
  `Sρ = (1 - ρ)(I - ρW)⁻¹` satisfies the unconditional factors
  `(1 - ρ) / (1 + ρ)` and `1`;
* for a row-stochastic `W`, the same bracket holds for dispersion weighted by any strictly
  positive stationary probability vector `π`.

The SAR bound does not assume symmetry or state eigenvalue formulas.  Double stochasticity
proves `W` is an `L²` contraction directly, while the Leontief identities prove mean
preservation and the Hilbert-valued equilibrium equation.  Consequently positive normalized
spatial feedback cannot increase uniform cross-sectional dispersion.  Although the intended
application is Hilbert-valued exposure, the algebra holds in every real normed space.
-/

namespace PricingPerspective.Connections.SpatialDispersion

open Finset Matrix PricingPerspective.Discrete.Spatial
open scoped Matrix.Norms.Operator

noncomputable section

variable {ι H : Type*} [Fintype ι]
  [NormedAddCommGroup H] [NormedSpace ℝ H]

/-! ## Finite cross-sections and their dispersion -/

/-- A finite cross-section of vectors in a common real normed risk space. -/
abbrev CrossSection (ι H : Type*) := ι → H

/-- Hilbert-valued matrix action, `(A • z)ᵢ = ∑ⱼ Aᵢⱼ zⱼ`, on a finite cross-section. -/
noncomputable def hilbertMulVec (A : Matrix ι ι ℝ) (z : CrossSection ι H) :
    CrossSection ι H :=
  fun i ↦ ∑ j, A i j • z j

/-- Evaluation formula for the Hilbert-valued matrix action. -/
@[simp] theorem hilbertMulVec_apply (A : Matrix ι ι ℝ) (z : CrossSection ι H) (i : ι) :
    hilbertMulVec A z i = ∑ j, A i j • z j := rfl

/-- The Hilbert-valued action of a fixed real matrix is real-linear. -/
noncomputable def hilbertMulVecLinear (A : Matrix ι ι ℝ) :
    CrossSection ι H →ₗ[ℝ] CrossSection ι H where
  toFun := hilbertMulVec A
  map_add' x y := by
    ext i
    simp only [hilbertMulVec_apply, Pi.add_apply, smul_add, sum_add_distrib]
  map_smul' c x := by
    ext i
    simp only [hilbertMulVec_apply, Pi.smul_apply, RingHom.id_apply, smul_smul]
    rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [← mul_smul, mul_comm, mul_smul]

/-- Applying the bundled linear action agrees with `hilbertMulVec`. -/
@[simp] theorem hilbertMulVecLinear_apply (A : Matrix ι ι ℝ) (z : CrossSection ι H) :
    hilbertMulVecLinear A z = hilbertMulVec A z := rfl

private theorem hilbertMulVec_one [DecidableEq ι] (z : CrossSection ι H) :
    hilbertMulVec (1 : Matrix ι ι ℝ) z = z := by
  ext i
  simp [hilbertMulVec, Matrix.one_apply]

private theorem hilbertMulVec_mul [DecidableEq ι]
    (A B : Matrix ι ι ℝ) (z : CrossSection ι H) :
    hilbertMulVec (A * B) z = hilbertMulVec A (hilbertMulVec B z) := by
  ext i
  simp only [hilbertMulVec_apply, Matrix.mul_apply]
  simp_rw [Finset.sum_smul, Finset.smul_sum, mul_smul]
  rw [Finset.sum_comm]

private theorem hilbertMulVec_sub (A B : Matrix ι ι ℝ) (z : CrossSection ι H) :
    hilbertMulVec (A - B) z = hilbertMulVec A z - hilbertMulVec B z := by
  ext i
  simp [hilbertMulVec, Matrix.sub_apply, sub_smul, Finset.sum_sub_distrib]

private theorem hilbertMulVec_matrix_smul
    (a : ℝ) (A : Matrix ι ι ℝ) (z : CrossSection ι H) :
    hilbertMulVec (a • A) z = a • hilbertMulVec A z := by
  ext i
  simp [hilbertMulVec, Matrix.smul_apply, smul_eq_mul, mul_smul, Finset.smul_sum]

/-- Unnormalized cross-sectional `L²` energy, `∑ᵢ ‖zᵢ‖²`. -/
noncomputable def crossSectionalEnergy (z : CrossSection ι H) : ℝ :=
  ∑ i, ‖z i‖ ^ 2

/-- The finite cross-sectional `L²` norm, `(∑ᵢ ‖zᵢ‖²)¹ᐟ²`. -/
noncomputable def crossSectionalL2Norm (z : CrossSection ι H) : ℝ :=
  Real.sqrt (crossSectionalEnergy z)

omit [NormedSpace ℝ H] in
/-- Cross-sectional energy is nonnegative. -/
theorem crossSectionalEnergy_nonneg (z : CrossSection ι H) :
    0 ≤ crossSectionalEnergy z := by
  unfold crossSectionalEnergy
  exact Finset.sum_nonneg fun _ _ ↦ sq_nonneg _

omit [NormedSpace ℝ H] in
/-- Squaring the cross-sectional `L²` norm recovers cross-sectional energy. -/
theorem crossSectionalL2Norm_sq (z : CrossSection ι H) :
    crossSectionalL2Norm z ^ 2 = crossSectionalEnergy z := by
  exact Real.sq_sqrt (crossSectionalEnergy_nonneg z)

omit [NormedSpace ℝ H] in
private theorem crossSectionalL2Norm_eq_piLpNorm (z : CrossSection ι H) :
    crossSectionalL2Norm z =
      ‖(WithLp.toLp 2 z : PiLp 2 (fun _ : ι ↦ H))‖ := by
  rw [PiLp.norm_eq_of_L2]
  rfl

omit [NormedSpace ℝ H] in
private theorem crossSectionalL2Norm_add_le (x y : CrossSection ι H) :
    crossSectionalL2Norm (x + y) ≤ crossSectionalL2Norm x + crossSectionalL2Norm y := by
  simp_rw [crossSectionalL2Norm_eq_piLpNorm]
  simpa using norm_add_le
    (WithLp.toLp 2 x : PiLp 2 (fun _ : ι ↦ H)) (WithLp.toLp 2 y)

omit [NormedSpace ℝ H] in
private theorem crossSectionalL2Norm_sub_le (x y : CrossSection ι H) :
    crossSectionalL2Norm (x - y) ≤ crossSectionalL2Norm x + crossSectionalL2Norm y := by
  simp_rw [crossSectionalL2Norm_eq_piLpNorm]
  simpa using norm_sub_le
    (WithLp.toLp 2 x : PiLp 2 (fun _ : ι ↦ H)) (WithLp.toLp 2 y)

private theorem crossSectionalL2Norm_smul (a : ℝ) (z : CrossSection ι H) :
    crossSectionalL2Norm (a • z) = |a| * crossSectionalL2Norm z := by
  rw [crossSectionalL2Norm_eq_piLpNorm, crossSectionalL2Norm_eq_piLpNorm]
  have htoLp : (WithLp.toLp 2 (a • z) : PiLp 2 (fun _ : ι ↦ H)) =
      a • WithLp.toLp 2 z := rfl
  rw [htoLp, norm_smul, Real.norm_eq_abs]

/-- The uniform cross-sectional mean, `z̄ = N⁻¹ ∑ᵢ zᵢ`. -/
noncomputable def crossSectionalMean (z : CrossSection ι H) : H :=
  (Fintype.card ι : ℝ)⁻¹ • ∑ i, z i

/-- The centered field, `zᵢ - z̄`. -/
noncomputable def centerCrossSection (z : CrossSection ι H) : CrossSection ι H :=
  fun i ↦ z i - crossSectionalMean z

/-- A finite field is mean-zero when its vector sum vanishes. -/
def IsCrossSectionalMeanZero (z : CrossSection ι H) : Prop :=
  ∑ i, z i = 0

/-- A mean-zero field has zero uniform cross-sectional mean. -/
theorem crossSectionalMean_eq_zero_of_meanZero
    {z : CrossSection ι H} (hz : IsCrossSectionalMeanZero z) :
    crossSectionalMean z = 0 := by
  unfold IsCrossSectionalMeanZero at hz
  unfold crossSectionalMean
  rw [hz, smul_zero]

/-- Centering leaves an already mean-zero field unchanged. -/
theorem centerCrossSection_eq_self_of_meanZero
    {z : CrossSection ι H} (hz : IsCrossSectionalMeanZero z) :
    centerCrossSection z = z := by
  ext i
  simp [centerCrossSection, crossSectionalMean_eq_zero_of_meanZero hz]

/-- On a nonempty finite index set, centering produces a mean-zero field. -/
theorem isCrossSectionalMeanZero_center [Nonempty ι] (z : CrossSection ι H) :
    IsCrossSectionalMeanZero (centerCrossSection z) := by
  unfold IsCrossSectionalMeanZero centerCrossSection crossSectionalMean
  rw [Finset.sum_sub_distrib]
  have hcard : (Fintype.card ι : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  rw [Finset.sum_const, Finset.card_univ, ← Nat.cast_smul_eq_nsmul ℝ, smul_smul,
    mul_inv_cancel₀ hcard, one_smul, sub_self]

/-- Uniform cross-sectional squared dispersion,
`D(z) = N⁻¹ ∑ᵢ ‖zᵢ - z̄‖²`, for a nonempty finite cross-section. -/
noncomputable def crossSectionalDispersionSq (z : CrossSection ι H) : ℝ :=
  (Fintype.card ι : ℝ)⁻¹ * crossSectionalEnergy (centerCrossSection z)

/-! ## Abstract multiplier theorem -/

/-- A linear cross-sectional multiplier commutes with uniform centering. -/
def CommutesWithCrossSectionalCentering
    (T : CrossSection ι H →ₗ[ℝ] CrossSection ι H) : Prop :=
  ∀ z, centerCrossSection (T z) = T (centerCrossSection z)

/-- Mean-zero `L²` norm factors for a linear cross-sectional multiplier.

The premise is an explicit operator property: on every mean-zero field `z`,
`lower ‖z‖₂ ≤ ‖Tz‖₂ ≤ upper ‖z‖₂`. -/
def HasMeanZeroL2NormBounds (lower upper : ℝ)
    (T : CrossSection ι H →ₗ[ℝ] CrossSection ι H) : Prop :=
  ∀ z, IsCrossSectionalMeanZero z →
    lower * crossSectionalL2Norm z ≤ crossSectionalL2Norm (T z) ∧
      crossSectionalL2Norm (T z) ≤ upper * crossSectionalL2Norm z

/-- Abstract squared-dispersion bracket for a centering-compatible multiplier.

If `T` has nonnegative lower and upper `L²` norm factors on the mean-zero subspace, then
`lower² D(z) ≤ D(Tz) ≤ upper² D(z)` for every finite nonempty cross-section. -/
theorem crossSectionalDispersionSq_bracket [Nonempty ι]
    {lower upper : ℝ} {T : CrossSection ι H →ₗ[ℝ] CrossSection ι H}
    (hlower : 0 ≤ lower) (hupper : 0 ≤ upper)
    (hcenter : CommutesWithCrossSectionalCentering T)
    (hbounds : HasMeanZeroL2NormBounds lower upper T) (z : CrossSection ι H) :
    lower ^ 2 * crossSectionalDispersionSq z ≤ crossSectionalDispersionSq (T z) ∧
      crossSectionalDispersionSq (T z) ≤ upper ^ 2 * crossSectionalDispersionSq z := by
  have hb := hbounds (centerCrossSection z) (isCrossSectionalMeanZero_center z)
  have hzn : 0 ≤ crossSectionalL2Norm (centerCrossSection z) := Real.sqrt_nonneg _
  have hTn : 0 ≤ crossSectionalL2Norm (T (centerCrossSection z)) := Real.sqrt_nonneg _
  have hlmul : 0 ≤ lower * crossSectionalL2Norm (centerCrossSection z) :=
    mul_nonneg hlower hzn
  have humul : 0 ≤ upper * crossSectionalL2Norm (centerCrossSection z) :=
    mul_nonneg hupper hzn
  have hloSq : (lower * crossSectionalL2Norm (centerCrossSection z)) ^ 2 ≤
      crossSectionalL2Norm (T (centerCrossSection z)) ^ 2 :=
    (sq_le_sq₀ hlmul hTn).2 hb.1
  have hupSq : crossSectionalL2Norm (T (centerCrossSection z)) ^ 2 ≤
      (upper * crossSectionalL2Norm (centerCrossSection z)) ^ 2 :=
    (sq_le_sq₀ hTn humul).2 hb.2
  have hcard : 0 ≤ (Fintype.card ι : ℝ)⁻¹ := by positivity
  constructor
  · unfold crossSectionalDispersionSq
    rw [hcenter z]
    rw [← crossSectionalL2Norm_sq, ← crossSectionalL2Norm_sq]
    nlinarith [mul_nonneg hcard (sub_nonneg.mpr hloSq)]
  · unfold crossSectionalDispersionSq
    rw [hcenter z]
    rw [← crossSectionalL2Norm_sq, ← crossSectionalL2Norm_sq]
    nlinarith [mul_nonneg hcard (sub_nonneg.mpr hupSq)]

/-! ## Doubly stochastic actions -/

private theorem norm_weightedSum_sq_le (w : ι → ℝ) (z : CrossSection ι H)
    (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1) :
    ‖∑ i, w i • z i‖ ^ 2 ≤ ∑ i, w i * ‖z i‖ ^ 2 := by
  have hnorm : ‖∑ i, w i • z i‖ ≤ ∑ i, w i * ‖z i‖ := by
    calc
      ‖∑ i, w i • z i‖ ≤ ∑ i, ‖w i • z i‖ := norm_sum_le _ _
      _ = ∑ i, w i * ‖z i‖ := by
        apply Finset.sum_congr rfl
        intro i _
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw i)]
  have hsumNonneg : 0 ≤ ∑ i, w i * ‖z i‖ := by
    exact Finset.sum_nonneg fun i _ ↦ mul_nonneg (hw i) (norm_nonneg _)
  have hnormSq : ‖∑ i, w i • z i‖ ^ 2 ≤ (∑ i, w i * ‖z i‖) ^ 2 :=
    (sq_le_sq₀ (norm_nonneg _) hsumNonneg).2 hnorm
  have hcauchy : (∑ i, w i * ‖z i‖) ^ 2 ≤
      (∑ i, w i) * ∑ i, w i * ‖z i‖ ^ 2 := by
    apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul Finset.univ
    · intro i _
      exact hw i
    · intro i _
      exact mul_nonneg (hw i) (sq_nonneg _)
    · intro i _
      ring_nf
      exact le_rfl
  rw [hsum, one_mul] at hcauchy
  exact hnormSq.trans hcauchy

/-- A doubly stochastic real matrix cannot increase Hilbert-valued cross-sectional energy. -/
theorem hilbertMulVec_energy_le_of_doublyStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι) (z : CrossSection ι H) :
    crossSectionalEnergy (hilbertMulVec W z) ≤ crossSectionalEnergy z := by
  unfold crossSectionalEnergy
  calc
    ∑ i, ‖hilbertMulVec W z i‖ ^ 2 ≤ ∑ i, ∑ j, W i j * ‖z j‖ ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      exact norm_weightedSum_sq_le (W i) z
        (fun _ ↦ nonneg_of_mem_doublyStochastic hW)
        (sum_row_of_mem_doublyStochastic hW i)
    _ = ∑ j, ‖z j‖ ^ 2 := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j _
      rw [← Finset.sum_mul, sum_col_of_mem_doublyStochastic hW j, one_mul]

/-- A doubly stochastic real matrix is a contraction for the finite Hilbert-valued `L²` norm. -/
theorem hilbertMulVec_l2Norm_le_of_doublyStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι) (z : CrossSection ι H) :
    crossSectionalL2Norm (hilbertMulVec W z) ≤ crossSectionalL2Norm z := by
  unfold crossSectionalL2Norm
  exact Real.sqrt_le_sqrt (hilbertMulVec_energy_le_of_doublyStochastic hW z)

private theorem sum_hilbertMulVec_of_doublyStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι) (z : CrossSection ι H) :
    ∑ i, hilbertMulVec W z i = ∑ i, z i := by
  unfold hilbertMulVec
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [← Finset.sum_smul, sum_col_of_mem_doublyStochastic hW j, one_smul]

private theorem hilbertMulVec_const_of_doublyStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι) (c : H) :
    hilbertMulVec W (fun _ ↦ c) = fun _ ↦ c := by
  ext i
  unfold hilbertMulVec
  rw [← Finset.sum_smul, sum_row_of_mem_doublyStochastic hW i, one_smul]

/-! ## Normalized SAR specialization -/

/-- Double stochasticity and `0 ≤ ρ < 1` imply the existing Leontief norm gate. -/
theorem doublyStochastic_sar_gate [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) : ‖ρ • W‖ < 1 := by
  have hWnorm : ‖W‖ ≤ 1 := by
    apply PricingPerspective.Connections.SpatialStability.norm_le_of_abs_row_sum_le
      (by norm_num)
    intro i
    calc
      ∑ j, |W i j| = ∑ j, W i j := by
        apply Finset.sum_congr rfl
        intro j _
        rw [abs_of_nonneg (nonneg_of_mem_doublyStochastic hW)]
      _ = 1 := sum_row_of_mem_doublyStochastic hW i
      _ ≤ 1 := le_rfl
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hρ0]
  calc
    ρ * ‖W‖ ≤ ρ * 1 := mul_le_mul_of_nonneg_left hWnorm hρ0
    _ < 1 := by simpa using hρ1

/-- The normalized SAR matrix, `Sρ = (1 - ρ)(I - ρW)⁻¹`. -/
noncomputable def normalizedSARMatrix [DecidableEq ι]
    (ρ : ℝ) (W : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  (1 - ρ) • leontief ρ W

/-- The normalized SAR matrix acting linearly on a Hilbert-valued cross-section. -/
noncomputable def normalizedSARLinear [DecidableEq ι]
    (ρ : ℝ) (W : Matrix ι ι ℝ) :
    CrossSection ι H →ₗ[ℝ] CrossSection ι H :=
  hilbertMulVecLinear (normalizedSARMatrix ρ W)

/-- Evaluation of the normalized Hilbert-valued SAR multiplier. -/
@[simp] theorem normalizedSARLinear_apply [DecidableEq ι]
    (ρ : ℝ) (W : Matrix ι ι ℝ) (z : CrossSection ι H) :
    normalizedSARLinear ρ W z =
      (1 - ρ) • hilbertMulVec (leontief ρ W) z := by
  rw [normalizedSARLinear, hilbertMulVecLinear_apply, normalizedSARMatrix,
    hilbertMulVec_matrix_smul]

/-- The Leontief action solves the Hilbert-valued unnormalized spatial reduced form. -/
theorem leontief_hilbert_reduced_form [DecidableEq ι]
    {W : Matrix ι ι ℝ} {ρ : ℝ} (hgate : ‖ρ • W‖ < 1) (z : CrossSection ι H) :
    hilbertMulVec (leontief ρ W) z =
      z + ρ • hilbertMulVec W (hilbertMulVec (leontief ρ W) z) := by
  let u := hilbertMulVec (leontief ρ W) z
  have hu : hilbertMulVec (1 - ρ • W) u = z := by
    calc
      hilbertMulVec (1 - ρ • W) u =
          hilbertMulVec ((1 - ρ • W) * leontief ρ W) z := by
        rw [hilbertMulVec_mul]
      _ = hilbertMulVec 1 z := by rw [one_sub_smul_mul_leontief hgate]
      _ = z := hilbertMulVec_one z
  have hu' : u - ρ • hilbertMulVec W u = z := by
    rw [hilbertMulVec_sub, hilbertMulVec_one, hilbertMulVec_matrix_smul] at hu
    exact hu
  change u = z + ρ • hilbertMulVec W u
  exact eq_add_of_sub_eq hu'

/-- Exact normalized Hilbert-valued SAR equation,
`Sρ z = (1 - ρ)z + ρW(Sρ z)`, under the Leontief norm gate. -/
theorem normalizedSAR_equilibrium [DecidableEq ι]
    {W : Matrix ι ι ℝ} {ρ : ℝ} (hgate : ‖ρ • W‖ < 1) (z : CrossSection ι H) :
    normalizedSARLinear ρ W z =
      (1 - ρ) • z + ρ • hilbertMulVec W (normalizedSARLinear ρ W z) := by
  let u := hilbertMulVec (leontief ρ W) z
  have hu : u = z + ρ • hilbertMulVec W u :=
    leontief_hilbert_reduced_form hgate z
  rw [normalizedSARLinear_apply]
  change (1 - ρ) • u = (1 - ρ) • z + ρ • hilbertMulVec W ((1 - ρ) • u)
  calc
    (1 - ρ) • u = (1 - ρ) • (z + ρ • hilbertMulVec W u) := congrArg _ hu
    _ = (1 - ρ) • z + (1 - ρ) • (ρ • hilbertMulVec W u) := smul_add _ _ _
    _ = (1 - ρ) • z + ρ • hilbertMulVec W ((1 - ρ) • u) := by
      congr 1
      change (1 - ρ) • (ρ • (hilbertMulVecLinear W) u) =
        ρ • (hilbertMulVecLinear W) ((1 - ρ) • u)
      rw [LinearMap.map_smul]
      simp only [smul_smul]
      congr 1
      ring

private theorem sum_normalizedSAR_eq [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    ∑ i, normalizedSARLinear ρ W z i = ∑ i, z i := by
  let y := normalizedSARLinear ρ W z
  have hy : y = (1 - ρ) • z + ρ • hilbertMulVec W y :=
    normalizedSAR_equilibrium (doublyStochastic_sar_gate hW hρ0 hρ1) z
  have hsum := congrArg (fun x : CrossSection ι H ↦ ∑ i, x i) hy
  simp only [Pi.add_apply, Pi.smul_apply, Finset.sum_add_distrib, ← Finset.smul_sum] at hsum
  rw [sum_hilbertMulVec_of_doublyStochastic hW] at hsum
  have hsub : (∑ i, y i) - ρ • ∑ i, y i = (1 - ρ) • ∑ i, z i := by
    calc
      (∑ i, y i) - ρ • ∑ i, y i =
          ((1 - ρ) • ∑ i, z i + ρ • ∑ i, y i) - ρ • ∑ i, y i :=
        congrArg (fun q ↦ q - ρ • ∑ i, y i) hsum
      _ = (1 - ρ) • ∑ i, z i := by abel
  have hscaled : (1 - ρ) • (∑ i, y i) = (1 - ρ) • ∑ i, z i := by
    calc
      (1 - ρ) • (∑ i, y i) = (∑ i, y i) - ρ • ∑ i, y i := by
        rw [sub_smul, one_smul]
      _ = (1 - ρ) • ∑ i, z i := hsub
  have ha : (1 - ρ : ℝ) ≠ 0 := ne_of_gt (sub_pos.mpr hρ1)
  exact smul_right_injective H ha hscaled

private theorem normalizedSAR_const [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (c : H) :
    normalizedSARLinear ρ W (fun _ ↦ c) = fun _ ↦ c := by
  let a : ℝ := 1 - ρ
  have ha : a ≠ 0 := ne_of_gt (sub_pos.mpr hρ1)
  let v : CrossSection ι H := fun _ ↦ a⁻¹ • c
  have hbase : hilbertMulVec (1 - ρ • W) v = fun _ ↦ c := by
    rw [hilbertMulVec_sub, hilbertMulVec_one, hilbertMulVec_matrix_smul,
      hilbertMulVec_const_of_doublyStochastic hW]
    ext i
    change a⁻¹ • c - ρ • (a⁻¹ • c) = c
    rw [smul_smul, ← sub_smul]
    have hscalar : a⁻¹ - ρ * a⁻¹ = 1 := by
      calc
        a⁻¹ - ρ * a⁻¹ = (1 - ρ) * a⁻¹ := by ring
        _ = a * a⁻¹ := by rfl
        _ = 1 := mul_inv_cancel₀ ha
    rw [hscalar, one_smul]
  have hgate := doublyStochastic_sar_gate hW hρ0 hρ1
  calc
    normalizedSARLinear ρ W (fun _ ↦ c) =
        a • hilbertMulVec (leontief ρ W) (fun _ ↦ c) := by
      rw [normalizedSARLinear_apply]
    _ = a • hilbertMulVec (leontief ρ W) (hilbertMulVec (1 - ρ • W) v) := by
      rw [hbase]
    _ = a • hilbertMulVec (leontief ρ W * (1 - ρ • W)) v := by
      rw [hilbertMulVec_mul]
    _ = a • hilbertMulVec 1 v := by rw [leontief_mul_one_sub_smul hgate]
    _ = a • v := by rw [hilbertMulVec_one]
    _ = fun _ ↦ c := by
      ext i
      change a • (a⁻¹ • c) = c
      rw [smul_smul, mul_inv_cancel₀ ha, one_smul]

/-- The normalized doubly stochastic SAR commutes with uniform centering. -/
theorem normalizedSAR_commutesWithCentering [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    CommutesWithCrossSectionalCentering (normalizedSARLinear ρ W :
      CrossSection ι H →ₗ[ℝ] CrossSection ι H) := by
  intro z
  have hmean : crossSectionalMean (normalizedSARLinear ρ W z) =
      crossSectionalMean z := by
    unfold crossSectionalMean
    rw [sum_normalizedSAR_eq hW hρ0 hρ1]
  calc
    centerCrossSection (normalizedSARLinear ρ W z) =
        normalizedSARLinear ρ W z -
          fun _ ↦ crossSectionalMean (normalizedSARLinear ρ W z) := rfl
    _ = normalizedSARLinear ρ W z - fun _ ↦ crossSectionalMean z := by rw [hmean]
    _ = normalizedSARLinear ρ W z -
        normalizedSARLinear ρ W (fun _ ↦ crossSectionalMean z) := by
      rw [normalizedSAR_const hW hρ0 hρ1]
    _ = normalizedSARLinear ρ W (z - fun _ ↦ crossSectionalMean z) := by
      rw [LinearMap.map_sub]
    _ = normalizedSARLinear ρ W (centerCrossSection z) := rfl

/-- Universal mean-zero norm factors for the normalized doubly stochastic SAR.

For `0 ≤ ρ < 1`, no symmetry or spectral premise is needed:
`((1 - ρ)/(1 + ρ)) ‖z‖₂ ≤ ‖Sρz‖₂ ≤ ‖z‖₂`. -/
theorem normalizedSAR_meanZeroL2NormBounds [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    HasMeanZeroL2NormBounds ((1 - ρ) / (1 + ρ)) 1
      (normalizedSARLinear ρ W : CrossSection ι H →ₗ[ℝ] CrossSection ι H) := by
  intro z _
  let y := normalizedSARLinear ρ W z
  have hy : y = (1 - ρ) • z + ρ • hilbertMulVec W y :=
    normalizedSAR_equilibrium (doublyStochastic_sar_gate hW hρ0 hρ1) z
  have hWcontract : crossSectionalL2Norm (hilbertMulVec W y) ≤
      crossSectionalL2Norm y :=
    hilbertMulVec_l2Norm_le_of_doublyStochastic hW y
  have ha0 : 0 ≤ 1 - ρ := sub_nonneg.mpr hρ1.le
  have hupper0 : crossSectionalL2Norm y ≤
      (1 - ρ) * crossSectionalL2Norm z + ρ * crossSectionalL2Norm y := by
    calc
      crossSectionalL2Norm y =
          crossSectionalL2Norm ((1 - ρ) • z + ρ • hilbertMulVec W y) :=
        congrArg crossSectionalL2Norm hy
      _ ≤ crossSectionalL2Norm ((1 - ρ) • z) +
          crossSectionalL2Norm (ρ • hilbertMulVec W y) :=
        crossSectionalL2Norm_add_le _ _
      _ = (1 - ρ) * crossSectionalL2Norm z +
          ρ * crossSectionalL2Norm (hilbertMulVec W y) := by
        rw [crossSectionalL2Norm_smul, crossSectionalL2Norm_smul,
          abs_of_nonneg ha0, abs_of_nonneg hρ0]
      _ ≤ (1 - ρ) * crossSectionalL2Norm z + ρ * crossSectionalL2Norm y := by
        exact add_le_add_right (mul_le_mul_of_nonneg_left hWcontract hρ0) _
  have hupper : crossSectionalL2Norm y ≤ crossSectionalL2Norm z := by
    by_contra hnot
    have hzy : crossSectionalL2Norm z < crossSectionalL2Norm y := lt_of_not_ge hnot
    nlinarith [mul_pos (sub_pos.mpr hρ1) (sub_pos.mpr hzy)]
  have hsolve : (1 - ρ) • z = y - ρ • hilbertMulVec W y := by
    calc
      (1 - ρ) • z = ((1 - ρ) • z + ρ • hilbertMulVec W y) -
          ρ • hilbertMulVec W y := by abel
      _ = y - ρ • hilbertMulVec W y := by rw [← hy]
  have hlower0 : (1 - ρ) * crossSectionalL2Norm z ≤
      (1 + ρ) * crossSectionalL2Norm y := by
    calc
      (1 - ρ) * crossSectionalL2Norm z = crossSectionalL2Norm ((1 - ρ) • z) := by
        rw [crossSectionalL2Norm_smul, abs_of_nonneg ha0]
      _ = crossSectionalL2Norm (y - ρ • hilbertMulVec W y) :=
        congrArg crossSectionalL2Norm hsolve
      _ ≤ crossSectionalL2Norm y + crossSectionalL2Norm (ρ • hilbertMulVec W y) :=
        crossSectionalL2Norm_sub_le _ _
      _ = crossSectionalL2Norm y +
          ρ * crossSectionalL2Norm (hilbertMulVec W y) := by
        rw [crossSectionalL2Norm_smul, abs_of_nonneg hρ0]
      _ ≤ crossSectionalL2Norm y + ρ * crossSectionalL2Norm y := by
        exact add_le_add_right (mul_le_mul_of_nonneg_left hWcontract hρ0) _
      _ = (1 + ρ) * crossSectionalL2Norm y := by ring
  constructor
  · rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith : 0 < 1 + ρ)]
    nlinarith
  · change crossSectionalL2Norm y ≤ 1 * crossSectionalL2Norm z
    simpa using hupper

/-- Paper 5 spatial-dispersion attenuation for the normalized doubly stochastic SAR.

For every nonempty finite cross-section and `0 ≤ ρ < 1`,
`((1 - ρ)/(1 + ρ))² D(z) ≤ D(Sρz) ≤ D(z)`.  These universal factors are coarser than
possible symmetric spectral factors but require no eigenvalue premise. -/
theorem normalizedSAR_crossSectionalDispersionSq_bracket
    [DecidableEq ι] [Nonempty ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ doublyStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    ((1 - ρ) / (1 + ρ)) ^ 2 * crossSectionalDispersionSq z ≤
        crossSectionalDispersionSq (normalizedSARLinear ρ W z) ∧
      crossSectionalDispersionSq (normalizedSARLinear ρ W z) ≤
        crossSectionalDispersionSq z := by
  have hlower : 0 ≤ (1 - ρ) / (1 + ρ) := by positivity
  simpa using crossSectionalDispersionSq_bracket hlower (by norm_num : (0 : ℝ) ≤ 1)
    (normalizedSAR_commutesWithCentering hW hρ0 hρ1)
    (normalizedSAR_meanZeroL2NormBounds hW hρ0 hρ1) z

/-! ## Stationary-weighted dispersion for row-stochastic actions -/

/-- Paper 5's strictly positive stationary probability-weight condition.

For a finite transition matrix `W`, this says that every `πᵢ` is positive, the weights sum
to one, and `πᵀW = πᵀ`. -/
def IsStrictlyPositiveStationaryWeight
    (W : Matrix ι ι ℝ) (π : ι → ℝ) : Prop :=
  (∀ i, 0 < π i) ∧ (∑ i, π i = 1) ∧ ∀ j, ∑ i, π i * W i j = π j

/-- Paper 5's stationary-weighted cross-sectional mean, `μπ(z) = ∑ᵢ πᵢ zᵢ`.

The intended weights are a strictly positive stationary probability vector, as recorded by
`IsStrictlyPositiveStationaryWeight`. -/
noncomputable def weightedCrossSectionalMean
    (π : ι → ℝ) (z : CrossSection ι H) : H :=
  ∑ i, π i • z i

/-- Paper 5's stationary-weighted centering operation, `zᵢ - μπ(z)`.

When the weights sum to one, its weighted mean is zero. -/
noncomputable def centerCrossSectionWeighted
    (π : ι → ℝ) (z : CrossSection ι H) : CrossSection ι H :=
  fun i ↦ z i - weightedCrossSectionalMean π z

/-- Paper 5's stationary-weighted squared cross-sectional dispersion.

For probability weights `π`, this is `Dπ(z) = ∑ᵢ πᵢ ‖zᵢ - μπ(z)‖²`. -/
noncomputable def weightedCrossSectionalDispersionSq
    (π : ι → ℝ) (z : CrossSection ι H) : ℝ :=
  ∑ i, π i * ‖centerCrossSectionWeighted π z i‖ ^ 2

private noncomputable def weightedCrossSectionalEnergy
    (π : ι → ℝ) (z : CrossSection ι H) : ℝ :=
  ∑ i, π i * ‖z i‖ ^ 2

private noncomputable def weightedCrossSectionalL2Norm
    (π : ι → ℝ) (z : CrossSection ι H) : ℝ :=
  Real.sqrt (weightedCrossSectionalEnergy π z)

private noncomputable def sqrtWeightCrossSection
    (π : ι → ℝ) (z : CrossSection ι H) : CrossSection ι H :=
  fun i ↦ Real.sqrt (π i) • z i

omit [NormedSpace ℝ H] in
private theorem weightedCrossSectionalEnergy_nonneg
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (z : CrossSection ι H) :
    0 ≤ weightedCrossSectionalEnergy π z := by
  exact Finset.sum_nonneg fun i _ ↦ mul_nonneg (hπ0 i) (sq_nonneg _)

omit [NormedSpace ℝ H] in
private theorem weightedCrossSectionalL2Norm_sq
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (z : CrossSection ι H) :
    weightedCrossSectionalL2Norm π z ^ 2 = weightedCrossSectionalEnergy π z := by
  exact Real.sq_sqrt (weightedCrossSectionalEnergy_nonneg hπ0 z)

private theorem weightedCrossSectionalEnergy_eq_sqrtWeight
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (z : CrossSection ι H) :
    weightedCrossSectionalEnergy π z =
      crossSectionalEnergy (sqrtWeightCrossSection π z) := by
  unfold weightedCrossSectionalEnergy crossSectionalEnergy sqrtWeightCrossSection
  apply Finset.sum_congr rfl
  intro i _
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg _), mul_pow,
    Real.sq_sqrt (hπ0 i)]

private theorem weightedCrossSectionalL2Norm_eq_sqrtWeight
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (z : CrossSection ι H) :
    weightedCrossSectionalL2Norm π z =
      crossSectionalL2Norm (sqrtWeightCrossSection π z) := by
  unfold weightedCrossSectionalL2Norm crossSectionalL2Norm
  rw [weightedCrossSectionalEnergy_eq_sqrtWeight hπ0]

private theorem weightedCrossSectionalL2Norm_add_le
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (x y : CrossSection ι H) :
    weightedCrossSectionalL2Norm π (x + y) ≤
      weightedCrossSectionalL2Norm π x + weightedCrossSectionalL2Norm π y := by
  simp_rw [weightedCrossSectionalL2Norm_eq_sqrtWeight hπ0]
  have hadd : sqrtWeightCrossSection π (x + y) =
      sqrtWeightCrossSection π x + sqrtWeightCrossSection π y := by
    ext i
    simp [sqrtWeightCrossSection, smul_add]
  rw [hadd]
  exact crossSectionalL2Norm_add_le _ _

private theorem weightedCrossSectionalL2Norm_sub_le
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (x y : CrossSection ι H) :
    weightedCrossSectionalL2Norm π (x - y) ≤
      weightedCrossSectionalL2Norm π x + weightedCrossSectionalL2Norm π y := by
  simp_rw [weightedCrossSectionalL2Norm_eq_sqrtWeight hπ0]
  have hsub : sqrtWeightCrossSection π (x - y) =
      sqrtWeightCrossSection π x - sqrtWeightCrossSection π y := by
    ext i
    simp [sqrtWeightCrossSection, smul_sub]
  rw [hsub]
  exact crossSectionalL2Norm_sub_le _ _

private theorem weightedCrossSectionalL2Norm_smul
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i) (a : ℝ) (z : CrossSection ι H) :
    weightedCrossSectionalL2Norm π (a • z) =
      |a| * weightedCrossSectionalL2Norm π z := by
  simp_rw [weightedCrossSectionalL2Norm_eq_sqrtWeight hπ0]
  have hsmul : sqrtWeightCrossSection π (a • z) =
      a • sqrtWeightCrossSection π z := by
    ext i
    simp only [sqrtWeightCrossSection, Pi.smul_apply, smul_smul]
    rw [mul_comm]
  rw [hsmul, crossSectionalL2Norm_smul]

private theorem hilbertMulVec_weightedEnergy_le_of_rowStochastic
    [DecidableEq ι] {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i)
    (hstationary : ∀ j, ∑ i, π i * W i j = π j) (z : CrossSection ι H) :
    weightedCrossSectionalEnergy π (hilbertMulVec W z) ≤
      weightedCrossSectionalEnergy π z := by
  unfold weightedCrossSectionalEnergy
  calc
    ∑ i, π i * ‖hilbertMulVec W z i‖ ^ 2 ≤
        ∑ i, π i * ∑ j, W i j * ‖z j‖ ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left
        (norm_weightedSum_sq_le (W i) z
          (fun _ ↦ Matrix.nonneg_of_mem_rowStochastic hW)
          (Matrix.sum_row_of_mem_rowStochastic hW i))
        (hπ0 i)
    _ = ∑ j, π j * ‖z j‖ ^ 2 := by
      simp_rw [Finset.mul_sum, ← mul_assoc]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j _
      rw [← Finset.sum_mul, hstationary j]

private theorem hilbertMulVec_weightedL2Norm_le_of_rowStochastic
    [DecidableEq ι] {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i)
    (hstationary : ∀ j, ∑ i, π i * W i j = π j) (z : CrossSection ι H) :
    weightedCrossSectionalL2Norm π (hilbertMulVec W z) ≤
      weightedCrossSectionalL2Norm π z := by
  unfold weightedCrossSectionalL2Norm
  exact Real.sqrt_le_sqrt
    (hilbertMulVec_weightedEnergy_le_of_rowStochastic hW hπ0 hstationary z)

private theorem rowStochastic_sar_gate [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) : ‖ρ • W‖ < 1 := by
  have hWnorm : ‖W‖ ≤ 1 := by
    apply PricingPerspective.Connections.SpatialStability.norm_le_of_abs_row_sum_le
      (by norm_num)
    intro i
    calc
      ∑ j, |W i j| = ∑ j, W i j := by
        apply Finset.sum_congr rfl
        intro j _
        rw [abs_of_nonneg (Matrix.nonneg_of_mem_rowStochastic hW)]
      _ = 1 := Matrix.sum_row_of_mem_rowStochastic hW i
      _ ≤ 1 := le_rfl
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hρ0]
  calc
    ρ * ‖W‖ ≤ ρ * 1 := mul_le_mul_of_nonneg_left hWnorm hρ0
    _ < 1 := by simpa using hρ1

private theorem weightedMean_hilbertMulVec_eq_of_stationary [DecidableEq ι]
    {W : Matrix ι ι ℝ} {π : ι → ℝ}
    (hstationary : ∀ j, ∑ i, π i * W i j = π j) (z : CrossSection ι H) :
    weightedCrossSectionalMean π (hilbertMulVec W z) =
      weightedCrossSectionalMean π z := by
  unfold weightedCrossSectionalMean hilbertMulVec
  simp_rw [Finset.smul_sum, smul_smul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [← Finset.sum_smul, hstationary j]

private theorem hilbertMulVec_const_of_rowStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι) (c : H) :
    hilbertMulVec W (fun _ ↦ c) = fun _ ↦ c := by
  ext i
  unfold hilbertMulVec
  rw [← Finset.sum_smul, Matrix.sum_row_of_mem_rowStochastic hW i, one_smul]

private theorem weightedMean_normalizedSAR_eq [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hstationary : ∀ j, ∑ i, π i * W i j = π j)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    weightedCrossSectionalMean π (normalizedSARLinear ρ W z) =
      weightedCrossSectionalMean π z := by
  let y := normalizedSARLinear ρ W z
  have hy : y = (1 - ρ) • z + ρ • hilbertMulVec W y :=
    normalizedSAR_equilibrium (rowStochastic_sar_gate hW hρ0 hρ1) z
  have hmean := congrArg (weightedCrossSectionalMean π) hy
  have hadd (x x' : CrossSection ι H) :
      weightedCrossSectionalMean π (x + x') =
        weightedCrossSectionalMean π x + weightedCrossSectionalMean π x' := by
    unfold weightedCrossSectionalMean
    simp [smul_add, Finset.sum_add_distrib]
  have hsmul (a : ℝ) (x : CrossSection ι H) :
      weightedCrossSectionalMean π (a • x) =
        a • weightedCrossSectionalMean π x := by
    unfold weightedCrossSectionalMean
    rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Pi.smul_apply, smul_smul]
    rw [mul_comm]
  rw [hadd, hsmul, hsmul, weightedMean_hilbertMulVec_eq_of_stationary hstationary] at hmean
  have hsub : weightedCrossSectionalMean π y -
      ρ • weightedCrossSectionalMean π y =
      (1 - ρ) • weightedCrossSectionalMean π z := by
    calc
      weightedCrossSectionalMean π y - ρ • weightedCrossSectionalMean π y =
          ((1 - ρ) • weightedCrossSectionalMean π z +
            ρ • weightedCrossSectionalMean π y) -
              ρ • weightedCrossSectionalMean π y :=
        congrArg (fun q ↦ q - ρ • weightedCrossSectionalMean π y) hmean
      _ = (1 - ρ) • weightedCrossSectionalMean π z := by abel
  have hscaled : (1 - ρ) • weightedCrossSectionalMean π y =
      (1 - ρ) • weightedCrossSectionalMean π z := by
    calc
      (1 - ρ) • weightedCrossSectionalMean π y =
          weightedCrossSectionalMean π y - ρ • weightedCrossSectionalMean π y := by
        rw [sub_smul, one_smul]
      _ = (1 - ρ) • weightedCrossSectionalMean π z := hsub
  exact smul_right_injective H (ne_of_gt (sub_pos.mpr hρ1)) hscaled

private theorem normalizedSAR_const_of_rowStochastic [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (c : H) :
    normalizedSARLinear ρ W (fun _ ↦ c) = fun _ ↦ c := by
  let a : ℝ := 1 - ρ
  have ha : a ≠ 0 := ne_of_gt (sub_pos.mpr hρ1)
  let v : CrossSection ι H := fun _ ↦ a⁻¹ • c
  have hbase : hilbertMulVec (1 - ρ • W) v = fun _ ↦ c := by
    rw [hilbertMulVec_sub, hilbertMulVec_one, hilbertMulVec_matrix_smul,
      hilbertMulVec_const_of_rowStochastic hW]
    ext i
    change a⁻¹ • c - ρ • (a⁻¹ • c) = c
    rw [smul_smul, ← sub_smul]
    have hscalar : a⁻¹ - ρ * a⁻¹ = 1 := by
      calc
        a⁻¹ - ρ * a⁻¹ = (1 - ρ) * a⁻¹ := by ring
        _ = a * a⁻¹ := by rfl
        _ = 1 := mul_inv_cancel₀ ha
    rw [hscalar, one_smul]
  have hgate := rowStochastic_sar_gate hW hρ0 hρ1
  calc
    normalizedSARLinear ρ W (fun _ ↦ c) =
        a • hilbertMulVec (leontief ρ W) (fun _ ↦ c) := by
      rw [normalizedSARLinear_apply]
    _ = a • hilbertMulVec (leontief ρ W) (hilbertMulVec (1 - ρ • W) v) := by
      rw [hbase]
    _ = a • hilbertMulVec (leontief ρ W * (1 - ρ • W)) v := by
      rw [hilbertMulVec_mul]
    _ = a • hilbertMulVec 1 v := by rw [leontief_mul_one_sub_smul hgate]
    _ = a • v := by rw [hilbertMulVec_one]
    _ = fun _ ↦ c := by
      ext i
      change a • (a⁻¹ • c) = c
      rw [smul_smul, mul_inv_cancel₀ ha, one_smul]

private theorem normalizedSAR_centerWeighted [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hstationary : ∀ j, ∑ i, π i * W i j = π j)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    centerCrossSectionWeighted π (normalizedSARLinear ρ W z) =
      normalizedSARLinear ρ W (centerCrossSectionWeighted π z) := by
  have hmean := weightedMean_normalizedSAR_eq hW hstationary hρ0 hρ1 z
  calc
    centerCrossSectionWeighted π (normalizedSARLinear ρ W z) =
        normalizedSARLinear ρ W z -
          fun _ ↦ weightedCrossSectionalMean π (normalizedSARLinear ρ W z) := rfl
    _ = normalizedSARLinear ρ W z - fun _ ↦ weightedCrossSectionalMean π z := by rw [hmean]
    _ = normalizedSARLinear ρ W z -
        normalizedSARLinear ρ W (fun _ ↦ weightedCrossSectionalMean π z) := by
      rw [normalizedSAR_const_of_rowStochastic hW hρ0 hρ1]
    _ = normalizedSARLinear ρ W
        (z - fun _ ↦ weightedCrossSectionalMean π z) := by rw [LinearMap.map_sub]
    _ = normalizedSARLinear ρ W (centerCrossSectionWeighted π z) := rfl

private theorem normalizedSAR_weightedL2NormBounds [DecidableEq ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hπ0 : ∀ i, 0 ≤ π i)
    (hstationary : ∀ j, ∑ i, π i * W i j = π j)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    ((1 - ρ) / (1 + ρ)) * weightedCrossSectionalL2Norm π z ≤
        weightedCrossSectionalL2Norm π (normalizedSARLinear ρ W z) ∧
      weightedCrossSectionalL2Norm π (normalizedSARLinear ρ W z) ≤
        weightedCrossSectionalL2Norm π z := by
  let y := normalizedSARLinear ρ W z
  have hy : y = (1 - ρ) • z + ρ • hilbertMulVec W y :=
    normalizedSAR_equilibrium (rowStochastic_sar_gate hW hρ0 hρ1) z
  have hWcontract : weightedCrossSectionalL2Norm π (hilbertMulVec W y) ≤
      weightedCrossSectionalL2Norm π y :=
    hilbertMulVec_weightedL2Norm_le_of_rowStochastic hW hπ0 hstationary y
  have ha0 : 0 ≤ 1 - ρ := sub_nonneg.mpr hρ1.le
  have hupper0 : weightedCrossSectionalL2Norm π y ≤
      (1 - ρ) * weightedCrossSectionalL2Norm π z +
        ρ * weightedCrossSectionalL2Norm π y := by
    calc
      weightedCrossSectionalL2Norm π y = weightedCrossSectionalL2Norm π
          ((1 - ρ) • z + ρ • hilbertMulVec W y) := congrArg _ hy
      _ ≤ weightedCrossSectionalL2Norm π ((1 - ρ) • z) +
          weightedCrossSectionalL2Norm π (ρ • hilbertMulVec W y) :=
        weightedCrossSectionalL2Norm_add_le hπ0 _ _
      _ = (1 - ρ) * weightedCrossSectionalL2Norm π z +
          ρ * weightedCrossSectionalL2Norm π (hilbertMulVec W y) := by
        rw [weightedCrossSectionalL2Norm_smul hπ0,
          weightedCrossSectionalL2Norm_smul hπ0, abs_of_nonneg ha0, abs_of_nonneg hρ0]
      _ ≤ (1 - ρ) * weightedCrossSectionalL2Norm π z +
          ρ * weightedCrossSectionalL2Norm π y := by
        exact add_le_add_right (mul_le_mul_of_nonneg_left hWcontract hρ0) _
  have hupper : weightedCrossSectionalL2Norm π y ≤ weightedCrossSectionalL2Norm π z := by
    by_contra hnot
    have hzy : weightedCrossSectionalL2Norm π z < weightedCrossSectionalL2Norm π y :=
      lt_of_not_ge hnot
    nlinarith [mul_pos (sub_pos.mpr hρ1) (sub_pos.mpr hzy)]
  have hsolve : (1 - ρ) • z = y - ρ • hilbertMulVec W y := by
    calc
      (1 - ρ) • z = ((1 - ρ) • z + ρ • hilbertMulVec W y) -
          ρ • hilbertMulVec W y := by abel
      _ = y - ρ • hilbertMulVec W y := by rw [← hy]
  have hlower0 : (1 - ρ) * weightedCrossSectionalL2Norm π z ≤
      (1 + ρ) * weightedCrossSectionalL2Norm π y := by
    calc
      (1 - ρ) * weightedCrossSectionalL2Norm π z =
          weightedCrossSectionalL2Norm π ((1 - ρ) • z) := by
        rw [weightedCrossSectionalL2Norm_smul hπ0, abs_of_nonneg ha0]
      _ = weightedCrossSectionalL2Norm π (y - ρ • hilbertMulVec W y) :=
        congrArg _ hsolve
      _ ≤ weightedCrossSectionalL2Norm π y +
          weightedCrossSectionalL2Norm π (ρ • hilbertMulVec W y) :=
        weightedCrossSectionalL2Norm_sub_le hπ0 _ _
      _ = weightedCrossSectionalL2Norm π y +
          ρ * weightedCrossSectionalL2Norm π (hilbertMulVec W y) := by
        rw [weightedCrossSectionalL2Norm_smul hπ0, abs_of_nonneg hρ0]
      _ ≤ weightedCrossSectionalL2Norm π y +
          ρ * weightedCrossSectionalL2Norm π y := by
        exact add_le_add_right (mul_le_mul_of_nonneg_left hWcontract hρ0) _
      _ = (1 + ρ) * weightedCrossSectionalL2Norm π y := by ring
  constructor
  · rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith : 0 < 1 + ρ)]
    nlinarith
  · change weightedCrossSectionalL2Norm π y ≤ weightedCrossSectionalL2Norm π z
    exact hupper

/-- Paper 5's stationary-weighted spatial-dispersion attenuation theorem.

Let `W` be a nonnegative row-stochastic matrix and let `π` be a strictly positive
probability vector satisfying `πᵀW = πᵀ`.  For every finite nonempty cross-section and
`0 ≤ ρ < 1`, the normalized SAR multiplier obeys
`((1 - ρ)/(1 + ρ))² Dπ(z) ≤ Dπ(Sρz) ≤ Dπ(z)`.  No reversibility, symmetry, or spectral
assumption is required. -/
theorem normalizedSAR_stationaryWeightedDispersionSq_bracket
    [DecidableEq ι] [Nonempty ι]
    {W : Matrix ι ι ℝ} (hW : W ∈ Matrix.rowStochastic ℝ ι)
    {π : ι → ℝ} (hπ : IsStrictlyPositiveStationaryWeight W π)
    {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (z : CrossSection ι H) :
    ((1 - ρ) / (1 + ρ)) ^ 2 * weightedCrossSectionalDispersionSq π z ≤
        weightedCrossSectionalDispersionSq π (normalizedSARLinear ρ W z) ∧
      weightedCrossSectionalDispersionSq π (normalizedSARLinear ρ W z) ≤
        weightedCrossSectionalDispersionSq π z := by
  have hπ0 : ∀ i, 0 ≤ π i := fun i ↦ (hπ.1 i).le
  have hstationary := hπ.2.2
  have hcenter := normalizedSAR_centerWeighted hW hstationary hρ0 hρ1 z
  have hb := normalizedSAR_weightedL2NormBounds hW hπ0 hstationary hρ0 hρ1
    (centerCrossSectionWeighted π z)
  have hzn : 0 ≤ weightedCrossSectionalL2Norm π (centerCrossSectionWeighted π z) :=
    Real.sqrt_nonneg _
  have hTn : 0 ≤ weightedCrossSectionalL2Norm π
      (normalizedSARLinear ρ W (centerCrossSectionWeighted π z)) := Real.sqrt_nonneg _
  have hlower : 0 ≤ (1 - ρ) / (1 + ρ) := by positivity
  have hlowerMul : 0 ≤ ((1 - ρ) / (1 + ρ)) *
      weightedCrossSectionalL2Norm π (centerCrossSectionWeighted π z) :=
    mul_nonneg hlower hzn
  have hloSq := (sq_le_sq₀ hlowerMul hTn).2 hb.1
  have hupSq := (sq_le_sq₀ hTn hzn).2 hb.2
  unfold weightedCrossSectionalDispersionSq
  change ((1 - ρ) / (1 + ρ)) ^ 2 *
        weightedCrossSectionalEnergy π (centerCrossSectionWeighted π z) ≤
      weightedCrossSectionalEnergy π
        (centerCrossSectionWeighted π (normalizedSARLinear ρ W z)) ∧
    weightedCrossSectionalEnergy π
        (centerCrossSectionWeighted π (normalizedSARLinear ρ W z)) ≤
      weightedCrossSectionalEnergy π (centerCrossSectionWeighted π z)
  rw [hcenter, ← weightedCrossSectionalL2Norm_sq hπ0,
    ← weightedCrossSectionalL2Norm_sq hπ0]
  constructor <;> nlinarith

end

end PricingPerspective.Connections.SpatialDispersion
