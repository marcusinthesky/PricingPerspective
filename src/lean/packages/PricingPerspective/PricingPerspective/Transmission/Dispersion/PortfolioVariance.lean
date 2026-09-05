import PricingPerspective.Transmission.Dispersion.Carrier
import PricingPerspective.Transmission.PortfolioLaw
import PricingPerspective.Continuous.Energy.Portfolio
import WassersteinGeometry.Multimarginal

set_option linter.style.longLine false

open scoped InnerProductSpace
open Finset WassersteinGeometry WassersteinGeometry.Multimarginal

/-!
# Information-certified asset and portfolio variance

The new operator is the weighted pairwise dispersion subtracted by Hilbert
polarization.  This file proves the systematic cap, its two-asset bridge to the
existing Paper 1 covariance theorem, and the exact residual-diversification
remainder for standardized total returns.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A H : Type*} [Fintype A]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The real pairwise dispersion certificate used in variance bounds. -/
noncomputable def pairwiseRiskCertificate
    (q : ProbabilityWeight A) (d : A → A → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * (d a b) ^ 2

/-- The systematic-variance cap: weighted marginal variance less certified
pairwise dispersion. -/
noncomputable def systematicVarianceCap
    (q : ProbabilityWeight A) (v : A → ℝ) (d : A → A → ℝ) : ℝ :=
  ∑ a, q a * v a - pairwiseRiskCertificate q d

/-- Exact systematic-variance decomposition under the realized pairwise
Hilbert distances. -/
theorem systematicVariance_eq_cap (q : ProbabilityWeight A) (z : A → H) :
    ‖∑ a, q a • z a‖ ^ 2 =
      systematicVarianceCap q (fun a ↦ ‖z a‖ ^ 2)
        (fun a b ↦ ‖z a - z b‖) := by
  rw [weighted_polarization]
  rfl

/-- A lower certificate on every pairwise loading distance yields an upper
certificate on systematic portfolio variance. -/
theorem systematicVariance_le_cap
    (q : ProbabilityWeight A) (z : A → H) (d : A → A → ℝ)
    (hd : ∀ a b, 0 ≤ d a b)
    (hfloor : ∀ a b, d a b ≤ ‖z a - z b‖) :
    ‖∑ a, q a • z a‖ ^ 2 ≤
      systematicVarianceCap q (fun a ↦ ‖z a‖ ^ 2) d := by
  have hcert : pairwiseRiskCertificate q d ≤
      pairwiseRiskCertificate q (fun a b ↦ ‖z a - z b‖) := by
    unfold pairwiseRiskCertificate
    refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
    refine Finset.sum_le_sum fun a _ ↦ ?_
    refine Finset.sum_le_sum fun b _ ↦ ?_
    have hsq : (d a b) ^ 2 ≤ ‖z a - z b‖ ^ 2 :=
      (sq_le_sq₀ (hd a b) (norm_nonneg _)).mpr (hfloor a b)
    exact mul_le_mul_of_nonneg_left hsq (mul_nonneg (q.nonneg a) (q.nonneg b))
  rw [systematicVariance_eq_cap]
  unfold systematicVarianceCap
  linarith

/-! ### Joint-law variance envelope -/

section JointLaw

variable [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-- The real pairwise-Wasserstein certificate for loading-law marginals. -/
noncomputable def wassersteinPairwiseRiskCertificate
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H) : ℝ :=
  (1 / 2 : ℝ) * ∑ a, ∑ b,
    q a * q b * (WassersteinDistanceSq (P a) (P b)).toReal

omit [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] in
/-- Squaring the real `W₂` definition recovers the real form of `W₂²`. -/
lemma wassersteinDistance_sq_eq_toReal
    (P Q : WassersteinMeasure H) :
    WassersteinDistance P Q ^ 2 = (WassersteinDistanceSq P Q).toReal := by
  unfold WassersteinDistance
  rw [← Real.rpow_natCast, ← Real.rpow_mul ENNReal.toReal_nonneg]
  norm_num

/-- **One coherent joint loading law obeys the pairwise Wasserstein variance cap.**

This is the measure-level asset/portfolio result.  It reuses Paper 1's
two-marginal covariance envelope entrywise, while `J` ensures those entries
come from one globally coherent covariance matrix.
-/
theorem jointPortfolioVariance_le_wassersteinCap
    (J : MeasureTheory.Measure (A → H)) (P : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = J.map fun x : A → H ↦ x a)
    (q : ProbabilityWeight A)
    (hint : ∀ a b : A,
      MeasureTheory.Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J) :
    PricingPerspective.Transmission.portfolioVariance J q ≤
      (∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a)) -
        wassersteinPairwiseRiskCertificate q P := by
  have hentry : ∀ a b,
      PricingPerspective.Transmission.jointCovariance J a b ≤
        (WassersteinGeometry.Hilbert.secondMoment (P a) +
          WassersteinGeometry.Hilbert.secondMoment (P b) -
            (WassersteinDistanceSq (P a) (P b)).toReal) / 2 := by
    intro a b
    rw [PricingPerspective.Transmission.jointCovariance_eq_systCov]
    let π := PricingPerspective.Transmission.jointPairCoupling J P hP a b
    exact WassersteinGeometry.Hilbert.systCov_le_envelope
      π.mem π.int_fst π.int_snd π.int_inner
  rw [PricingPerspective.Transmission.portfolioVariance_eq_quadraticForm J q hint]
  calc
    (∑ a, ∑ b, q a * q b *
        PricingPerspective.Transmission.jointCovariance J a b) ≤
        ∑ a, ∑ b, q a * q b *
          ((WassersteinGeometry.Hilbert.secondMoment (P a) +
            WassersteinGeometry.Hilbert.secondMoment (P b) -
              (WassersteinDistanceSq (P a) (P b)).toReal) / 2) := by
      refine Finset.sum_le_sum fun a _ ↦ ?_
      refine Finset.sum_le_sum fun b _ ↦ ?_
      exact mul_le_mul_of_nonneg_left (hentry a b)
        (mul_nonneg (q.nonneg a) (q.nonneg b))
    _ = (∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a)) -
        wassersteinPairwiseRiskCertificate q P := by
      unfold wassersteinPairwiseRiskCertificate
      have hleft :
          ∑ a, ∑ b, q a * q b * WassersteinGeometry.Hilbert.secondMoment (P a) =
            ∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a) := by
        calc
          ∑ a, ∑ b, q a * q b * WassersteinGeometry.Hilbert.secondMoment (P a) =
              ∑ a, (q a * WassersteinGeometry.Hilbert.secondMoment (P a)) *
                ∑ b, q b := by
            refine Finset.sum_congr rfl fun a _ ↦ ?_
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun b _ ↦ ?_
            ring
          _ = ∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a) := by
            rw [q.mass_one]
            simp
      have hright :
          ∑ a, ∑ b, q a * q b * WassersteinGeometry.Hilbert.secondMoment (P b) =
            ∑ b, q b * WassersteinGeometry.Hilbert.secondMoment (P b) := by
        rw [Finset.sum_comm]
        simpa [mul_comm] using hleft
      have hsplit : ∀ a b,
          q a * q b *
              ((WassersteinGeometry.Hilbert.secondMoment (P a) +
                WassersteinGeometry.Hilbert.secondMoment (P b) -
                  (WassersteinDistanceSq (P a) (P b)).toReal) / 2) =
            (1 / 2 : ℝ) *
                (q a * q b * WassersteinGeometry.Hilbert.secondMoment (P a)) +
              (1 / 2 : ℝ) *
                (q a * q b * WassersteinGeometry.Hilbert.secondMoment (P b)) -
              (1 / 2 : ℝ) *
                (q a * q b * (WassersteinDistanceSq (P a) (P b)).toReal) := by
        intro a b
        ring
      simp_rw [hsplit, Finset.sum_sub_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum]
      rw [hleft, hright]
      ring

/-- A computable nonnegative lower bound on every pairwise `W₂` may replace the
unobserved Wasserstein terms in the coherent-joint-law variance envelope. -/
theorem jointPortfolioVariance_le_pairwiseFloorCap
    (J : MeasureTheory.Measure (A → H)) (P : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = J.map fun x : A → H ↦ x a)
    (q : ProbabilityWeight A) (d : A → A → ℝ)
    (hd : ∀ a b, 0 ≤ d a b)
    (hfloor : ∀ a b, d a b ≤ WassersteinDistance (P a) (P b))
    (hint : ∀ a b : A,
      MeasureTheory.Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J) :
    PricingPerspective.Transmission.portfolioVariance J q ≤
      (∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a)) -
        pairwiseRiskCertificate q d := by
  have hcert : pairwiseRiskCertificate q d ≤ wassersteinPairwiseRiskCertificate q P := by
    unfold pairwiseRiskCertificate wassersteinPairwiseRiskCertificate
    refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
    refine Finset.sum_le_sum fun a _ ↦ ?_
    refine Finset.sum_le_sum fun b _ ↦ ?_
    have hW : 0 ≤ WassersteinDistance (P a) (P b) := by
      unfold WassersteinDistance
      positivity
    have hsq : (d a b) ^ 2 ≤ (WassersteinDistance (P a) (P b)) ^ 2 :=
      (sq_le_sq₀ (hd a b) hW).mpr (hfloor a b)
    rw [wassersteinDistance_sq_eq_toReal] at hsq
    exact mul_le_mul_of_nonneg_left hsq (mul_nonneg (q.nonneg a) (q.nonneg b))
  exact (jointPortfolioVariance_le_wassersteinCap J P hP q hint).trans
    (sub_le_sub_left hcert _)

end JointLaw

/-- **Two-asset Paper 1 bridge.**

The pairwise covariance cap already proved in `Continuous.Energy.Portfolio`
implies the corresponding variance cap for a convex two-asset portfolio.
-/
theorem twoAssetVariance_le_of_pairwiseFloor
    (βi βj : H) (θ d : ℝ)
    (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hd : 0 ≤ d)
    (hfloor : d ≤ ‖βi - βj‖) :
    ‖θ • βi + (1 - θ) • βj‖ ^ 2 ≤
      θ ^ 2 * ‖βi‖ ^ 2 + (1 - θ) ^ 2 * ‖βj‖ ^ 2 +
        2 * θ * (1 - θ) *
          ((1 / 2 : ℝ) * (‖βi‖ ^ 2 + ‖βj‖ ^ 2 - d ^ 2)) := by
  have hcov : ⟪βi, βj⟫_ℝ ≤
      (1 / 2 : ℝ) * (‖βi‖ ^ 2 + ‖βj‖ ^ 2 - d ^ 2) := by
    simpa using PricingPerspective.ContinuousAPT.systematic_covariance_upper_bound
      βi βj 1 d (by norm_num) hd (by simpa using hfloor)
  have hcross :
      2 * θ * (1 - θ) * ⟪βi, βj⟫_ℝ ≤
        2 * θ * (1 - θ) *
          ((1 / 2 : ℝ) * (‖βi‖ ^ 2 + ‖βj‖ ^ 2 - d ^ 2)) :=
    mul_le_mul_of_nonneg_left hcov (mul_nonneg (mul_nonneg (by norm_num) hθ0)
      (sub_nonneg.mpr hθ1))
  rw [norm_add_sq_real, norm_smul, norm_smul, real_inner_smul_left,
    real_inner_smul_right, Real.norm_of_nonneg hθ0,
    Real.norm_of_nonneg (sub_nonneg.mpr hθ1)]
  nlinarith

/-- Diversification in the idiosyncratic variance left after standardization. -/
noncomputable def residualDiversificationRemainder
    (q : ProbabilityWeight A) (v : A → ℝ) : ℝ :=
  ∑ a, q a * (1 - q a) * (1 - v a)

/-- The exact standardized-return cap, before dropping the residual remainder. -/
noncomputable def standardizedVarianceCap
    (q : ProbabilityWeight A) (v : A → ℝ) (d : A → A → ℝ) : ℝ :=
  1 - pairwiseRiskCertificate q d - residualDiversificationRemainder q v

/-- Marginal systematic shares plus diagonal residual variance equal one less
the exact residual-diversification remainder. -/
theorem systematic_add_residual_eq_one_sub_remainder
    (q : ProbabilityWeight A) (v : A → ℝ) :
    (∑ a, q a * v a) + (∑ a, (q a) ^ 2 * (1 - v a)) =
      1 - residualDiversificationRemainder q v := by
  calc
    (∑ a, q a * v a) + (∑ a, (q a) ^ 2 * (1 - v a)) =
        ∑ a, (q a * v a + (q a) ^ 2 * (1 - v a)) := by
          rw [Finset.sum_add_distrib]
    _ = ∑ a, (q a - q a * (1 - q a) * (1 - v a)) := by
      refine Finset.sum_congr rfl fun a _ ↦ ?_
      ring
    _ = (∑ a, q a) - ∑ a, q a * (1 - q a) * (1 - v a) := by
      rw [Finset.sum_sub_distrib]
    _ = 1 - residualDiversificationRemainder q v := by
      rw [q.mass_one]
      rfl

/-- Systematic and diagonal-residual bounds combine into the exact
standardized total-return cap. -/
theorem totalVariance_le_standardizedVarianceCap
    (q : ProbabilityWeight A) (v : A → ℝ) (d : A → A → ℝ)
    (systematic residual total : ℝ)
    (htotal : total = systematic + residual)
    (hsystematic : systematic ≤ ∑ a, q a * v a - pairwiseRiskCertificate q d)
    (hresidual : residual ≤ ∑ a, (q a) ^ 2 * (1 - v a)) :
    total ≤ standardizedVarianceCap q v d := by
  have hid := systematic_add_residual_eq_one_sub_remainder q v
  unfold standardizedVarianceCap
  linarith

/-- A probability weight never exceeds one. -/
lemma probabilityWeight_le_one (q : ProbabilityWeight A) (a : A) : q a ≤ 1 := by
  rw [← q.mass_one]
  exact Finset.single_le_sum (fun b _ ↦ q.nonneg b) (Finset.mem_univ a)

/-- The exact standardized cap is no larger than the headline
`1 - certificate` bound when systematic shares lie in `[0,1]`. -/
theorem standardizedVarianceCap_le_headline
    (q : ProbabilityWeight A) (v : A → ℝ) (d : A → A → ℝ)
    (hv : ∀ a, v a ≤ 1) :
    standardizedVarianceCap q v d ≤ 1 - pairwiseRiskCertificate q d := by
  have hrem : 0 ≤ residualDiversificationRemainder q v := by
    unfold residualDiversificationRemainder
    exact Finset.sum_nonneg fun a _ ↦
      mul_nonneg (mul_nonneg (q.nonneg a) (sub_nonneg.mpr (probabilityWeight_le_one q a)))
        (sub_nonneg.mpr (hv a))
  unfold standardizedVarianceCap
  linarith

/-- The raw-return cap obtained by multiplying a normalized certificate by the
squared gross volatility scale. -/
noncomputable def rawVarianceCap
    (scale : ℝ) (q : ProbabilityWeight A) (d : A → A → ℝ) : ℝ :=
  scale ^ 2 * (1 - pairwiseRiskCertificate q d)

/-- Scaling the standardized portfolio preserves its certified upper bound. -/
theorem rawVariance_le_cap
    (scale standardizedVariance : ℝ) (q : ProbabilityWeight A)
    (d : A → A → ℝ) (hstandardized :
      standardizedVariance ≤ 1 - pairwiseRiskCertificate q d) :
    scale ^ 2 * standardizedVariance ≤ rawVarianceCap scale q d := by
  unfold rawVarianceCap
  exact mul_le_mul_of_nonneg_left hstandardized (sq_nonneg scale)

end PricingPerspective.Transmission.Dispersion
