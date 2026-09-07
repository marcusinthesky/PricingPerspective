import WassersteinGeometry.Geodesics

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory Finset

/-!
# Finite multi-marginal quadratic dispersion

This file defines probability weights, coherent finite joint couplings, and the
multi-marginal quadratic-dispersion infimum.  The construction is metric-only;
Hilbert polarization and portfolio interpretations live in downstream modules.

The optimizer is represented by a supplied certificate.  This matches the
certificate-first optimal-coupling API already used by `WassersteinGeometry` and
does not assert an optimal-plan existence theorem.
-/

namespace WassersteinGeometry.Multimarginal

variable {A Ω : Type*} [Fintype A]

/-- Long-only finite weights of total mass one. -/
structure ProbabilityWeight (A : Type*) [Fintype A] where
  /-- The real weight assigned to an index. -/
  weight : A → ℝ
  /-- Every weight is nonnegative. -/
  nonneg : ∀ a, 0 ≤ weight a
  /-- The weights have total mass one. -/
  mass_one : ∑ a, weight a = 1

namespace ProbabilityWeight

instance : CoeFun (ProbabilityWeight A) (fun _ ↦ A → ℝ) := ⟨ProbabilityWeight.weight⟩

/-- A convex combination of two probability weights. -/
def mix (q r : ProbabilityWeight A) (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    ProbabilityWeight A where
  weight a := θ * q a + (1 - θ) * r a
  nonneg a := add_nonneg (mul_nonneg hθ0 (q.nonneg a))
    (mul_nonneg (sub_nonneg.mpr hθ1) (r.nonneg a))
  mass_one := by
    simp_rw [Finset.sum_add_distrib, ← Finset.mul_sum, q.mass_one, r.mass_one]
    ring

end ProbabilityWeight

variable [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]

/-- A law on all coordinates whose coordinate marginals are the prescribed laws. -/
structure JointCoupling (P : A → WassersteinMeasure Ω) where
  /-- The coherent joint law. -/
  measure : Measure (A → Ω)
  /-- Every coordinate projection has its prescribed marginal. -/
  marginal : ∀ a, measure.map (fun x ↦ x a) = (P a).measure

/-- The independent product of a finite probability-law family is always a
coherent joint coupling.  It supplies nonemptiness without asserting that the
independent plan is dispersion-optimal. -/
noncomputable def independentJointCoupling
    (P : A → WassersteinMeasure Ω) : JointCoupling P := by
  classical
  letI : ∀ a, IsProbabilityMeasure (P a).measure :=
    fun a ↦ ⟨(P a).is_probability⟩
  exact
    { measure := Measure.pi fun a ↦ (P a).measure
      marginal := fun a ↦ by
        rw [Measure.pi_map_eval]
        simp }

/-- The extended-real quadratic dispersion cost of one coherent joint coupling.

The ordered double sum carries a factor `1/2`, avoiding an arbitrary ordering on
the asset index while representing `∑_{i<j} qᵢqⱼ d(Xᵢ,Xⱼ)²` for symmetric costs.
-/
noncomputable def weightedDispersionCost
    {P : A → WassersteinMeasure Ω} (q : ProbabilityWeight A) (J : JointCoupling P) : ℝ≥0∞ :=
  ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
    ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure

/-- The least weighted quadratic dispersion over coherent joint couplings. -/
noncomputable def wassersteinDispersionSq
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) : ℝ≥0∞ :=
  sInf {c | ∃ J : JointCoupling P, c = weightedDispersionCost q J}

/-- The real square-root multi-marginal dispersion.

As with `WassersteinDistance`, callers use supplied finiteness/optimality
certificates when converting extended-real squared costs into real algebra.
-/
noncomputable def wassersteinDispersion
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) : ℝ :=
  (wassersteinDispersionSq q P).toReal ^ (1 / 2 : ℝ)

/-- Multi-marginal Wasserstein dispersion is nonnegative. -/
theorem wassersteinDispersion_nonneg
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) :
    0 ≤ wassersteinDispersion q P := by
  unfold wassersteinDispersion
  positivity

/-- Squaring finite real multi-marginal dispersion recovers the real form of
its extended-real quadratic infimum. -/
theorem wassersteinDispersion_sq_eq_toReal
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) :
    wassersteinDispersion q P ^ 2 = (wassersteinDispersionSq q P).toReal := by
  unfold wassersteinDispersion
  rw [← Real.rpow_natCast, ← Real.rpow_mul ENNReal.toReal_nonneg]
  norm_num

/-- A supplied joint plan certifying the multi-marginal dispersion minimum.

Existence is deliberately not asserted.  The certificate is the multi-marginal
counterpart of `OptimalCoupling` and Paper 1's `OptimalTransportCost`.
-/
structure OptimalJointDispersion
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) where
  /-- A joint coupling attaining the claimed minimum. -/
  plan : JointCoupling P
  /-- The certified minimum value. -/
  value : ℝ≥0∞
  /-- The supplied plan attains the value. -/
  plan_cost : weightedDispersionCost q plan = value
  /-- No coherent joint coupling has lower cost. -/
  minimal : ∀ J : JointCoupling P, value ≤ weightedDispersionCost q J

/-- A certified value agrees with the infimum definition. -/
theorem OptimalJointDispersion.value_eq_wassersteinDispersionSq
    {q : ProbabilityWeight A} {P : A → WassersteinMeasure Ω}
    (cert : OptimalJointDispersion q P) :
    cert.value = wassersteinDispersionSq q P := by
  apply le_antisymm
  · refine le_sInf ?_
    rintro c ⟨J, rfl⟩
    exact cert.minimal J
  · unfold wassersteinDispersionSq
    exact sInf_le ⟨cert.plan, cert.plan_cost.symm⟩

end WassersteinGeometry.Multimarginal
