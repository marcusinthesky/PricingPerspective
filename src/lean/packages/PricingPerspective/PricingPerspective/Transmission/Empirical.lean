import PricingPerspective.Transmission.Frechet
import PricingPerspective.Transmission.Defs

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry WassersteinGeometry.Hilbert Finset

/-!
# The empirical layer is an estimator of the measure layer

`Transmission.Defs` develops the envelope for finite weighted point clouds, because that is
what the pipeline actually computes: `m` article embeddings per firm, and a coupling that is
literally an `m × m` weight matrix. `Transmission.Frechet` develops the same envelope for
laws in `P₂(H)`, which is what the manuscript states. This file connects them.

Without the connection the two developments are merely analogous, and a manuscript claim
proved about clouds would not certify a statement about laws. What is proved below is that
the finite objects *are* the measure objects, evaluated at finitely supported measures:

* `ofCloud` sends a weighted cloud to the finite-support measure `∑ₐ pₐ δ_{Zₐ}`;
* `integral_ofCloud` turns every integral against it into the corresponding finite sum, so
  `secondMoment`, `systCov` and `transportCost` agree on the nose
  (`secondMoment_ofCloud`, `systCov_ofCoupling`, `transportCost_ofCoupling`);
* `ofCoupling_mem_couplingSet` shows a finite coupling is a coupling of the induced laws.

The payoff is `RandomExposure.systCov_le_envelope`: the covariance the pipeline computes from
a finite coupling is bounded by the measure-level envelope of the induced laws. The finite
computation is an estimator of the population object, not a different object.

## One direction only

`ofCoupling_mem_couplingSet` embeds finite couplings into `Π(P, Q)`; it does **not** say
every coupling of two finitely-supported laws is finitely supported. That is true, and would
upgrade a `FiniteOptimalCostCertificate` into an `OptimalTransportCost`, but it needs a
support argument this file does not make. So a finite optimality certificate still certifies
minimality only among finite couplings, and `RandomExposure.systCov_le_envelope` is stated
against a measure-level certificate rather than a finite one. That gap is deliberate and is
the honest reading of what the pipeline computes.

## References

* `Transmission.Defs`, "Why finite index sets".
-/

namespace PricingPerspective.Transmission

open RandomExposure

variable {ι ι' : Type*} [Fintype ι] [Fintype ι']

/-! ## Construction

The finite-support measures themselves need nothing but a measurable space; the Hilbert
structure enters only where the *functionals* do, in the bridge section below. Splitting the
variable block accordingly keeps each statement honest about what it uses. -/

section Construction

variable {H : Type*} [MeasurableSpace H]

/-! ### The finite-support measure carried by a cloud -/

/-- The law of a finite weighted point cloud: `∑ₐ pₐ δ_{Zₐ}`. -/
noncomputable def ofCloud (p : ι → ℝ) (Z : ι → H) : Measure H :=
  ∑ a, (ENNReal.ofReal (p a)) • Measure.dirac (Z a)

/-- Every function is integrable against a finite-support law. -/
lemma integrable_ofCloud (p : ι → ℝ) (Z : ι → H) {f : H → ℝ} (hf : StronglyMeasurable f) :
    Integrable f (ofCloud p Z) := by
  refine integrable_finsetSum_measure.2 fun a _ => ?_
  exact (integrable_dirac' hf (by simp)).smul_measure (by simp)

/-- Integration against a cloud law is the corresponding finite sum. -/
theorem integral_ofCloud (p : ι → ℝ) (hp : ∀ a, 0 ≤ p a) (Z : ι → H)
    {f : H → ℝ} (hf : StronglyMeasurable f) :
    ∫ x, f x ∂(ofCloud p Z) = ∑ a, p a * f (Z a) := by
  unfold ofCloud
  rw [integral_finsetSum_measure fun a _ =>
    (integrable_dirac' hf (by simp)).smul_measure (by simp)]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [integral_smul_measure, integral_dirac' f _ hf, ENNReal.toReal_ofReal (hp a),
    smul_eq_mul]

/-- A cloud law is a probability measure. -/
lemma isProbabilityMeasure_ofCloud {p : ι → ℝ} (hp : ∀ a, 0 ≤ p a)
    (hmass : ∑ a, p a = 1) (Z : ι → H) : IsProbabilityMeasure (ofCloud p Z) := by
  constructor
  unfold ofCloud
  simp only [Measure.coe_finsetSum, Finset.sum_apply, Measure.smul_apply,
    Measure.dirac_apply_of_mem (Set.mem_univ _), smul_eq_mul, mul_one]
  rw [← ENNReal.ofReal_sum_of_nonneg fun a _ => hp a, hmass, ENNReal.ofReal_one]

/-! ### The joint law carried by a finite coupling -/

variable {p : ι → ℝ} {q : ι' → ℝ}

/-- The joint law of a finite coupling: `∑ₐ∑_b w(a,b) δ_{(Zₐ, Y_b)}`. -/
noncomputable def ofCoupling (π : Coupling p q) (Z : ι → H) (Y : ι' → H) : Measure (H × H) :=
  ∑ a, ∑ b, (ENNReal.ofReal (π.w a b)) • Measure.dirac (Z a, Y b)

/-- Every function is integrable against a finite coupling's joint law. -/
lemma integrable_ofCoupling (π : Coupling p q) (Z : ι → H) (Y : ι' → H)
    {f : H × H → ℝ} (hf : StronglyMeasurable f) :
    Integrable f (ofCoupling π Z Y) := by
  refine integrable_finsetSum_measure.2 fun a _ => ?_
  refine integrable_finsetSum_measure.2 fun b _ => ?_
  exact (integrable_dirac' hf (by simp)).smul_measure (by simp)

/-- Integration against a coupling's joint law is the corresponding double sum. -/
theorem integral_ofCoupling (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H) {f : H × H → ℝ} (hf : StronglyMeasurable f) :
    ∫ z, f z ∂(ofCoupling π Z Y) = ∑ a, ∑ b, π.w a b * f (Z a, Y b) := by
  unfold ofCoupling
  rw [integral_finsetSum_measure fun a _ =>
    integrable_finsetSum_measure.2 fun b _ =>
      (integrable_dirac' hf (by simp)).smul_measure (by simp)]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [integral_finsetSum_measure fun b _ =>
    (integrable_dirac' hf (by simp)).smul_measure (by simp)]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [integral_smul_measure, integral_dirac' f _ hf, ENNReal.toReal_ofReal (hw a b),
    smul_eq_mul]

/-- A finite coupling's joint law has the induced cloud laws as marginals. -/
theorem ofCoupling_mem_couplingSet (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H) :
    ofCoupling π Z Y ∈ couplingSet (ofCloud p Z) (ofCloud q Y) := by
  constructor
  · ext s hs
    rw [Measure.map_apply measurable_fst hs]
    unfold ofCoupling ofCloud
    simp only [Measure.coe_finsetSum, Finset.sum_apply, Measure.smul_apply, smul_eq_mul,
      Measure.dirac_apply' _ (measurable_fst hs), Measure.dirac_apply' _ hs]
    refine Finset.sum_congr rfl fun a _ => ?_
    -- The indicator does not see the second coordinate, so it factors out of the `b` sum.
    have hind : ∀ b : ι', (Prod.fst ⁻¹' s).indicator (1 : H × H → ℝ≥0∞) (Z a, Y b)
        = s.indicator (1 : H → ℝ≥0∞) (Z a) := by
      intro b; by_cases h : Z a ∈ s <;> simp [h]
    simp_rw [hind]
    rw [← Finset.sum_mul, ← ENNReal.ofReal_sum_of_nonneg fun b _ => hw a b,
      π.marginal_fst a]
  · ext s hs
    rw [Measure.map_apply measurable_snd hs]
    unfold ofCoupling ofCloud
    simp only [Measure.coe_finsetSum, Finset.sum_apply, Measure.smul_apply, smul_eq_mul,
      Measure.dirac_apply' _ (measurable_snd hs), Measure.dirac_apply' _ hs]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    have hind : ∀ a : ι, (Prod.snd ⁻¹' s).indicator (1 : H × H → ℝ≥0∞) (Z a, Y b)
        = s.indicator (1 : H → ℝ≥0∞) (Y b) := by
      intro a; by_cases h : Y b ∈ s <;> simp [h]
    simp_rw [hind]
    rw [← Finset.sum_mul, ← ENNReal.ofReal_sum_of_nonneg fun a _ => hw a b,
      π.marginal_snd b]

end Construction

/-! ## Bridge

From here on the Hilbert structure is genuinely used: these are the statements that the
finite functionals *are* the measure-level ones. -/

section Bridge

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]
variable {p : ι → ℝ} {q : ι' → ℝ}

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- The second moment of a cloud law is the finite second moment. -/
theorem secondMoment_ofCloud (p : ι → ℝ) (hp : ∀ a, 0 ≤ p a) (Z : ι → H)
    (P : WassersteinMeasure H) (hP : P.measure = ofCloud p Z) :
    Hilbert.secondMoment P = RandomExposure.secondMoment p Z := by
  unfold Hilbert.secondMoment RandomExposure.secondMoment
  rw [hP, integral_ofCloud p hp Z ((continuous_norm.pow 2).stronglyMeasurable)]

omit [Inhabited H] in
/-- Systematic covariance: the finite double sum is the Bochner integral. -/
theorem systCov_ofCoupling (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H) :
    Hilbert.systCov (ofCoupling π Z Y) = RandomExposure.systCov π Z Y := by
  unfold Hilbert.systCov RandomExposure.systCov
  exact integral_ofCoupling π hw Z Y
    (by exact (continuous_inner (𝕜 := ℝ)).stronglyMeasurable)

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- Transport cost: the finite double sum is the Bochner integral. -/
theorem transportCost_ofCoupling (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H) :
    Hilbert.transportCost (ofCoupling π Z Y) = RandomExposure.transportCost π Z Y := by
  unfold Hilbert.transportCost RandomExposure.transportCost
  exact integral_ofCoupling π hw Z Y
    (by exact ((continuous_norm.comp (continuous_fst.sub continuous_snd)).pow 2).stronglyMeasurable)

/-! ### The empirical coupling as an integrable coupling of the induced laws -/

/-- A finite coupling induces an `IntegrableCoupling` of the induced loading laws.

    Integrability is automatic: a finite-support law integrates everything measurable. -/
noncomputable def toIntegrableCoupling
    {P Q : WassersteinMeasure H} (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H)
    (hP : P.measure = ofCloud p Z) (hQ : Q.measure = ofCloud q Y) :
    IntegrableCoupling P Q where
  toMeasure := ofCoupling π Z Y
  mem := by rw [hP, hQ]; exact ofCoupling_mem_couplingSet π hw Z Y
  int_fst := integrable_ofCoupling π Z Y
    (((continuous_norm.comp continuous_fst).pow 2).stronglyMeasurable)
  int_snd := integrable_ofCoupling π Z Y
    (((continuous_norm.comp continuous_snd).pow 2).stronglyMeasurable)
  int_inner := integrable_ofCoupling π Z Y
    (by exact (continuous_inner (𝕜 := ℝ)).stronglyMeasurable)

/-- **The pipeline's covariance is bounded by the population envelope.**

    The systematic covariance computed from a finite coupling of two article clouds lies
    under the measure-level ceiling of the laws those clouds induce.

    This is what makes the finite development an *estimator* of the measure-level one rather
    than a separate model: the number the pipeline reports is a member of the Fréchet class
    the manuscript's theorem is about. -/
theorem RandomExposure.systCov_le_envelope
    {P Q : WassersteinMeasure H} (π : Coupling p q) (hw : ∀ a b, 0 ≤ π.w a b)
    (Z : ι → H) (Y : ι' → H)
    (hP : P.measure = ofCloud p Z) (hQ : Q.measure = ofCloud q Y)
    (cert : OptimalTransportCost P Q) :
    RandomExposure.systCov π Z Y ≤ ceiling cert := by
  have hmem : Hilbert.systCov (toIntegrableCoupling π hw Z Y hP hQ).toMeasure ∈
      frechetClass P Q := mem_frechetClass _
  have hle := (isGreatest_frechetClass cert).2 hmem
  rwa [show (toIntegrableCoupling π hw Z Y hP hQ).toMeasure = ofCoupling π Z Y from rfl,
    systCov_ofCoupling π hw Z Y] at hle

end Bridge

end PricingPerspective.Transmission
