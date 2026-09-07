import Mathlib.Analysis.MeanInequalities
import WassersteinGeometry.Multimarginal.Hilbert

open scoped ENNReal NNReal InnerProductSpace
open Finset

/-!
# Certificate-first Hilbert--Wasserstein barycenters

This file defines the weighted Fréchet objective on a finite family of
`WassersteinMeasure`s and proves that the square root of its infimum is 1-Lipschitz in the
weighted product Wasserstein metric.  That stability theorem is unconditional and does not
assert that a barycenter exists.

The exact identity with multi-marginal quadratic dispersion is isolated behind
`HilbertBarycenterBridge`.  Its two constructions are the standard weighted-mean pushforward
and finite gluing arguments of Agueh--Carlier (2011).  Keeping them in a supplied certificate
matches the repository's certificate-first transport API: no optimizer-existence theorem or
additional trust premise is introduced.
-/

namespace WassersteinGeometry.Multimarginal

variable {A H : Type*} [Fintype A]
  [MeasurableSpace H] [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [CompleteSpace H] [SecondCountableTopology H] [BorelSpace H] [Inhabited H]

/-- The weighted quadratic Fréchet objective at a candidate Wasserstein barycenter.

For weights `q`, marginal laws `P`, and candidate center `Q`, this is
`sum a, q a * W₂(P a, Q)²`.  It is real-valued because `WassersteinMeasure` contains the
finite-second-moment premise needed for finite Wasserstein distances.
-/
noncomputable def weightedFrechetCost (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) : ℝ :=
  ∑ a, q a * dist (P a) Q ^ 2

/-- Extended-real form of the weighted quadratic Fréchet objective.

The coupling orientation is `Q` first and `P a` second, matching the common-first-marginal
gluing theorem used by the unconditional MMOT--barycenter equivalence.
-/
noncomputable def weightedFrechetCostENNReal (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) : ℝ≥0∞ :=
  ∑ a, ENNReal.ofReal (q a) * WassersteinDistanceSq Q (P a)

/-- The square root of the weighted quadratic Fréchet objective.

This is the distance from a tuple of laws to one point on the diagonal of the weighted
product Wasserstein space.
-/
noncomputable def weightedFrechetRadius (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) : ℝ :=
  √(weightedFrechetCost q P Q)

/-- The infimum of the weighted quadratic Fréchet objective over candidate centers.

This definition does not claim that the infimum is attained.  It is the barycenter-side
quantity that the Hilbert multi-marginal bridge identifies with `wassersteinDispersionSq`.
-/
noncomputable def wassersteinBarycenterDispersionSq (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) : ℝ :=
  sInf (Set.range (weightedFrechetCost q P))

/-- The square root of the infimal weighted Fréchet objective.

This is the distance from the tuple `P` to the diagonal in the weighted product
Wasserstein pseudometric.
-/
noncomputable def wassersteinBarycenterDispersion (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) : ℝ :=
  √(wassersteinBarycenterDispersionSq q P)

omit [InnerProductSpace ℝ H] in
/-- The weighted Fréchet objective is nonnegative under nonnegative probability weights. -/
lemma weightedFrechetCost_nonneg (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) :
    0 ≤ weightedFrechetCost q P Q := by
  unfold weightedFrechetCost
  exact Finset.sum_nonneg fun a _ ↦ mul_nonneg (q.nonneg a) (sq_nonneg _)

omit [InnerProductSpace ℝ H] in
/-- Squaring the real Wasserstein distance recovers the real form of `W₂²`. -/
lemma wassersteinDistance_sq_eq_toReal (P Q : WassersteinMeasure H) :
    dist P Q ^ 2 = (WassersteinDistanceSq P Q).toReal := by
  change WassersteinDistance P Q ^ 2 = (WassersteinDistanceSq P Q).toReal
  unfold WassersteinDistance
  rw [← Real.rpow_natCast, ← Real.rpow_mul ENNReal.toReal_nonneg]
  norm_num

omit [InnerProductSpace ℝ H] in
/-- Squared Wasserstein distance is symmetric in extended-real form. -/
lemma wassersteinDistanceSq_symm (P Q : WassersteinMeasure H) :
    WassersteinDistanceSq P Q = WassersteinDistanceSq Q P := by
  apply (ENNReal.toReal_eq_toReal_iff' (wassersteinDistanceSq_lt_top P Q).ne
    (wassersteinDistanceSq_lt_top Q P).ne).mp
  rw [← wassersteinDistance_sq_eq_toReal, ← wassersteinDistance_sq_eq_toReal, dist_comm]

omit [InnerProductSpace ℝ H] in
/-- The extended-real Fréchet objective is the `ENNReal.ofReal` image of the real one. -/
lemma weightedFrechetCostENNReal_eq_ofReal (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) :
    weightedFrechetCostENNReal q P Q = ENNReal.ofReal (weightedFrechetCost q P Q) := by
  unfold weightedFrechetCostENNReal weightedFrechetCost
  rw [ENNReal.ofReal_sum_of_nonneg]
  · apply Finset.sum_congr rfl
    intro a _
    rw [ENNReal.ofReal_mul (q.nonneg a), wassersteinDistanceSq_symm Q (P a),
      wassersteinDistance_sq_eq_toReal,
      ENNReal.ofReal_toReal (wassersteinDistanceSq_lt_top (P a) Q).ne]
  · intro a _
    exact mul_nonneg (q.nonneg a) (sq_nonneg _)

omit [InnerProductSpace ℝ H] [CompleteSpace H] in
/-- Every squared Wasserstein infimum admits a coupling within any positive tolerance.

This is selection from the `sInf` definition only; it does not assert that an optimal coupling
exists.
-/
lemma exists_coupling_cost_lt_add (P Q : WassersteinMeasure H) (ε : ℝ≥0) (hε : 0 < ε) :
    ∃ π ∈ couplingSet P.measure Q.measure,
      ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π < WassersteinDistanceSq P Q + (ε : ℝ≥0∞) := by
  let S : Set ℝ≥0∞ :=
    {c | ∃ π ∈ couplingSet P.measure Q.measure,
      c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π}
  have hW : WassersteinDistanceSq P Q = sInf S := rfl
  have hlt : sInf S < sInf S + (ε : ℝ≥0∞) := by
    apply ENNReal.lt_add_right
    · rw [← hW]
      exact (wassersteinDistanceSq_lt_top P Q).ne
    · exact_mod_cast hε.ne'
  obtain ⟨c, ⟨π, hπ, rfl⟩, hc⟩ := sInf_lt_iff.mp hlt
  exact ⟨π, hπ, by simpa [hW] using hc⟩

omit [InnerProductSpace ℝ H] in
/-- Weighted Euclidean Minkowski plus the Wasserstein triangle inequality.

The result compares the Fréchet radius of `P` around `R` with the coordinatewise
Wasserstein perturbation from `P` to `Q` and the radius of `Q` around `R`.
-/
lemma weightedFrechetRadius_triangle (q : ProbabilityWeight A)
    (P Q : A → WassersteinMeasure H) (R : WassersteinMeasure H) :
    weightedFrechetRadius q P R ≤
      √(∑ a, q a * dist (P a) (Q a) ^ 2) + weightedFrechetRadius q Q R := by
  let f : A → ℝ := fun a ↦ √(q a) * dist (P a) (Q a)
  let g : A → ℝ := fun a ↦ √(q a) * dist (Q a) R
  have hcoord (a : A) :
      √(q a) * dist (P a) R ≤ f a + g a := by
    dsimp [f, g]
    simpa [mul_add] using
      mul_le_mul_of_nonneg_left (dist_triangle (P a) (Q a) R)
        (Real.sqrt_nonneg (q a))
  have hweight (a : A) (d : ℝ) (hd : 0 ≤ d) :
      |√(q a) * d| ^ (2 : ℝ) = q a * d ^ 2 := by
    rw [abs_of_nonneg (mul_nonneg (Real.sqrt_nonneg _) hd), Real.rpow_two, mul_pow,
      Real.sq_sqrt (q.nonneg a)]
  have hsum :
      ∑ a, |√(q a) * dist (P a) R| ^ (2 : ℝ) ≤
        ∑ a, |f a + g a| ^ (2 : ℝ) := by
    refine Finset.sum_le_sum fun a _ ↦ ?_
    have hl : 0 ≤ √(q a) * dist (P a) R :=
      mul_nonneg (Real.sqrt_nonneg _) dist_nonneg
    have hr : 0 ≤ f a + g a := by
      dsimp [f, g]
      positivity
    simpa [abs_of_nonneg hl, abs_of_nonneg hr] using
      (Real.rpow_le_rpow hl (hcoord a) (by norm_num : (0 : ℝ) ≤ 2))
  have hmink := Real.Lp_add_le (s := (Finset.univ : Finset A))
    (f := f) (g := g) (p := (2 : ℝ)) (by norm_num)
  unfold weightedFrechetRadius weightedFrechetCost
  rw [Real.sqrt_eq_rpow]
  calc
    (∑ a, q a * dist (P a) R ^ 2) ^ (1 / 2 : ℝ) =
        (∑ a, |√(q a) * dist (P a) R| ^ (2 : ℝ)) ^ (1 / 2 : ℝ) := by
      congr 1
      exact Finset.sum_congr rfl fun a _ ↦ (hweight a _ dist_nonneg).symm
    _ ≤ (∑ a, |f a + g a| ^ (2 : ℝ)) ^ (1 / 2 : ℝ) := by
      exact Real.rpow_le_rpow (by positivity) hsum (by norm_num)
    _ ≤ (∑ a, |f a| ^ (2 : ℝ)) ^ (1 / 2 : ℝ) +
        (∑ a, |g a| ^ (2 : ℝ)) ^ (1 / 2 : ℝ) := hmink
    _ = √(∑ a, q a * dist (P a) (Q a) ^ 2) +
        √(∑ a, q a * dist (Q a) R ^ 2) := by
      rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
      have hfSum : (∑ a, |f a| ^ (2 : ℝ)) =
          ∑ a, q a * dist (P a) (Q a) ^ 2 := by
        exact Finset.sum_congr rfl fun a _ ↦ by
          simpa [f] using hweight a _ dist_nonneg
      have hgSum : (∑ a, |g a| ^ (2 : ℝ)) =
          ∑ a, q a * dist (Q a) R ^ 2 := by
        exact Finset.sum_congr rfl fun a _ ↦ by
          simpa [g] using hweight a _ dist_nonneg
      rw [hfSum, hgSum]

private lemma ProbabilityWeight.index_nonempty (q : ProbabilityWeight A) : Nonempty A := by
  rw [← Finset.univ_nonempty_iff]
  by_contra h
  have hmass := q.mass_one
  rw [Finset.not_nonempty_iff_eq_empty.mp h] at hmass
  simp at hmass

omit [InnerProductSpace ℝ H] in
/-- The infimal weighted Fréchet objective is nonnegative. -/
lemma wassersteinBarycenterDispersionSq_nonneg (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) :
    0 ≤ wassersteinBarycenterDispersionSq q P := by
  unfold wassersteinBarycenterDispersionSq
  have hrange : (Set.range (weightedFrechetCost q P)).Nonempty := by
    let a := Classical.choice q.index_nonempty
    exact ⟨weightedFrechetCost q P (P a), ⟨P a, rfl⟩⟩
  refine le_csInf hrange ?_
  rintro c ⟨Q, rfl⟩
  exact weightedFrechetCost_nonneg q P Q

omit [InnerProductSpace ℝ H] in
/-- The infimal Fréchet radius is no larger than the radius at any supplied center. -/
lemma wassersteinBarycenterDispersion_le_radius (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) (Q : WassersteinMeasure H) :
    wassersteinBarycenterDispersion q P ≤ weightedFrechetRadius q P Q := by
  unfold wassersteinBarycenterDispersion weightedFrechetRadius
  apply Real.sqrt_le_sqrt
  unfold wassersteinBarycenterDispersionSq
  exact csInf_le
    ⟨0, fun c ⟨R, hr⟩ ↦ hr ▸ weightedFrechetCost_nonneg q P R⟩ ⟨Q, rfl⟩

omit [InnerProductSpace ℝ H] in
/-- A common lower bound on every Fréchet radius bounds the infimal radius. -/
lemma le_wassersteinBarycenterDispersion (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) {r : ℝ}
    (hr : ∀ Q, r ≤ weightedFrechetRadius q P Q) :
    r ≤ wassersteinBarycenterDispersion q P := by
  by_cases hr0 : r ≤ 0
  · exact hr0.trans (Real.sqrt_nonneg _)
  have hr0' : 0 ≤ r := le_of_not_ge hr0
  unfold wassersteinBarycenterDispersion
  rw [Real.le_sqrt hr0' (wassersteinBarycenterDispersionSq_nonneg q P)]
  unfold wassersteinBarycenterDispersionSq
  have hrange : (Set.range (weightedFrechetCost q P)).Nonempty := by
    let a := Classical.choice q.index_nonempty
    exact ⟨weightedFrechetCost q P (P a), ⟨P a, rfl⟩⟩
  refine le_csInf hrange ?_
  rintro c ⟨Q, rfl⟩
  have hQ := hr Q
  unfold weightedFrechetRadius at hQ
  exact (Real.le_sqrt hr0' (weightedFrechetCost_nonneg q P Q)).mp hQ

omit [InnerProductSpace ℝ H] in
/-- One-sided stability of the infimal Fréchet radius under marginal perturbations. -/
theorem wassersteinBarycenterDispersion_le_add (q : ProbabilityWeight A)
    (P Q : A → WassersteinMeasure H) :
    wassersteinBarycenterDispersion q P ≤
      √(∑ a, q a * dist (P a) (Q a) ^ 2) +
        wassersteinBarycenterDispersion q Q := by
  let δ := √(∑ a, q a * dist (P a) (Q a) ^ 2)
  have hlower : wassersteinBarycenterDispersion q P - δ ≤
      wassersteinBarycenterDispersion q Q := by
    apply le_wassersteinBarycenterDispersion q Q
    intro R
    have hP := wassersteinBarycenterDispersion_le_radius q P R
    have htri := weightedFrechetRadius_triangle q P Q R
    change weightedFrechetRadius q P R ≤ δ + weightedFrechetRadius q Q R at htri
    linarith
  change wassersteinBarycenterDispersion q P ≤
    δ + wassersteinBarycenterDispersion q Q
  linarith

omit [InnerProductSpace ℝ H] in
/-- The square-root barycenter objective is 1-Lipschitz in weighted product `W₂`.

No barycenter attainment is assumed.  The proof combines the Wasserstein triangle inequality,
finite weighted Minkowski, and the order characterization of the Fréchet infimum.
-/
theorem wassersteinBarycenterDispersion_stability (q : ProbabilityWeight A)
    (P Q : A → WassersteinMeasure H) :
    |wassersteinBarycenterDispersion q P - wassersteinBarycenterDispersion q Q| ≤
      √(∑ a, q a * dist (P a) (Q a) ^ 2) := by
  rw [abs_sub_le_iff]
  constructor
  · linarith [wassersteinBarycenterDispersion_le_add q P Q]
  · have h := wassersteinBarycenterDispersion_le_add q Q P
    have hδ : √(∑ a, q a * dist (Q a) (P a) ^ 2) =
        √(∑ a, q a * dist (P a) (Q a) ^ 2) := by
      congr 1
      exact Finset.sum_congr rfl fun a _ ↦ by rw [dist_comm]
    rw [hδ] at h
    linarith

/-- A supplied Wasserstein barycenter minimizing the weighted Fréchet objective.

Existence is deliberately not asserted.  The certificate records one candidate, its value,
and global minimality, in the same style as `OptimalJointDispersion`.
-/
structure OptimalWassersteinBarycenter (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) where
  /-- The supplied barycenter law. -/
  center : WassersteinMeasure H
  /-- The certified real objective value. -/
  value : ℝ
  /-- The supplied center attains the certified value. -/
  center_cost : weightedFrechetCost q P center = value
  /-- No candidate Wasserstein law has lower weighted Fréchet cost. -/
  minimal : ∀ Q, value ≤ weightedFrechetCost q P Q

omit [InnerProductSpace ℝ H] in
/-- A certified Wasserstein barycenter value equals the Fréchet infimum definition. -/
theorem OptimalWassersteinBarycenter.value_eq_wassersteinBarycenterDispersionSq
    {q : ProbabilityWeight A} {P : A → WassersteinMeasure H}
    (cert : OptimalWassersteinBarycenter q P) :
    cert.value = wassersteinBarycenterDispersionSq q P := by
  apply le_antisymm
  · unfold wassersteinBarycenterDispersionSq
    refine le_csInf ⟨cert.value, ⟨cert.center, cert.center_cost⟩⟩ ?_
    rintro c ⟨Q, rfl⟩
    exact cert.minimal Q
  · unfold wassersteinBarycenterDispersionSq
    have hbdd : BddBelow (Set.range (weightedFrechetCost q P)) :=
      ⟨0, fun c ⟨Q, hQ⟩ ↦ hQ ▸ weightedFrechetCost_nonneg q P Q⟩
    exact csInf_le hbdd ⟨cert.center, cert.center_cost⟩

/-- The pointwise weighted mean used by the Hilbert barycenter construction. -/
def weightedHilbertMean (q : ProbabilityWeight A) (z : A → H) : H :=
  ∑ a, q a • z a

/-- A certificate for the two constructions behind the Hilbert MMOT--barycenter bridge.

`jointCenter` is the weighted-mean pushforward of any coherent joint law.  `centerJoint`
records a finite gluing around any candidate center.  Their cost fields state the two
polarization inequalities.  Supplying these constructions is weaker than assuming either
optimization problem has a minimizer and keeps the missing finite-gluing theorem explicit.
-/
structure HilbertBarycenterBridge (q : ProbabilityWeight A)
    (P : A → WassersteinMeasure H) where
  /-- The weighted-mean law induced by a coherent joint coupling. -/
  jointCenter : JointCoupling P → WassersteinMeasure H
  /-- The induced center is exactly the weighted-mean pushforward. -/
  jointCenter_measure : ∀ J,
    (jointCenter J).measure = J.measure.map (weightedHilbertMean q)
  /-- The induced center's Fréchet cost does not exceed its source joint cost. -/
  joint_to_center_cost : ∀ J,
    ENNReal.ofReal (weightedFrechetCost q P (jointCenter J)) ≤
      weightedDispersionCost q J
  /-- A coherent marginal coupling obtained by finite gluing around a candidate center. -/
  centerJoint : WassersteinMeasure H → JointCoupling P
  /-- The glued joint cost does not exceed the candidate center's Fréchet cost. -/
  center_to_joint_cost : ∀ Q,
    weightedDispersionCost q (centerJoint Q) ≤
      ENNReal.ofReal (weightedFrechetCost q P Q)

/-- A Hilbert barycenter bridge certifies that multi-marginal dispersion is finite. -/
lemma wassersteinDispersionSq_ne_top_of_barycenterBridge
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (bridge : HilbertBarycenterBridge q P) :
    wassersteinDispersionSq q P ≠ ⊤ := by
  let a := Classical.choice q.index_nonempty
  have hD : wassersteinDispersionSq q P ≤
      ENNReal.ofReal (weightedFrechetCost q P (P a)) := by
    calc
      wassersteinDispersionSq q P ≤
          weightedDispersionCost q (bridge.centerJoint (P a)) := by
        unfold wassersteinDispersionSq
        exact sInf_le ⟨bridge.centerJoint (P a), rfl⟩
      _ ≤ ENNReal.ofReal (weightedFrechetCost q P (P a)) :=
        bridge.center_to_joint_cost (P a)
  exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top hD

/-- Certificate-first Hilbert MMOT--barycenter equivalence.

Given only the two bridge constructions, the infimal weighted Fréchet cost equals the real
form of multi-marginal quadratic dispersion.  No optimal joint plan or barycenter is assumed.
-/
theorem wassersteinBarycenterDispersionSq_eq_toReal_wassersteinDispersionSq
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (bridge : HilbertBarycenterBridge q P) :
    wassersteinBarycenterDispersionSq q P =
      (wassersteinDispersionSq q P).toReal := by
  have hDtop := wassersteinDispersionSq_ne_top_of_barycenterBridge q P bridge
  apply le_antisymm
  · have hB : ENNReal.ofReal (wassersteinBarycenterDispersionSq q P) ≤
        wassersteinDispersionSq q P := by
      unfold wassersteinDispersionSq
      refine le_sInf ?_
      rintro c ⟨J, rfl⟩
      unfold wassersteinBarycenterDispersionSq
      have hbdd : BddBelow (Set.range (weightedFrechetCost q P)) :=
        ⟨0, fun c ⟨R, hR⟩ ↦ hR ▸ weightedFrechetCost_nonneg q P R⟩
      have hcenter := csInf_le hbdd ⟨bridge.jointCenter J, rfl⟩
      exact (ENNReal.ofReal_le_ofReal hcenter).trans (bridge.joint_to_center_cost J)
    exact (ENNReal.ofReal_le_iff_le_toReal hDtop).mp hB
  · unfold wassersteinBarycenterDispersionSq
    have hrange : (Set.range (weightedFrechetCost q P)).Nonempty := by
      let a := Classical.choice q.index_nonempty
      exact ⟨weightedFrechetCost q P (P a), ⟨P a, rfl⟩⟩
    refine le_csInf hrange ?_
    rintro c ⟨Q, rfl⟩
    have hD : wassersteinDispersionSq q P ≤
        weightedDispersionCost q (bridge.centerJoint Q) := by
      unfold wassersteinDispersionSq
      exact sInf_le ⟨bridge.centerJoint Q, rfl⟩
    exact ENNReal.toReal_le_of_le_ofReal (weightedFrechetCost_nonneg q P Q)
      (hD.trans (bridge.center_to_joint_cost Q))

/-- Extended-real form of the certificate-first Hilbert MMOT--barycenter equivalence. -/
theorem ofReal_wassersteinBarycenterDispersionSq_eq_wassersteinDispersionSq
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (bridge : HilbertBarycenterBridge q P) :
    ENNReal.ofReal (wassersteinBarycenterDispersionSq q P) =
      wassersteinDispersionSq q P := by
  rw [wassersteinBarycenterDispersionSq_eq_toReal_wassersteinDispersionSq q P bridge]
  exact ENNReal.ofReal_toReal
    (wassersteinDispersionSq_ne_top_of_barycenterBridge q P bridge)

/-- The square-root Fréchet dispersion equals square-root MMOT dispersion under the bridge. -/
theorem wassersteinBarycenterDispersion_eq_wassersteinDispersion
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (bridge : HilbertBarycenterBridge q P) :
    wassersteinBarycenterDispersion q P = wassersteinDispersion q P := by
  unfold wassersteinBarycenterDispersion wassersteinDispersion
  rw [wassersteinBarycenterDispersionSq_eq_toReal_wassersteinDispersionSq q P bridge,
    Real.sqrt_eq_rpow]

/-- Multi-marginal square-root dispersion inherits barycenter 1-Lipschitz stability.

The bridge certificates replace a global finite-gluing theorem; neither tuple is assumed to
have an optimal multi-marginal coupling or an attained barycenter.
-/
theorem wassersteinDispersion_stability
    (q : ProbabilityWeight A) (P Q : A → WassersteinMeasure H)
    (bridgeP : HilbertBarycenterBridge q P) (bridgeQ : HilbertBarycenterBridge q Q) :
    |wassersteinDispersion q P - wassersteinDispersion q Q| ≤
      √(∑ a, q a * WassersteinDistance (P a) (Q a) ^ 2) := by
  rw [← wassersteinBarycenterDispersion_eq_wassersteinDispersion q P bridgeP,
    ← wassersteinBarycenterDispersion_eq_wassersteinDispersion q Q bridgeQ]
  exact wassersteinBarycenterDispersion_stability q P Q

end WassersteinGeometry.Multimarginal
