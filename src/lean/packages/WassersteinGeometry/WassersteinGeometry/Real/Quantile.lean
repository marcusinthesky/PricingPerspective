import Mathlib.MeasureTheory.Constructions.UnitInterval
import Mathlib.Probability.CDF
import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import WassersteinGeometry.Real.Defs

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory Set Filter

namespace WassersteinGeometry

namespace P2Real

/-- The unit Lebesgue probability measure on the open interval `(0, 1)`.

The open interval is deliberate: an unbounded law has no finite real-valued quantile at an
endpoint.  Values of a representation outside this interval are immaterial for its pushforward.
-/
noncomputable def uniform01 : Measure ℝ :=
  volume.restrict (Set.Ioo (0 : ℝ) 1)

instance : IsProbabilityMeasure uniform01 where
  measure_univ := by simp [uniform01]

/-- An increasing square-integrable uniform representation of a real `P₂` law.

Downstream transport arguments depend on this representation contract, not on a particular
formula for a generalized inverse.  The representation may be modified on the complement of
`(0, 1)` without changing its pushforward law.
-/
structure QuantileRep (μ : P2Real) where
  value : ℝ → ℝ
  monotoneOn : MonotoneOn value (Set.Ioo (0 : ℝ) 1)
  measurable : Measurable value
  map_eq : Measure.map value uniform01 = μ.measure
  memLp : MemLp value 2 uniform01

/-- The generalized inverse of the CDF, with arbitrary values off the open unit interval.

The endpoint convention is not used by `QuantileRep`; it only supports the separate canonical
quantile layer.

Reference: Panaretos & Zemel (2020), Chapter 2 §2.1. -/
noncomputable def quantile (μ : P2Real) (u : ℝ) : ℝ :=
  if u ≤ 0 ∨ 1 ≤ u then 0 else
    sInf {x : ℝ | u ≤ ProbabilityTheory.cdf μ.measure x}

/-- The generalized inverse is characterized by a CDF threshold on the open unit interval.

This is the keystone interface for the canonical inverse layer.  Endpoint values are excluded
because an unbounded law need not have finite real quantiles at `0` or `1`.
-/
lemma quantile_le_iff_cdf {μ : P2Real} {u x : ℝ} (hu : u ∈ Set.Ioo (0 : ℝ) 1) :
    quantile μ u ≤ x ↔ u ≤ ProbabilityTheory.cdf μ.measure x := by
  let S : Set ℝ := {y | u ≤ ProbabilityTheory.cdf μ.measure y}
  have hS : S.Nonempty := by
    have he : ∀ᶠ y in atTop, u < ProbabilityTheory.cdf μ.measure y :=
      (tendsto_order.1 (ProbabilityTheory.tendsto_cdf_atTop μ.measure)).1 u hu.2
    rcases eventually_atTop.1 he with ⟨y, hy⟩
    exact ⟨y, le_of_lt (hy y le_rfl)⟩
  have hB : BddBelow S := by
    have he : ∀ᶠ y in atBot, ProbabilityTheory.cdf μ.measure y < u :=
      (tendsto_order.1 (ProbabilityTheory.tendsto_cdf_atBot μ.measure)).2 u hu.1
    rcases eventually_atBot.1 he with ⟨y, hy⟩
    refine ⟨y, ?_⟩
    intro z hz
    by_contra hzy
    have hzy' : z < y := lt_of_not_ge hzy
    exact (not_le_of_gt (hy z (le_of_lt hzy')) hz)
  have hq : u ≤ ProbabilityTheory.cdf μ.measure (sInf S) := by
    rw [← (ProbabilityTheory.cdf μ.measure).iInf_Ioi_eq (sInf S)]
    refine le_ciInf ?_
    intro y
    rcases (csInf_lt_iff hB hS).1 y.property with ⟨z, hzS, hzy⟩
    exact hzS.trans (ProbabilityTheory.monotone_cdf μ.measure hzy.le)
  have hq_le : sInf S ≤ x ↔ u ≤ ProbabilityTheory.cdf μ.measure x := by
    constructor
    · intro hqx
      exact hq.trans (ProbabilityTheory.monotone_cdf μ.measure hqx)
    · intro hux
      exact csInf_le hB hux
  simpa [quantile, S, not_le_of_gt hu.1, not_le_of_gt hu.2] using hq_le

/-- The CDF threshold description of every quantile sublevel set. -/
lemma quantile_preimage_Iic (μ : P2Real) (x : ℝ) :
    quantile μ ⁻¹' Iic x =
      if 0 ≤ x then
        (Ioo (0 : ℝ) 1 ∩ Iic (ProbabilityTheory.cdf μ.measure x)) ∪
          (Ioo (0 : ℝ) 1)ᶜ
      else Ioo (0 : ℝ) 1 ∩ Iic (ProbabilityTheory.cdf μ.measure x) := by
  by_cases hx : 0 ≤ x
  · simp only [if_pos hx]
    ext u
    by_cases hu : u ∈ Ioo (0 : ℝ) 1
    · change quantile μ u ≤ x ↔ _
      rw [quantile_le_iff_cdf hu]
      simp [hu]
    · have hu' : u ≤ 0 ∨ 1 ≤ u := by
        rcases le_or_gt u 0 with hu0 | hu0
        · exact Or.inl hu0
        · exact Or.inr (le_of_not_gt fun h1 => hu ⟨hu0, h1⟩)
      change quantile μ u ≤ x ↔ _
      simp [quantile, hu', hx, hu]
  · simp only [if_neg hx]
    ext u
    by_cases hu : u ∈ Ioo (0 : ℝ) 1
    · change quantile μ u ≤ x ↔ _
      rw [quantile_le_iff_cdf hu]
      simp [hu]
    · have hu' : u ≤ 0 ∨ 1 ≤ u := by
        rcases le_or_gt u 0 with hu0 | hu0
        · exact Or.inl hu0
        · exact Or.inr (le_of_not_gt fun h1 => hu ⟨hu0, h1⟩)
      change quantile μ u ≤ x ↔ _
      simp [quantile, hu', hx, hu]

/-- The canonical generalized inverse is measurable from its CDF sublevel sets. -/
lemma quantile_measurable (μ : P2Real) : Measurable (quantile μ) := by
  apply measurable_of_Iic
  intro x
  rw [quantile_preimage_Iic]
  split_ifs with hx
  · exact (measurableSet_Ioo.inter measurableSet_Iic).union measurableSet_Ioo.compl
  · exact measurableSet_Ioo.inter measurableSet_Iic

private lemma uniform01_Iic {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    uniform01 (Ioo (0 : ℝ) 1 ∩ Iic a) = ENNReal.ofReal a := by
  by_cases ha : a = 0
  · subst a
    have hset : Ioo (0 : ℝ) 1 ∩ Iic 0 = ∅ := by
      ext u
      simp only [mem_inter_iff, mem_Ioo, mem_Iic, mem_empty_iff_false, iff_false]
      intro hu
      exact (not_lt_of_ge hu.2) hu.1.1
    rw [hset]
    simp
  by_cases ha_one : a = 1
  · subst a
    have hset : Ioo (0 : ℝ) 1 ∩ Iic 1 = Ioo (0 : ℝ) 1 := by
      ext u
      simp only [mem_inter_iff, mem_Ioo, mem_Iic]
      constructor
      · exact fun hu => hu.1
      · exact fun hu => ⟨hu, le_of_lt hu.2⟩
    rw [hset, uniform01, Measure.restrict_apply measurableSet_Ioo]
    simp
  have ha_pos : 0 < a := lt_of_le_of_ne ha0 (Ne.symm ha)
  have ha_lt_one : a < 1 := lt_of_le_of_ne ha1 ha_one
  have hset : Ioo (0 : ℝ) 1 ∩ Iic a = Ioc 0 a := by
    ext u
    constructor
    · intro hu
      exact ⟨hu.1.1, hu.2⟩
    · intro hu
      exact ⟨⟨hu.1, lt_of_le_of_lt hu.2 ha_lt_one⟩, hu.2⟩
  rw [hset, uniform01, Measure.restrict_apply measurableSet_Ioc]
  have hset' : Ioc (0 : ℝ) a ∩ Ioo 0 1 = Ioc 0 a := by
    ext u
    constructor
    · intro hu
      exact hu.1
    · intro hu
      exact ⟨hu, ⟨hu.1, lt_of_le_of_lt hu.2 ha_lt_one⟩⟩
  rw [hset', Real.volume_Ioc]
  simp

/-- The canonical generalized inverse pushes uniform measure forward to its law. -/
lemma quantile_pushforward (μ : P2Real) :
    Measure.map (quantile μ) uniform01 = μ.measure := by
  letI : IsProbabilityMeasure (Measure.map (quantile μ) uniform01) :=
    Measure.isProbabilityMeasure_map (quantile_measurable μ).aemeasurable
  apply Measure.eq_of_cdf
  apply StieltjesFunction.ext
  intro x
  rw [ProbabilityTheory.cdf_eq_real, measureReal_def,
    Measure.map_apply (quantile_measurable μ) measurableSet_Iic,
    ProbabilityTheory.cdf_eq_real]
  rw [quantile_preimage_Iic]
  split_ifs with hx
  · have hF0 : 0 ≤ ProbabilityTheory.cdf μ.measure x :=
      ProbabilityTheory.cdf_nonneg μ.measure x
    have hF1 : ProbabilityTheory.cdf μ.measure x ≤ 1 :=
      ProbabilityTheory.cdf_le_one μ.measure x
    have hrestrict : uniform01 ((Ioo (0 : ℝ) 1 ∩
        Iic (ProbabilityTheory.cdf μ.measure x)) ∪ (Ioo (0 : ℝ) 1)ᶜ) =
        uniform01 (Ioo (0 : ℝ) 1 ∩ Iic (ProbabilityTheory.cdf μ.measure x)) := by
      rw [uniform01, Measure.restrict_apply
        ((measurableSet_Ioo.inter measurableSet_Iic).union measurableSet_Ioo.compl)]
      have hset :
          ((Ioo (0 : ℝ) 1 ∩ Iic (ProbabilityTheory.cdf μ.measure x)) ∪
            (Ioo (0 : ℝ) 1)ᶜ) ∩ Ioo (0 : ℝ) 1 =
            Ioo (0 : ℝ) 1 ∩ Iic (ProbabilityTheory.cdf μ.measure x) := by
        ext u
        by_cases hu : u ∈ Ioo (0 : ℝ) 1 <;> simp [hu]
      rw [hset]
      rw [Measure.restrict_apply (measurableSet_Ioo.inter measurableSet_Iic)]
      congr 1
      ext u
      aesop
    rw [hrestrict, uniform01_Iic hF0 hF1, ENNReal.toReal_ofReal hF0,
      ProbabilityTheory.cdf_eq_real]
  · have hF0 : 0 ≤ ProbabilityTheory.cdf μ.measure x :=
      ProbabilityTheory.cdf_nonneg μ.measure x
    have hF1 : ProbabilityTheory.cdf μ.measure x ≤ 1 :=
      ProbabilityTheory.cdf_le_one μ.measure x
    rw [uniform01_Iic hF0 hF1, ENNReal.toReal_ofReal hF0,
      ProbabilityTheory.cdf_eq_real]

/-- The canonical inverse is increasing on the open unit interval. -/
lemma quantile_monotoneOn (μ : P2Real) :
    MonotoneOn (quantile μ) (Ioo (0 : ℝ) 1) := by
  intro u hu v hv huv
  apply (quantile_le_iff_cdf hu).2
  exact huv.trans ((quantile_le_iff_cdf hv).1 le_rfl)

/-- The canonical inverse inherits square integrability from its pushforward law. -/
lemma quantile_memLp (μ : P2Real) : MemLp (quantile μ) 2 uniform01 := by
  have hid : MemLp id 2 (Measure.map (quantile μ) uniform01) := by
    simpa [quantile_pushforward] using μ.id_memLp
  have h := (memLp_map_measure_iff (μ := uniform01) (f := quantile μ) (g := id)
    (p := (2 : ℝ≥0∞)) measurable_id.aestronglyMeasurable
    (quantile_measurable μ).aemeasurable)
  simpa [Function.comp_def] using h.1 hid

/-- A canonical generalized-inverse certificate extending an increasing uniform representation.

The inverse inequalities are retained here because they characterize this particular `sInf`
construction.  The transport, geodesic, and barycenter layers use only `QuantileRep`.
-/
structure CanonicalQuantileSpec (μ : P2Real) extends QuantileRep μ where
  value_eq_quantile : value = quantile μ
  lower_inverse : ∀ᵐ u ∂uniform01,
    ∀ x : ℝ, x < value u → ProbabilityTheory.cdf μ.measure x < u
  upper_inverse : ∀ᵐ u ∂uniform01,
    ∀ x : ℝ, value u < x → u ≤ ProbabilityTheory.cdf μ.measure x
/-- The CDF threshold theorem supplies a complete canonical quantile certificate. -/
noncomputable def canonicalQuantileSpec (μ : P2Real) : CanonicalQuantileSpec μ :=
  { value := quantile μ
    monotoneOn := quantile_monotoneOn μ
    measurable := quantile_measurable μ
    map_eq := quantile_pushforward μ
    memLp := quantile_memLp μ
    value_eq_quantile := rfl
    lower_inverse := by
      filter_upwards [ae_restrict_mem measurableSet_Ioo] with u hu
      intro x hx
      by_contra hnot
      have hux : u ≤ ProbabilityTheory.cdf μ.measure x := le_of_not_gt hnot
      exact (not_lt_of_ge ((quantile_le_iff_cdf hu).2 hux)) hx
    upper_inverse := by
      filter_upwards [ae_restrict_mem measurableSet_Ioo] with u hu
      intro x hxu
      exact (quantile_le_iff_cdf hu).1 hxu.le }


/-- Canonical quantile existence is kept separate from the representation API. -/
def HasCanonicalQuantileSpec (μ : P2Real) : Prop := Nonempty (CanonicalQuantileSpec μ)

lemma hasCanonicalQuantileSpec (μ : P2Real) : HasCanonicalQuantileSpec μ :=
  ⟨canonicalQuantileSpec μ⟩

/-- Select a canonical quantile certificate only after one has been supplied. -/
noncomputable def chooseCanonicalQuantileSpec {μ : P2Real} (h : HasCanonicalQuantileSpec μ) :
    CanonicalQuantileSpec μ :=
  Classical.choice h

/-- The monotone coupling generated by two increasing uniform representations. -/
noncomputable def quantileCoupling {μ ν : P2Real}
    (qμ : QuantileRep μ) (qν : QuantileRep ν) : Measure (ℝ × ℝ) :=
  Measure.map (fun u : ℝ => (qμ.value u, qν.value u)) uniform01

lemma quantileCoupling_fst {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    (quantileCoupling qμ qν).map Prod.fst = μ.measure := by
  rw [quantileCoupling, Measure.map_map measurable_fst (qμ.measurable.prodMk qν.measurable)]
  simpa [Function.comp_def] using qμ.map_eq

lemma quantileCoupling_snd {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    (quantileCoupling qμ qν).map Prod.snd = ν.measure := by
  rw [quantileCoupling, Measure.map_map measurable_snd (qμ.measurable.prodMk qν.measurable)]
  simpa [Function.comp_def] using qν.map_eq

lemma quantileCoupling_mem {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    quantileCoupling qμ qν ∈ couplingSet μ.measure ν.measure :=
  ⟨quantileCoupling_fst qμ qν, quantileCoupling_snd qμ qν⟩

/-- The squared cost of a monotone representation coupling on the uniform parameter. -/
noncomputable def quantileCost {μ ν : P2Real}
    (qμ : QuantileRep μ) (qν : QuantileRep ν) : ℝ≥0∞ :=
  ∫⁻ u : ℝ, (edist (qμ.value u) (qν.value u)) ^ 2 ∂uniform01

/-- The quantile coupling cost is an unconditional change-of-variables identity. -/
lemma quantileCostIdentity {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    (∫⁻ p, (edist p.1 p.2) ^ 2 ∂quantileCoupling qμ qν) = quantileCost qμ qν := by
  have hcost : Measurable (fun p : ℝ × ℝ => (edist p.1 p.2) ^ 2) :=
    measurable_edist.pow measurable_const
  rw [quantileCoupling, lintegral_map hcost
    (qμ.measurable.prodMk qν.measurable)]
  rfl

/-- Monotone-rearrangement optimality as an explicit, reusable proposition. -/
def MonotoneRearrangementOptimality {μ ν : P2Real}
    (qμ : QuantileRep μ) (qν : QuantileRep ν) : Prop :=
  ∀ π ∈ couplingSet μ.measure ν.measure,
    quantileCost qμ qν ≤ ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π

/-- The conditional Wasserstein bridge once monotone rearrangement is supplied. -/
lemma wassersteinDistanceSq_eq_quantileCost {μ ν : P2Real} (qμ : QuantileRep μ)
    (qν : QuantileRep ν) (hoptimal : MonotoneRearrangementOptimality qμ qν) :
    WassersteinDistanceSq μ.toGeneric ν.toGeneric = quantileCost qμ qν := by
  apply le_antisymm
  · rw [WassersteinDistanceSq]
    exact sInf_le ⟨quantileCoupling qμ qν, quantileCoupling_mem qμ qν,
      (quantileCostIdentity qμ qν).symm⟩
  · rw [WassersteinDistanceSq]
    refine le_sInf ?_
    rintro c ⟨π, hπ, rfl⟩
    exact hoptimal π hπ

end P2Real

end WassersteinGeometry
