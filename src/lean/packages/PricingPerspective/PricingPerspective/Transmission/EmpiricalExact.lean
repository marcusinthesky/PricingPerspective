import PricingPerspective.Transmission.Empirical

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry WassersteinGeometry.Hilbert Finset

/-!
# The empirical envelope is exact, not merely conservative

`Transmission.Empirical` embeds finite couplings into `Π(P, Q)`. That gives one inequality:
the measure-level infimum is at most the finite minimum, so the measure-level ceiling is at
**least** the finite one.

That direction is the wrong one to rely on. The ceiling is an *upper* bound on attainable
covariance, so a finite computation that understates the population ceiling would report
envelope *violations* that are not violations at all — an artefact of the discretization
rather than evidence against the model. Reading the pipeline's numbers as certifying the
population envelope requires the reverse inclusion, and that is what this file proves.

## The result

`transportCost_eq_finite` and `toOptimalTransportCost`: for finitely supported
loading laws with distinct support points, every measure-level coupling *is* a finite
coupling, so a `FiniteOptimalCostCertificate` upgrades to a full `OptimalTransportCost`.
The two infima coincide, and `ceiling` computed from the pipeline's transport solve is the
population ceiling.

## The route

1. `eq_sum_dirac_of_finite_support` — a measure that is null off a finite set is the sum of
   its atoms there. This is a general fact about measures, stated for its own sake.
2. `coupling_compl_prod_range_null` — a coupling of two finitely supported laws is null off the
   product of the supports, because each marginal is.
3. `eq_ofCoupling` — hence it *equals* `ofCoupling w Z Y` for the weight matrix read off its
   atoms. Not approximated by: equals.
4. The cost identity and the certificate upgrade then follow from `Transmission.Empirical`.

## The injectivity hypothesis

`Z` and `Y` are assumed injective — distinct cloud points. This is what the pipeline has
(distinct article embeddings), and it is what makes the atom weights `π {(Zₐ, Y_b)}` read off
a well-defined matrix. Without it, mass at a coincident point would have to be split back
across the colliding indices, and the splitting is not unique. The envelope is unaffected —
duplicate points can be merged by summing their weights first — but the *statement* would
need that preprocessing, so it is cleaner to assume distinctness than to hide it.
-/

namespace PricingPerspective.Transmission

open RandomExposure

/-! ### A measure with finite support is the sum of its atoms -/

/-- A measure that is null off a finite set is the finite sum of its atoms there. -/
lemma eq_sum_dirac_of_finite_support {X : Type*} [MeasurableSpace X]
    [MeasurableSingletonClass X] {μ : Measure X} (s : Finset X)
    (hs : μ (↑s : Set X)ᶜ = 0) :
    μ = ∑ i ∈ s, (μ {i}) • Measure.dirac i := by
  ext A hA
  rw [Measure.coe_finsetSum, Finset.sum_apply]
  simp only [Measure.smul_apply, Measure.dirac_apply' _ hA, smul_eq_mul]
  have hdiff : μ (A \ (↑s : Set X)) = 0 :=
    measure_mono_null (Set.sdiff_subset_compl _ _) hs
  have hsplit : μ (A ∩ (↑s : Set X)) = μ A := by
    have := measure_inter_add_sdiff (μ := μ) A (s.finite_toSet.measurableSet)
    rw [hdiff, add_zero] at this
    exact this
  have hcover : A ∩ (↑s : Set X) = ⋃ i ∈ s, ({i} ∩ A) := by
    ext x; simp [and_comm]
  have hmeas : ∀ i ∈ s, MeasurableSet ({i} ∩ A) := fun i _ =>
    (measurableSet_singleton i).inter hA
  have hdisj : (↑s : Set X).PairwiseDisjoint fun i => ({i} ∩ A) := by
    intro i _ j _ hij
    exact Set.disjoint_left.2 fun x hx hx' => hij (by
      simp only [Set.mem_inter_iff, Set.mem_singleton_iff] at hx hx'
      rw [← hx.1, ← hx'.1])
  rw [← hsplit, hcover, measure_biUnion_finset hdisj hmeas]
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h : i ∈ A
  · rw [Set.inter_eq_self_of_subset_left (Set.singleton_subset_iff.2 h)]
    simp [Set.indicator_of_mem h]
  · have hempty : ({i} : Set X) ∩ A = ∅ := by
      ext y
      simp only [Set.mem_inter_iff, Set.mem_singleton_iff, Set.mem_empty_iff_false,
        iff_false, not_and]
      rintro rfl; exact h
    rw [hempty]
    simp [Set.indicator_of_notMem h]

variable {ι ι' : Type*} [Fintype ι] [Fintype ι']
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]
variable {p : ι → ℝ} {q : ι' → ℝ}

/-! ### Cloud laws are null off their support -/

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- A cloud law charges only its own support points. -/
lemma ofCloud_compl_range_null (p : ι → ℝ) (Z : ι → H) :
    ofCloud p Z (Set.range Z)ᶜ = 0 := by
  unfold ofCloud
  rw [Measure.coe_finsetSum, Finset.sum_apply]
  refine Finset.sum_eq_zero fun a _ => ?_
  rw [Measure.smul_apply,
    Measure.dirac_apply' _ (Set.finite_range Z).measurableSet.compl]
  simp [Set.indicator_of_notMem, Set.mem_range_self a]

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- The atom of a cloud law at a support point is its weight, when points are distinct. -/
lemma ofCloud_singleton {Z : ι → H} (hZ : Function.Injective Z) (a : ι) :
    ofCloud p Z {Z a} = ENNReal.ofReal (p a) := by
  unfold ofCloud
  rw [Measure.coe_finsetSum, Finset.sum_apply]
  rw [Finset.sum_eq_single a]
  · simp
  · intro c _ hc
    have hne : Z c ∉ ({Z a} : Set H) := by
      simp only [Set.mem_singleton_iff]
      exact fun h => hc (hZ h)
    rw [Measure.smul_apply, Measure.dirac_apply' _ (measurableSet_singleton _),
      Set.indicator_of_notMem hne, smul_zero]
  · simp

/-! ### A coupling of cloud laws lives on the product of the supports -/

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- A coupling of two cloud laws is null off the product of their supports. -/
lemma coupling_compl_prod_range_null {Z : ι → H} {Y : ι' → H}
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y)) :
    π ((Set.range Z ×ˢ Set.range Y)ᶜ) = 0 := by
  have h1 : π (Prod.fst ⁻¹' (Set.range Z)ᶜ) = 0 := by
    rw [← Measure.map_apply measurable_fst (Set.finite_range Z).measurableSet.compl, hπ.1]
    exact ofCloud_compl_range_null p Z
  have h2 : π (Prod.snd ⁻¹' (Set.range Y)ᶜ) = 0 := by
    rw [← Measure.map_apply measurable_snd (Set.finite_range Y).measurableSet.compl, hπ.2]
    exact ofCloud_compl_range_null q Y
  refine measure_mono_null ?_ (measure_union_null h1 h2)
  intro z hz
  by_contra h
  simp only [Set.mem_union, Set.mem_preimage, Set.mem_compl_iff, not_or, not_not] at h
  exact hz ⟨h.1, h.2⟩

section Support

variable [DecidableEq H]

/-- The finite support carried by two clouds. -/
def supportFinset (Z : ι → H) (Y : ι' → H) : Finset (H × H) :=
  Finset.univ.image (fun ab : ι × ι' => (Z ab.1, Y ab.2))

omit [NormedAddCommGroup H] [InnerProductSpace ℝ H] [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
lemma coe_supportFinset (Z : ι → H) (Y : ι' → H) :
    (↑(supportFinset Z Y) : Set (H × H)) = Set.range Z ×ˢ Set.range Y := by
  ext ⟨z₁, z₂⟩
  simp only [supportFinset, Finset.coe_image, Finset.coe_univ, Set.image_univ,
    Set.mem_range, Set.mem_prod, Prod.exists, Prod.mk.injEq]
  constructor
  · rintro ⟨a, b, h₁, h₂⟩; exact ⟨⟨a, h₁⟩, ⟨b, h₂⟩⟩
  · rintro ⟨⟨a, h₁⟩, ⟨b, h₂⟩⟩; exact ⟨a, b, h₁, h₂⟩

end Support

/-! ### The weight matrix read off a coupling's atoms -/

/-- The finite weight matrix carried by a coupling of two cloud laws. -/
noncomputable def toFiniteWeights (π : Measure (H × H)) (Z : ι → H) (Y : ι' → H) :
    ι → ι' → ℝ :=
  fun a b => (π {(Z a, Y b)}).toReal

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- The atoms along one row exhaust that row's marginal mass. -/
lemma sum_atoms_row {Z : ι → H} {Y : ι' → H} (hY : Function.Injective Y)
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y)) (a : ι) :
    ∑ b, π {(Z a, Y b)} = ofCloud p Z {Z a} := by
  classical
  -- The row's atoms are disjoint and cover `{Z a} ×ˢ range Y`.
  have hdisj : (↑(Finset.univ : Finset ι') : Set ι').PairwiseDisjoint
      fun b => ({(Z a, Y b)} : Set (H × H)) := by
    intro b _ b' _ hbb'
    refine Set.disjoint_left.2 fun z hz hz' => hbb' ?_
    simp only [Set.mem_singleton_iff] at hz hz'
    exact hY (congrArg Prod.snd (hz.symm.trans hz'))
  have hmeas : ∀ b ∈ (Finset.univ : Finset ι'),
      MeasurableSet ({(Z a, Y b)} : Set (H × H)) := fun b _ => measurableSet_singleton _
  have hcover : (⋃ b ∈ (Finset.univ : Finset ι'), ({(Z a, Y b)} : Set (H × H)))
      = ({Z a} : Set H) ×ˢ Set.range Y := by
    apply Set.Subset.antisymm
    · intro z hz
      simp only [Set.mem_iUnion, Finset.mem_univ, Set.mem_singleton_iff, exists_prop,
        true_and] at hz
      obtain ⟨b, rfl⟩ := hz
      exact ⟨rfl, ⟨b, rfl⟩⟩
    · rintro ⟨z₁, z₂⟩ ⟨h₁, ⟨b, hb⟩⟩
      simp only [Set.mem_singleton_iff] at h₁
      subst h₁; subst hb
      exact Set.mem_biUnion (Finset.mem_univ b) rfl
  rw [← measure_biUnion_finset hdisj hmeas, hcover]
  -- That set differs from the full fibre by a null set.
  have hnull : π (Prod.fst ⁻¹' {Z a} \ (({Z a} : Set H) ×ˢ Set.range Y)) = 0 := by
    refine measure_mono_null ?_ (coupling_compl_prod_range_null hπ)
    intro z hz
    simp only [Set.mem_sdiff, Set.mem_preimage, Set.mem_singleton_iff, Set.mem_prod,
      Set.mem_range, not_and] at hz
    simp only [Set.mem_compl_iff, Set.mem_prod, Set.mem_range, not_and]
    exact fun _ => hz.2 hz.1
  have hsub : (({Z a} : Set H) ×ˢ Set.range Y) ⊆ Prod.fst ⁻¹' {Z a} := fun z hz => hz.1
  have hsplit := measure_inter_add_sdiff (μ := π) (Prod.fst ⁻¹' {Z a})
    ((measurableSet_singleton (Z a)).prod (Set.finite_range Y).measurableSet)
  rw [Set.inter_eq_self_of_subset_right hsub, hnull, add_zero] at hsplit
  rw [hsplit, ← Measure.map_apply measurable_fst (measurableSet_singleton _), hπ.1]

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- The atoms along one column exhaust that column's marginal mass. -/
lemma sum_atoms_col {Z : ι → H} {Y : ι' → H} (hZ : Function.Injective Z)
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y)) (b : ι') :
    ∑ a, π {(Z a, Y b)} = ofCloud q Y {Y b} := by
  classical
  have hdisj : (↑(Finset.univ : Finset ι) : Set ι).PairwiseDisjoint
      fun a => ({(Z a, Y b)} : Set (H × H)) := by
    intro a _ a' _ haa'
    refine Set.disjoint_left.2 fun z hz hz' => haa' ?_
    simp only [Set.mem_singleton_iff] at hz hz'
    exact hZ (congrArg Prod.fst (hz.symm.trans hz'))
  have hmeas : ∀ a ∈ (Finset.univ : Finset ι),
      MeasurableSet ({(Z a, Y b)} : Set (H × H)) := fun a _ => measurableSet_singleton _
  have hcover : (⋃ a ∈ (Finset.univ : Finset ι), ({(Z a, Y b)} : Set (H × H)))
      = Set.range Z ×ˢ ({Y b} : Set H) := by
    apply Set.Subset.antisymm
    · intro z hz
      simp only [Set.mem_iUnion, Finset.mem_univ, Set.mem_singleton_iff, exists_prop,
        true_and] at hz
      obtain ⟨a, rfl⟩ := hz
      exact ⟨⟨a, rfl⟩, rfl⟩
    · rintro ⟨z₁, z₂⟩ ⟨⟨a, ha⟩, h₂⟩
      simp only [Set.mem_singleton_iff] at h₂
      subst h₂; subst ha
      exact Set.mem_biUnion (Finset.mem_univ a) rfl
  rw [← measure_biUnion_finset hdisj hmeas, hcover]
  have hnull : π (Prod.snd ⁻¹' {Y b} \ (Set.range Z ×ˢ ({Y b} : Set H))) = 0 := by
    refine measure_mono_null ?_ (coupling_compl_prod_range_null hπ)
    intro z hz
    simp only [Set.mem_sdiff, Set.mem_preimage, Set.mem_singleton_iff, Set.mem_prod,
      Set.mem_range, not_and] at hz
    simp only [Set.mem_compl_iff, Set.mem_prod, Set.mem_range, not_and]
    intro hzr
    exact absurd hz.1 (by simpa using fun h => hz.2 hzr h)
  have hsub : (Set.range Z ×ˢ ({Y b} : Set H)) ⊆ Prod.snd ⁻¹' {Y b} := fun z hz => hz.2
  have hsplit := measure_inter_add_sdiff (μ := π) (Prod.snd ⁻¹' {Y b})
    ((Set.finite_range Z).measurableSet.prod (measurableSet_singleton (Y b)))
  rw [Set.inter_eq_self_of_subset_right hsub, hnull, add_zero] at hsplit
  rw [hsplit, ← Measure.map_apply measurable_snd (measurableSet_singleton _), hπ.2]

/-- **Every coupling of two cloud laws is a finite coupling.**

    The reverse of `Transmission.Empirical.ofCoupling_mem_couplingSet`. Its weights are read
    off the coupling's atoms, which is well defined exactly because the support points are
    distinct. -/
noncomputable def toFiniteCoupling
    (hp : ∀ a, 0 ≤ p a) (hmp : ∑ a, p a = 1) (hq : ∀ b, 0 ≤ q b)
    {Z : ι → H} {Y : ι' → H} (hZ : Function.Injective Z) (hY : Function.Injective Y)
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y)) :
    RandomExposure.Coupling p q where
  w := toFiniteWeights π Z Y
  nonneg := fun _ _ => ENNReal.toReal_nonneg
  marginal_fst := fun a => by
    haveI := isProbabilityMeasure_ofCloud hp hmp Z
    haveI : IsProbabilityMeasure π :=
      coupling_isProbabilityMeasure (measure_univ (μ := ofCloud p Z)) hπ
    show ∑ b, (π {(Z a, Y b)}).toReal = p a
    rw [← ENNReal.toReal_sum fun b _ => measure_ne_top π _, sum_atoms_row hY hπ a,
      ofCloud_singleton hZ a, ENNReal.toReal_ofReal (hp a)]
  marginal_snd := fun b => by
    haveI := isProbabilityMeasure_ofCloud hp hmp Z
    haveI : IsProbabilityMeasure π :=
      coupling_isProbabilityMeasure (measure_univ (μ := ofCloud p Z)) hπ
    show ∑ a, (π {(Z a, Y b)}).toReal = q b
    rw [← ENNReal.toReal_sum fun a _ => measure_ne_top π _, sum_atoms_col hZ hπ b,
      ofCloud_singleton hY b, ENNReal.toReal_ofReal (hq b)]

/-! ### The two infima coincide -/

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- **A coupling of cloud laws equals the joint law of its own weight matrix.**

    Not "is approximated by": equals. This is the reverse inclusion, and it is what makes
    the finite transport problem the *same* problem as the measure-level one. -/
theorem eq_ofCoupling
    (hp : ∀ a, 0 ≤ p a) (hmp : ∑ a, p a = 1)
    {Z : ι → H} {Y : ι' → H} (hZ : Function.Injective Z) (hY : Function.Injective Y)
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y))
    (w : RandomExposure.Coupling p q) (hw : ∀ a b, w.w a b = (π {(Z a, Y b)}).toReal) :
    π = ofCoupling w Z Y := by
  classical
  haveI := isProbabilityMeasure_ofCloud hp hmp Z
  haveI : IsProbabilityMeasure π :=
    coupling_isProbabilityMeasure (measure_univ (μ := ofCloud p Z)) hπ
  have hnull : π (↑(supportFinset Z Y) : Set (H × H))ᶜ = 0 := by
    rw [coe_supportFinset]; exact coupling_compl_prod_range_null hπ
  have hinj : ∀ x ∈ (Finset.univ : Finset (ι × ι')), ∀ y ∈ (Finset.univ : Finset (ι × ι')),
      (Z x.1, Y x.2) = (Z y.1, Y y.2) → x = y := by
    intro x _ y _ h
    exact Prod.ext (hZ (congrArg Prod.fst h)) (hY (congrArg Prod.snd h))
  rw [eq_sum_dirac_of_finite_support _ hnull]
  show _ = ∑ a, ∑ b, _
  rw [supportFinset, Finset.sum_image hinj, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  rw [hw a b, ENNReal.ofReal_toReal (measure_ne_top π _)]

omit [InnerProductSpace ℝ H] [Inhabited H] in
/-- The transport cost of any coupling of cloud laws is a finite transport cost. -/
theorem transportCost_eq_finite
    (hp : ∀ a, 0 ≤ p a) (hmp : ∑ a, p a = 1)
    {Z : ι → H} {Y : ι' → H} (hZ : Function.Injective Z) (hY : Function.Injective Y)
    {π : Measure (H × H)} (hπ : π ∈ couplingSet (ofCloud p Z) (ofCloud q Y))
    (w : RandomExposure.Coupling p q) (hw : ∀ a b, w.w a b = (π {(Z a, Y b)}).toReal) :
    Hilbert.transportCost π = RandomExposure.transportCost w Z Y := by
  rw [eq_ofCoupling hp hmp hZ hY hπ w hw]
  exact transportCost_ofCoupling w (fun a b => (hw a b) ▸ ENNReal.toReal_nonneg) Z Y

/-- **A finite optimality certificate certifies the population optimum.**

    The pipeline's transport solve, which minimizes over `m × m` weight matrices, is a
    minimizer over *all* couplings of the induced laws. So `ceiling` computed from it is the
    population ceiling, not a lower bound on it, and an envelope violation measured against
    it is a real violation rather than a discretization artefact. -/
noncomputable def toOptimalTransportCost
    (hp : ∀ a, 0 ≤ p a) (hmp : ∑ a, p a = 1) (hq : ∀ b, 0 ≤ q b)
    {Z : ι → H} {Y : ι' → H} (hZ : Function.Injective Z) (hY : Function.Injective Y)
    {P Q : WassersteinMeasure H}
    (hP : P.measure = ofCloud p Z) (hQ : Q.measure = ofCloud q Y)
    (cert : RandomExposure.FiniteOptimalCostCertificate
      (fun w : RandomExposure.Coupling p q => RandomExposure.transportCost w Z Y)) :
    OptimalTransportCost P Q where
  plan := toIntegrableCoupling cert.plan cert.plan.nonneg Z Y hP hQ
  value := cert.value
  plan_cost := by
    show Hilbert.transportCost (ofCoupling cert.plan Z Y) = cert.value
    rw [transportCost_ofCoupling cert.plan cert.plan.nonneg Z Y]
    exact cert.plan_cost
  minimal := fun σ => by
    have hmem : σ.toMeasure ∈ couplingSet (ofCloud p Z) (ofCloud q Y) := by
      rw [← hP, ← hQ]; exact σ.mem
    rw [transportCost_eq_finite hp hmp hZ hY hmem
      (toFiniteCoupling hp hmp hq hZ hY hmem) fun _ _ => rfl]
    exact cert.minimal _

end PricingPerspective.Transmission
