import Mathlib.MeasureTheory.MeasurableSpace.Embedding
import Mathlib.Probability.Kernel.Composition.Lemmas
import Mathlib.Probability.Kernel.Disintegration.StandardBorel

open MeasureTheory

open scoped ProbabilityTheory

namespace ProbabilityTheory.Kernel

variable {A X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]

private theorem exists_fin_product {n : ℕ} (kappa : Fin n → Kernel Y X)
    [∀ i, IsMarkovKernel (kappa i)] :
    ∃ K : Kernel Y (Fin n → X), IsMarkovKernel K ∧
      ∀ i, K.map (fun x ↦ x i) = kappa i := by
  induction n with
  | zero =>
      let K : Kernel Y (Fin 0 → X) :=
        Kernel.deterministic (fun _ i ↦ Fin.elim0 i) measurable_const
      refine ⟨K, inferInstance, fun i ↦ Fin.elim0 i⟩
  | succ n ih =>
      let kappaTail : Fin n → Kernel Y X := fun i ↦ kappa i.succ
      letI : ∀ i, IsMarkovKernel (kappaTail i) := fun i ↦ inferInstance
      obtain ⟨Ktail, hKtail, hKtail_marginal⟩ := ih kappaTail
      letI : IsMarkovKernel Ktail := hKtail
      let pairKernel : Kernel Y (X × (Fin n → X)) := kappa 0 ×ₖ Ktail
      let assemble : X × (Fin n → X) → Fin (n + 1) → X :=
        fun p ↦ Fin.cons p.1 p.2
      have hassemble : Measurable assemble := by
        rw [measurable_pi_iff]
        intro i
        refine Fin.cases measurable_fst (fun j ↦ ?_) i
        exact (measurable_pi_apply j).comp measurable_snd
      let K : Kernel Y (Fin (n + 1) → X) := pairKernel.map assemble
      have hK : IsMarkovKernel K := IsMarkovKernel.map pairKernel hassemble
      refine ⟨K, hK, ?_⟩
      intro i
      refine Fin.cases ?_ (fun j ↦ ?_) i
      · have hcomp : (fun f ↦ f 0) ∘ assemble = Prod.fst := by
          funext p
          rfl
        change (pairKernel.map assemble).map (fun x ↦ x 0) = kappa 0
        rw [← Kernel.map_comp_right pairKernel hassemble (measurable_pi_apply 0), hcomp,
          ← Kernel.fst_eq, Kernel.fst_prod]
      · have hcomp : (fun f ↦ f j.succ) ∘ assemble =
            (fun x ↦ x j) ∘ Prod.snd := by
          funext p
          rfl
        change (pairKernel.map assemble).map (fun x ↦ x j.succ) = kappa j.succ
        rw [← Kernel.map_comp_right pairKernel hassemble (measurable_pi_apply j.succ),
          hcomp, Kernel.map_comp_right pairKernel measurable_snd (measurable_pi_apply j),
          ← Kernel.snd_eq, Kernel.snd_prod, hKtail_marginal]

/-- A finite family of Markov kernels with a common source admits a product Markov kernel.

The output coordinates are conditionally independent given the source, and mapping the product
kernel through any coordinate evaluation recovers the corresponding input kernel. The index type
may be any finite type, including the empty type. This is the finite kernel-product construction
used in Agueh--Carlier (2011), Proposition 4.2; the only assumptions are finiteness of the index
type and the Markov property of every input kernel. -/
theorem exists_fintype_product [Fintype A] (kappa : A → Kernel Y X)
    [∀ a, IsMarkovKernel (kappa a)] :
    ∃ K : Kernel Y (A → X), IsMarkovKernel K ∧
      ∀ a, K.map (fun x ↦ x a) = kappa a := by
  classical
  let e : A ≃ Fin (Fintype.card A) := Fintype.equivFin A
  let kappaFin : Fin (Fintype.card A) → Kernel Y X := fun i ↦ kappa (e.symm i)
  letI : ∀ i, IsMarkovKernel (kappaFin i) := fun i ↦ inferInstance
  obtain ⟨Kfin, hKfin, hKfin_marginal⟩ := exists_fin_product kappaFin
  letI : IsMarkovKernel Kfin := hKfin
  let reindex : (Fin (Fintype.card A) → X) → A → X := fun x a ↦ x (e a)
  have hreindex : Measurable reindex := by
    rw [measurable_pi_iff]
    exact fun a ↦ measurable_pi_apply (e a)
  let K : Kernel Y (A → X) := Kfin.map reindex
  have hK : IsMarkovKernel K := IsMarkovKernel.map Kfin hreindex
  refine ⟨K, hK, ?_⟩
  intro a
  have hcomp : (fun f ↦ f a) ∘ reindex = fun x ↦ x (e a) := rfl
  change (Kfin.map reindex).map (fun x ↦ x a) = kappa a
  rw [← Kernel.map_comp_right Kfin hreindex (measurable_pi_apply a), hcomp,
    hKfin_marginal]
  simp [kappaFin, e]

end ProbabilityTheory.Kernel

namespace MeasureTheory.Measure

variable {A X Y : Type*} [Fintype A] [MeasurableSpace X] [MeasurableSpace Y]
  [Nonempty X] [StandardBorelSpace X]

/-- A finite family of probability measures with a common first marginal admits a gluing.

More precisely, if every `pi a` is a law on `Y × X` whose first marginal is `nu`, then there
is a probability law on `Y × (A → X)` whose `(Y, a)` marginal is `pi a` for every `a`.
This is the finite conditional-product gluing construction from Agueh--Carlier (2011),
Proposition 4.2, and also covers an empty index type. The standard-Borel assumption on `X`
supplies the conditional kernels used to disintegrate the pair laws. -/
theorem exists_common_fst_fintype_gluing (nu : Measure Y) [IsProbabilityMeasure nu]
    (pi : A → Measure (Y × X)) (hpi : ∀ a, (pi a).fst = nu) :
    ∃ eta : Measure (Y × (A → X)), IsProbabilityMeasure eta ∧
      ∀ a, eta.map (fun p ↦ (p.1, p.2 a)) = pi a := by
  letI : ∀ a, IsProbabilityMeasure (pi a) := fun a ↦
    ⟨by rw [← Measure.fst_univ, hpi a]; exact measure_univ⟩
  let kappa : A → ProbabilityTheory.Kernel Y X := fun a ↦ (pi a).condKernel
  letI : ∀ a, ProbabilityTheory.IsMarkovKernel (kappa a) := fun a ↦ inferInstance
  obtain ⟨K, hK, hK_marginal⟩ := ProbabilityTheory.Kernel.exists_fintype_product kappa
  letI : ProbabilityTheory.IsMarkovKernel K := hK
  let eta : Measure (Y × (A → X)) := nu ⊗ₘ K
  have heta : IsProbabilityMeasure eta := by
    dsimp [eta]
    infer_instance
  refine ⟨eta, heta, ?_⟩
  intro a
  calc
    eta.map (fun p ↦ (p.1, p.2 a))
        = nu ⊗ₘ (K.map (fun x ↦ x a)) := by
            change (nu ⊗ₘ K).map (Prod.map id (fun x ↦ x a)) =
              nu ⊗ₘ (K.map (fun x ↦ x a))
            exact (Measure.compProd_map (μ := nu) (κ := K) (measurable_pi_apply a)).symm
    _ = nu ⊗ₘ (pi a).condKernel := by rw [hK_marginal a]
    _ = (pi a).fst ⊗ₘ (pi a).condKernel := by rw [hpi a]
    _ = pi a := Measure.disintegrate (pi a) (pi a).condKernel

end MeasureTheory.Measure
