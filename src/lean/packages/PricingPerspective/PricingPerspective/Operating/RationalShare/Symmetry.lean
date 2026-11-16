import Architect
import PricingPerspective.Operating.RationalShare

set_option linter.style.longLine false

/-!
# Firm relabeling and the symmetry boundary

The rational-share game permits firm-specific linear and quadratic costs. When those primitives
are instead common across firms, every permutation of firm labels preserves the game. This module
makes that symmetry explicit and combines it with the unique-equilibrium theorem from
`Operating/RationalShare.lean`.

The consequence is a no-go result: a unique equilibrium of the fully firm-symmetric game cannot
differentiate firms. Every firm has the same capacity block and, whenever that common block has
positive mass, the same normalized composition. Differentiated unique equilibria therefore require
firm-varying primitives or a weaker symmetry than arbitrary relabeling.
-/

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### Relabeling -/

/-- Relabel the firm coordinate of a capacity profile by a permutation. The new block at `i` is
the old block at `σ.symm i`; activity labels are unchanged. -/
def relabelFirms (σ : Equiv.Perm ι) (x : Profile ι α) : Profile ι α :=
  WithLp.toLp 2 fun p ↦ x (σ.symm p.1, p.2)

omit [DecidableEq ι] in
/-- Applying a relabeled profile reads the original block at the inverse firm label. -/
@[simp] theorem relabelFirms_apply (σ : Equiv.Perm ι) (x : Profile ι α) (i : ι) (a : α) :
    relabelFirms σ x (i, a) = x (σ.symm i, a) := rfl

omit [DecidableEq ι] in
/-- Relabeling by the inverse permutation recovers the original profile. -/
@[simp] theorem relabelFirms_symm (σ : Equiv.Perm ι) (x : Profile ι α) :
    relabelFirms σ.symm (relabelFirms σ x) = x := by
  ext p
  rcases p with ⟨i, a⟩
  simp

omit [DecidableEq ι] in
/-- Relabeling first by the inverse and then by the permutation recovers the profile. -/
@[simp] theorem relabelFirms_symm' (σ : Equiv.Perm ι) (x : Profile ι α) :
    relabelFirms σ (relabelFirms σ.symm x) = x := by
  simpa using relabelFirms_symm σ.symm x

namespace Nodes

variable (N : Nodes κ α)

omit [DecidableEq ι] in
/-- Relabeling firms relabels their appeal blocks in the same way. -/
@[simp] theorem appeal_relabelFirms (σ : Equiv.Perm ι) (x : Profile ι α) (i : ι) (l : κ) :
    N.appeal (relabelFirms σ x) i l = N.appeal x (σ.symm i) l := by
  simp [appeal]

omit [DecidableEq ι] in
/-- The aggregate denominator is invariant under a permutation of firm labels. -/
@[simp] theorem denom_relabelFirms (σ : Equiv.Perm ι) (x : Profile ι α) (l : κ) :
    N.denom (relabelFirms σ x) l = N.denom x l := by
  rw [denom, denom]
  simp only [appeal_relabelFirms]
  congr 1
  exact Equiv.sum_comp σ.symm fun i ↦ N.appeal x i l

omit [DecidableEq ι] in
/-- A firm's rival appeal is relabeled with its own block. -/
@[simp] theorem rival_relabelFirms (σ : Equiv.Perm ι) (x : Profile ι α) (i : ι) (l : κ) :
    N.rival (relabelFirms σ x) i l = N.rival x (σ.symm i) l := by
  simp [rival]

end Nodes

/-- Replacing a block commutes with relabeling when the replaced firm is relabeled too. -/
theorem replaceFirm_relabelFirms (σ : Equiv.Perm ι) (x : Profile ι α) (i : ι)
    (q : α → ℝ) :
    replaceFirm (relabelFirms σ x) i q =
      relabelFirms σ (replaceFirm x (σ.symm i) q) := by
  ext p
  rcases p with ⟨j, a⟩
  by_cases hji : j = i
  · subst j
    simp
  · have hsymm : σ.symm j ≠ σ.symm i := fun h ↦ hji (σ.symm.injective h)
    simp [replaceFirm, hji, hsymm]

omit [DecidableEq ι] in
/-- With common firm costs, relabeling a profile only relabels whose payoff is being read. -/
@[simp] theorem payoff_relabelFirms (N : Nodes κ α) (c₀ : α → ℝ) (H₀ : α → α → ℝ)
    (σ : Equiv.Perm ι) (i : ι) (x : Profile ι α) :
    payoff N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) i (relabelFirms σ x) =
      payoff N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) (σ.symm i) x := by
  simp [payoff]

/-- A firm permutation carries every Nash equilibrium of the common-cost game to another Nash
equilibrium of the same game. -/
theorem isNashOrthant_relabelFirms (N : Nodes κ α) (c₀ : α → ℝ) (H₀ : α → α → ℝ)
    (σ : Equiv.Perm ι) {x : Profile ι α}
    (hx : IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) x) :
    IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) (relabelFirms σ x) := by
  refine ⟨?_, ?_⟩
  · rintro ⟨i, a⟩
    exact hx.1 (σ.symm i, a)
  · intro i q hq
    have hdev := hx.2 (σ.symm i) q hq
    rw [replaceFirm_relabelFirms, payoff_relabelFirms, payoff_relabelFirms]
    exact hdev

/-- Firm relabeling preserves and reflects the Nash property in a common-cost game. -/
theorem isNashOrthant_relabelFirms_iff (N : Nodes κ α) (c₀ : α → ℝ)
    (H₀ : α → α → ℝ) (σ : Equiv.Perm ι) (x : Profile ι α) :
    IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) (relabelFirms σ x) ↔
      IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) x := by
  constructor
  · intro hx
    have := isNashOrthant_relabelFirms N c₀ H₀ σ.symm hx
    simpa using this
  · exact isNashOrthant_relabelFirms N c₀ H₀ σ

/-! ### Unique symmetric equilibria -/

/-- A unique equilibrium of a common-cost game is fixed by every firm relabeling. -/
theorem relabelFirms_eq_of_unique_isNashOrthant (N : Nodes κ α) (c₀ : α → ℝ)
    (H₀ : α → α → ℝ) {v : Profile ι α}
    (hv : IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) v)
    (huniq : ∀ w, IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) w → w = v)
    (σ : Equiv.Perm ι) : relabelFirms σ v = v :=
  huniq _ (isNashOrthant_relabelFirms N c₀ H₀ σ hv)

/-- **Symmetry boundary for the rational-share game.** Under common linear and quadratic cost
primitives, the capstone's unique Nash equilibrium gives every pair of firms exactly the same
capacity at every activity. -/
@[blueprint "app:symmetry-proof"
  (statement := /--
    Identical primitives and uniqueness imply equal firm blocks. Assumptions: Common firm
    primitives and a unique Nash equilibrium.
  -/)]
theorem equilibrium_blocks_eq_of_common_primitives (N : Nodes κ α) (c₀ : α → ℝ)
    {H₀ : α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ a b, H₀ a b = H₀ b a) (hmu : 0 < mu)
    (hH : ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H₀ a b * u a * u b)
    (hc : ∀ a, 0 ≤ c₀ a) {v : Profile ι α}
    (hv : IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) v)
    (i j : ι) (a : α) : v (i, a) = v (j, a) := by
  obtain ⟨vstar, _, huniq⟩ := existsUnique_isNashOrthant_of_primitives
    N (fun _ : ι ↦ c₀) (fun _ _ _ ↦ hsymm _ _) hmu (fun _ u ↦ hH u) (fun _ a ↦ hc a)
  have hv_eq : v = vstar := huniq v hv
  have hrel : IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀)
      (relabelFirms (Equiv.swap i j) v) :=
    isNashOrthant_relabelFirms N c₀ H₀ (Equiv.swap i j) hv
  have hfixed : relabelFirms (Equiv.swap i j) v = v := (huniq _ hrel).trans hv_eq.symm
  have hcoord := congrArg (fun z : Profile ι α ↦ z (j, a)) hfixed
  by_cases hij : i = j
  · subst j
    rfl
  · simpa [Equiv.swap_apply_def, hij, Ne.symm hij] using hcoord

/-! ### Positive-mass normalized compositions -/

/-- Total capacity operated by one firm. -/
def firmMass (x : Profile ι α) (i : ι) : ℝ := ∑ a, x (i, a)

/-- A firm's activity composition, normalized by its total capacity. Economic uses should supply
strict positivity of `firmMass`; the definition remains total because real division is total. -/
noncomputable def normalizedComposition (x : Profile ι α) (i : ι) (a : α) : ℝ :=
  x (i, a) / firmMass x i

omit [DecidableEq ι] in
/-- Equal capacity blocks have equal total mass. -/
theorem firmMass_eq_of_blocks_eq {x : Profile ι α} {i j : ι}
    (hblocks : ∀ a, x (i, a) = x (j, a)) : firmMass x i = firmMass x j := by
  exact Finset.sum_congr rfl fun a _ ↦ hblocks a

/-- **Normalized-composition corollary.** If one equilibrium firm has positive total capacity,
then every other firm in the common-primitives game has the same positive mass and exactly the
same normalized activity composition. -/
theorem equilibrium_normalizedComposition_eq_of_common_primitives (N : Nodes κ α)
    (c₀ : α → ℝ) {H₀ : α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ a b, H₀ a b = H₀ b a) (hmu : 0 < mu)
    (hH : ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H₀ a b * u a * u b)
    (hc : ∀ a, 0 ≤ c₀ a) {v : Profile ι α}
    (hv : IsNashOrthant N (fun _ : ι ↦ c₀) (fun _ : ι ↦ H₀) v)
    (i j : ι) (hi : 0 < firmMass v i) :
    0 < firmMass v j ∧ normalizedComposition v i = normalizedComposition v j := by
  have hblocks : ∀ a, v (i, a) = v (j, a) :=
    equilibrium_blocks_eq_of_common_primitives N c₀ hsymm hmu hH hc hv i j
  have hmass : firmMass v i = firmMass v j := firmMass_eq_of_blocks_eq hblocks
  constructor
  · rwa [← hmass]
  · funext a
    rw [normalizedComposition, normalizedComposition, hblocks a, hmass]

end PricingPerspective.Operating.RationalShare
