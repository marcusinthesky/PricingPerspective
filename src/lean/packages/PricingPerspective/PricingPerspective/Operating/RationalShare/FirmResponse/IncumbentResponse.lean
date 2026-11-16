import PricingPerspective.Operating.RationalShare.FirmResponse.Response

/-!
# Exact response at a feasible incumbent profile

`Operating/RationalShare/FirmResponse/Response.lean` solves a focal firm's problem against an abstract strictly positive
rival field. This module supplies the model-facing bridge: a feasible full profile generates such
a field, the proximal centre is the incumbent firm's own block, and the frozen-field objective is
exactly the original unilateral replacement payoff minus its adjustment penalty.
-/

namespace PricingPerspective.Operating.RationalShare.Nodes

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse
  PricingPerspective.Operating.RationalShare.FirmResponse.Response

open Finset
open Response

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]
  [DecidableEq ι] [DecidableEq α]

variable (N : RationalShare.Nodes κ α)

omit [DecidableEq ι] [DecidableEq α] in
/-- At a nonnegative full profile, outside appeal makes every firm's frozen rival field strictly
positive, even when every other firm operates zero capacity. -/
theorem rival_pos_of_nonneg {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (l : κ) :
    0 < N.rival x i l := by
  have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
    Finset.single_le_sum (fun j _ ↦ N.appeal_nonneg hx j l) (Finset.mem_univ i)
  rw [RationalShare.Nodes.rival, RationalShare.Nodes.denom]
  linarith [N.outside_pos l]

end PricingPerspective.Operating.RationalShare.Nodes

namespace PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse.Response

open Finset
open Response

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]
  [DecidableEq ι] [DecidableEq α]

/-- Embed firm `i`'s incumbent activity coordinates as the centre of its block problem. -/
def incumbentBlock (x : Profile ι α) (i : ι) : Block α :=
  ofActivity fun a ↦ x (i, a)

omit [DecidableEq ι] [DecidableEq α] in
/-- The incumbent block reads back the corresponding full-profile capacity coordinate. -/
@[simp] theorem incumbentBlock_apply (x : Profile ι α) (i : ι) (a : α) :
    incumbentBlock x i ((), a) = x (i, a) := rfl

omit [DecidableEq ι] [DecidableEq α] in
/-- A nonnegative full profile gives every firm a feasible incumbent block. -/
theorem incumbentBlock_mem_orthant {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) :
    incumbentBlock x i ∈ (orthant : Set (Block α)) := by
  rintro ⟨j, a⟩
  cases j
  exact hx (i, a)

/-- The original unilateral replacement payoff net of an incumbent-centred Euclidean adjustment
penalty, written as an objective on one-firm blocks. -/
noncomputable def incumbentObjective (N : RationalShare.Nodes κ α) (c : ι → α → ℝ)
    (H : ι → α → α → ℝ) (x : Profile ι α) (i : ι) (kappa : ℝ)
    (q : Block α) : ℝ :=
  payoff N c H i (replaceFirm x i (activity q))
    - (kappa / 2) * ∑ a, (activity q a - x (i, a)) ^ 2

omit [DecidableEq α] in
/-- The frozen-field block payoff at the incumbent centre equals the original replacement
objective exactly. No approximation, derivative, or equilibrium condition is used. -/
theorem blockPayoff_eq_incumbentObjective (N : RationalShare.Nodes κ α)
    (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α)
    (i : ι) (kappa : ℝ) (q : Block α) :
    blockPayoff N (N.rival x i) (c i) (H i) kappa (incumbentBlock x i) q
      = incumbentObjective N c H x i kappa q := by
  have hbridge := blockPayoff_eq_payoff_replaceFirm_sub_penalty
    N c H x i kappa (incumbentBlock x i) (activity q)
  simpa [incumbentObjective] using hbridge

/-- A firm's incumbent-centred exact response against the rivals present in a full profile. -/
def IsIncumbentResponse (N : RationalShare.Nodes κ α) (c : ι → α → ℝ)
    (H : ι → α → α → ℝ) (x : Profile ι α) (i : ι) (kappa : ℝ)
    (q : Block α) : Prop :=
  IsExactResponse N (N.rival x i) (c i) (H i) kappa (incumbentBlock x i) q

omit [DecidableEq α] in
/-- Incumbent response means global maximisation of the original unilateral replacement payoff
net of the stated adjustment penalty over every nonnegative block. -/
theorem isIncumbentResponse_iff_maximizes (N : RationalShare.Nodes κ α)
    (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α)
    (i : ι) (kappa : ℝ) (q : Block α) :
    IsIncumbentResponse N c H x i kappa q ↔
      q ∈ orthant ∧ ∀ y ∈ orthant,
        incumbentObjective N c H x i kappa y ≤ incumbentObjective N c H x i kappa q := by
  rw [IsIncumbentResponse, IsExactResponse]
  constructor
  · rintro ⟨hq, hmax⟩
    refine ⟨hq, fun y hy ↦ ?_⟩
    rw [← blockPayoff_eq_incumbentObjective, ← blockPayoff_eq_incumbentObjective]
    exact hmax y hy
  · rintro ⟨hq, hmax⟩
    refine ⟨hq, fun y hy ↦ ?_⟩
    rw [blockPayoff_eq_incumbentObjective, blockPayoff_eq_incumbentObjective]
    exact hmax y hy

omit [DecidableEq ι] in
/-- **Unique exact response at a feasible incumbent.** Positive outside appeal supplies the frozen
rival-field premise; positive own curvature and nonnegative adjustment weight supply uniqueness. -/
theorem existsUnique_isIncumbentResponse (N : RationalShare.Nodes κ α)
    (c : ι → α → ℝ) {H : ι → α → α → ℝ} {x : Profile ι α} (i : ι)
    {mu kappa : ℝ} (hx : ∀ p, 0 ≤ x p) (hsymm : ∀ a b, H i a b = H i b a)
    (hmu : 0 < mu)
    (hH : ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hkappa : 0 ≤ kappa) :
    ∃! q, IsIncumbentResponse N c H x i kappa q := by
  exact existsUnique_isExactResponse N (N.rival x i) (Nodes.rival_pos_of_nonneg N hx i)
    (c i) (H := H i) (mu := mu) (kappa := kappa) (incumbentBlock x i)
    hsymm hmu hH hkappa

/-- **Unique maximizer of the original incumbent-centred objective.** This is the model-facing
form of `existsUnique_isIncumbentResponse`: its objective mentions only unilateral replacement in
the original full game and the Euclidean adjustment penalty. -/
theorem existsUnique_incumbentObjective_maximizer (N : RationalShare.Nodes κ α)
    (c : ι → α → ℝ) {H : ι → α → α → ℝ} {x : Profile ι α} (i : ι)
    {mu kappa : ℝ} (hx : ∀ p, 0 ≤ x p) (hsymm : ∀ a b, H i a b = H i b a)
    (hmu : 0 < mu)
    (hH : ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hkappa : 0 ≤ kappa) :
    ∃! q, q ∈ orthant ∧ ∀ y ∈ orthant,
      incumbentObjective N c H x i kappa y ≤ incumbentObjective N c H x i kappa q := by
  obtain ⟨q, hq, huniq⟩ :=
    existsUnique_isIncumbentResponse N c i hx hsymm hmu hH hkappa
  refine ⟨q, (isIncumbentResponse_iff_maximizes N c H x i kappa q).1 hq, fun y hy ↦ ?_⟩
  exact huniq y ((isIncumbentResponse_iff_maximizes N c H x i kappa y).2 hy)

/-! ### Satisfiability -/

/-- A one-node, one-activity demand side for the incumbent-response acceptance check. -/
def incumbentProbeNodes : RationalShare.Nodes (Fin 1) (Fin 1) where
  revenue := fun _ ↦ 1
  kernel := fun _ _ ↦ 1
  outside := fun _ ↦ 1
  revenue_nonneg := fun _ ↦ zero_le_one
  kernel_nonneg := fun _ _ ↦ zero_le_one
  outside_pos := fun _ ↦ one_pos

/-- The incumbent-response hypothesis set is jointly satisfiable in a concrete feasible economy. -/
theorem incumbentProbe_existsUnique :
    ∃! q, IsIncumbentResponse incumbentProbeNodes
      (fun _ _ ↦ (0 : ℝ)) (fun _ _ _ ↦ (1 : ℝ))
      (0 : Profile (Fin 1) (Fin 1)) 0 1 q := by
  apply existsUnique_isIncumbentResponse incumbentProbeNodes (fun _ _ ↦ 0) 0
  · intro p
    simp
  · intro a b
    rfl
  · exact one_pos
  · intro u
    simp
    nlinarith [sq_nonneg (u 0)]
  · exact zero_le_one

end PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse
