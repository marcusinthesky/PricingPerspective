import PricingPerspective.Operating.RationalShare
import Mathlib.Data.NNReal.Defs

/-!
# Two-date revealed-preference restrictions

This module records an observable restriction implied by two choices under a stable cost
function.  If an old position is preferred in the old environment and a new position is
preferred in the new environment, then the new position's revenue advantage over the old
position cannot have fallen.  Adding the two choice inequalities cancels the unobserved cost.

The result is a retrospective consistency restriction.  It does not identify why the
environment changed, assert that observed choices are equilibria, or turn a normalized
composition into unnormalized capacity.  Mutual feasibility enters through the two cross-choice
comparisons: each observed position must be an admissible alternative at the other date.
-/

namespace PricingPerspective.Operating.RationalShare.Reallocation.RevealedPreference

open PricingPerspective.Operating.VariationalInequality

variable {Position Environment : Type*}

/-- Operating revenue net of a position-specific cost.  The same `cost` function at both
environments is the stability restriction that makes cost cancellation possible. -/
def netPayoff (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (position : Position) (environment : Environment) : ℝ :=
  revenue position environment - cost position

/-- The two-date reallocation statistic

`J = [R(new, newEnvironment) - R(old, newEnvironment)]
    - [R(new, oldEnvironment) - R(old, oldEnvironment)]`.

It compares the new position's relative operating-revenue advantage across environments. -/
def reallocationGain (revenue : Position → Environment → ℝ) (oldPosition newPosition : Position)
    (oldEnvironment newEnvironment : Environment) : ℝ :=
  (revenue newPosition newEnvironment - revenue oldPosition newEnvironment) -
    (revenue newPosition oldEnvironment - revenue oldPosition oldEnvironment)

/-- Two cross-date choice inequalities imply a nonnegative reallocation gain.

The old inequality compares the new position with the chosen old position in the old
environment; the new inequality makes the reverse comparison in the new environment.  Thus the
hypotheses are exactly the pairwise consequences of optimization under mutual feasibility. -/
theorem reallocationGain_nonneg_of_pairwise_choice
    (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldEnvironment newEnvironment : Environment)
    (oldChoice :
      netPayoff revenue cost newPosition oldEnvironment ≤
        netPayoff revenue cost oldPosition oldEnvironment)
    (newChoice :
      netPayoff revenue cost oldPosition newEnvironment ≤
        netPayoff revenue cost newPosition newEnvironment) :
    0 ≤ reallocationGain revenue oldPosition newPosition oldEnvironment newEnvironment := by
  simp only [netPayoff, reallocationGain] at oldChoice newChoice ⊢
  linarith

/-- A position maximizes net operating payoff over a declared feasible set.  Membership of the
chosen position itself is deliberately separate: the reallocation theorem needs only the two
cross-feasibility facts used to instantiate the maximizing inequalities. -/
def IsOptimalOn (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (feasible : Set Position) (environment : Environment) (position : Position) : Prop :=
  ∀ alternative ∈ feasible,
    netPayoff revenue cost alternative environment ≤ netPayoff revenue cost position environment

/-- Optimality at each date and mutual feasibility of the two observed positions imply a
nonnegative reallocation gain.  The feasible sets may differ across dates. -/
theorem reallocationGain_nonneg_of_optimalOn
    (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (oldFeasible newFeasible : Set Position) (oldPosition newPosition : Position)
    (oldEnvironment newEnvironment : Environment)
    (oldOptimal : IsOptimalOn revenue cost oldFeasible oldEnvironment oldPosition)
    (newOptimal : IsOptimalOn revenue cost newFeasible newEnvironment newPosition)
    (newFeasibleAtOld : newPosition ∈ oldFeasible)
    (oldFeasibleAtNew : oldPosition ∈ newFeasible) :
    0 ≤ reallocationGain revenue oldPosition newPosition oldEnvironment newEnvironment :=
  reallocationGain_nonneg_of_pairwise_choice revenue cost oldPosition newPosition
    oldEnvironment newEnvironment (oldOptimal newPosition newFeasibleAtOld)
    (newOptimal oldPosition oldFeasibleAtNew)

/-- With nonnegative payoff-regret allowances, approximate choice implies
`J ≥ -(oldAllowance + newAllowance)`.

The allowances have type `NNReal`, so their signs are part of the statement.  They bound payoff
regret directly; a numerical solver tolerance does not satisfy these hypotheses without a
separate payoff-error argument. -/
theorem reallocationGain_ge_neg_allowances
    (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldEnvironment newEnvironment : Environment)
    (oldAllowance newAllowance : NNReal)
    (oldChoice :
      netPayoff revenue cost newPosition oldEnvironment ≤
        netPayoff revenue cost oldPosition oldEnvironment + (oldAllowance : ℝ))
    (newChoice :
      netPayoff revenue cost oldPosition newEnvironment ≤
        netPayoff revenue cost newPosition newEnvironment + (newAllowance : ℝ)) :
    -((oldAllowance : ℝ) + (newAllowance : ℝ)) ≤
      reallocationGain revenue oldPosition newPosition oldEnvironment newEnvironment := by
  simp only [netPayoff, reallocationGain] at oldChoice newChoice ⊢
  linarith

/-- If adopting the new position in the new environment beats staying at the old position after
paying `adjustmentCost`, then the reallocation gain is at least that adjustment cost.  The old
position must still beat the new position in the old environment. -/
theorem adjustmentCost_le_reallocationGain
    (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldEnvironment newEnvironment : Environment)
    (adjustmentCost : ℝ)
    (oldChoice :
      netPayoff revenue cost newPosition oldEnvironment ≤
        netPayoff revenue cost oldPosition oldEnvironment)
    (newChoiceAfterAdjustment :
      netPayoff revenue cost oldPosition newEnvironment + adjustmentCost ≤
        netPayoff revenue cost newPosition newEnvironment) :
    adjustmentCost ≤
      reallocationGain revenue oldPosition newPosition oldEnvironment newEnvironment := by
  simp only [netPayoff, reallocationGain] at oldChoice newChoiceAfterAdjustment ⊢
  linarith

/-- A nonnegative adjustment cost and the one-step choice inequalities imply a nonnegative
reallocation gain.  The stronger lower bound by the adjustment cost is available from
`adjustmentCost_le_reallocationGain`. -/
theorem reallocationGain_nonneg_of_adjustmentCost
    (revenue : Position → Environment → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldEnvironment newEnvironment : Environment)
    (adjustmentCost : ℝ) (adjustmentCost_nonneg : 0 ≤ adjustmentCost)
    (oldChoice :
      netPayoff revenue cost newPosition oldEnvironment ≤
        netPayoff revenue cost oldPosition oldEnvironment)
    (newChoiceAfterAdjustment :
      netPayoff revenue cost oldPosition newEnvironment + adjustmentCost ≤
        netPayoff revenue cost newPosition newEnvironment) :
    0 ≤ reallocationGain revenue oldPosition newPosition oldEnvironment newEnvironment :=
  adjustmentCost_nonneg.trans
    (adjustmentCost_le_reallocationGain revenue cost oldPosition newPosition oldEnvironment
      newEnvironment adjustmentCost oldChoice newChoiceAfterAdjustment)

/-! ## Finite-change rational-share algebra -/

/-- The contribution of one demand node to the two-date reallocation statistic.

Here `a0` and `a1` are the firm's old and new appeals, while `b0` and `b1` are the
corresponding outside-plus-rival appeals. -/
noncomputable def nodeReallocationGain (r a0 a1 b0 b1 : ℝ) : ℝ :=
  (r * a1 / (a1 + b1) - r * a0 / (a0 + b1)) -
    (r * a1 / (a1 + b0) - r * a0 / (a0 + b0))

/-- The product of the four share denominators in the nodewise finite-change identity. -/
def nodeDenominator (a0 a1 b0 b1 : ℝ) : ℝ :=
  (a0 + b0) * (a1 + b0) * (a0 + b1) * (a1 + b1)

/-- The exact four-term rational-share factorization at one demand node.

No sign condition is used.  The four nonzero hypotheses are exactly those needed to put the four
shares over a common denominator. -/
theorem nodeReallocationGain_factorization (r a0 a1 b0 b1 : ℝ)
    (h00 : a0 + b0 ≠ 0) (h10 : a1 + b0 ≠ 0)
    (h01 : a0 + b1 ≠ 0) (h11 : a1 + b1 ≠ 0) :
    nodeReallocationGain r a0 a1 b0 b1 =
      r * ((a1 - a0) * (b1 - b0)) * (a0 * a1 - b0 * b1) /
        nodeDenominator a0 a1 b0 b1 := by
  simp only [nodeReallocationGain, nodeDenominator]
  field_simp
  ring

/-- The nonnegative weight on local own-rival anti-comovement under rival-product dominance. -/
noncomputable def crowdingCoefficient (r a0 a1 b0 b1 : ℝ) : ℝ :=
  r * (b0 * b1 - a0 * a1) / nodeDenominator a0 a1 b0 b1

/-- The nodewise gain is minus the crowding coefficient times the product of the own- and
rival-appeal changes. -/
theorem nodeReallocationGain_eq_neg_coefficient_mul (r a0 a1 b0 b1 : ℝ)
    (h00 : a0 + b0 ≠ 0) (h10 : a1 + b0 ≠ 0)
    (h01 : a0 + b1 ≠ 0) (h11 : a1 + b1 ≠ 0) :
    nodeReallocationGain r a0 a1 b0 b1 =
      -crowdingCoefficient r a0 a1 b0 b1 * ((a1 - a0) * (b1 - b0)) := by
  rw [nodeReallocationGain_factorization r a0 a1 b0 b1 h00 h10 h01 h11]
  simp only [crowdingCoefficient]
  ring

/-- Nonnegative revenue, positive share denominators, and weak rival-product dominance make the
crowding coefficient nonnegative. -/
theorem crowdingCoefficient_nonneg (r a0 a1 b0 b1 : ℝ) (hr : 0 ≤ r)
    (h00 : 0 < a0 + b0) (h10 : 0 < a1 + b0)
    (h01 : 0 < a0 + b1) (h11 : 0 < a1 + b1)
    (hdominance : a0 * a1 ≤ b0 * b1) :
    0 ≤ crowdingCoefficient r a0 a1 b0 b1 := by
  apply div_nonneg
  · exact mul_nonneg hr (sub_nonneg.mpr hdominance)
  · simp only [nodeDenominator]
    positivity

/-- Positive revenue and strict rival-product dominance make the crowding coefficient strictly
positive when all four share denominators are positive. -/
theorem crowdingCoefficient_pos (r a0 a1 b0 b1 : ℝ) (hr : 0 < r)
    (h00 : 0 < a0 + b0) (h10 : 0 < a1 + b0)
    (h01 : 0 < a0 + b1) (h11 : 0 < a1 + b1)
    (hdominance : a0 * a1 < b0 * b1) :
    0 < crowdingCoefficient r a0 a1 b0 b1 := by
  apply div_pos
  · exact mul_pos hr (sub_pos.mpr hdominance)
  · simp only [nodeDenominator]
    positivity

/-- Under weak rival-product dominance, local own-rival anti-comovement is sufficient for a
nonnegative nodewise reallocation gain. -/
theorem nodeReallocationGain_nonneg_of_antiComovement (r a0 a1 b0 b1 : ℝ) (hr : 0 ≤ r)
    (h00 : 0 < a0 + b0) (h10 : 0 < a1 + b0)
    (h01 : 0 < a0 + b1) (h11 : 0 < a1 + b1)
    (hdominance : a0 * a1 ≤ b0 * b1)
    (hanti : (a1 - a0) * (b1 - b0) ≤ 0) :
    0 ≤ nodeReallocationGain r a0 a1 b0 b1 := by
  rw [nodeReallocationGain_eq_neg_coefficient_mul r a0 a1 b0 b1 h00.ne' h10.ne'
    h01.ne' h11.ne']
  exact mul_nonneg_of_nonpos_of_nonpos
    (neg_nonpos.mpr (crowdingCoefficient_nonneg r a0 a1 b0 b1 hr h00 h10 h01 h11 hdominance))
    hanti

/-- With positive revenue and strict rival-product dominance, a nodewise gain is nonnegative
exactly when own and rival appeal changes weakly anti-comove. -/
theorem nodeReallocationGain_nonneg_iff_antiComovement (r a0 a1 b0 b1 : ℝ) (hr : 0 < r)
    (h00 : 0 < a0 + b0) (h10 : 0 < a1 + b0)
    (h01 : 0 < a0 + b1) (h11 : 0 < a1 + b1)
    (hdominance : a0 * a1 < b0 * b1) :
    0 ≤ nodeReallocationGain r a0 a1 b0 b1 ↔
      (a1 - a0) * (b1 - b0) ≤ 0 := by
  rw [nodeReallocationGain_eq_neg_coefficient_mul r a0 a1 b0 b1 h00.ne' h10.ne'
    h01.ne' h11.ne']
  let c := crowdingCoefficient r a0 a1 b0 b1
  have hc : 0 < c := crowdingCoefficient_pos r a0 a1 b0 b1 hr h00 h10 h01 h11 hdominance
  constructor
  · intro hgain
    by_contra hanti
    have hchange : 0 < (a1 - a0) * (b1 - b0) := lt_of_not_ge hanti
    have : -c * ((a1 - a0) * (b1 - b0)) < 0 :=
      mul_neg_of_neg_of_pos (neg_neg_of_pos hc) hchange
    exact (not_lt_of_ge hgain) this
  · intro hanti
    exact mul_nonneg_of_nonpos_of_nonpos (neg_nonpos.mpr hc.le) hanti

/-- A zero revenue weight makes the node's reallocation gain zero. -/
@[simp] theorem nodeReallocationGain_zero_revenue (a0 a1 b0 b1 : ℝ) :
    nodeReallocationGain 0 a0 a1 b0 b1 = 0 := by
  simp [nodeReallocationGain]

/-- No own-appeal change makes the node's reallocation gain zero. -/
@[simp] theorem nodeReallocationGain_same_own (r a b0 b1 : ℝ) :
    nodeReallocationGain r a a b0 b1 = 0 := by
  simp [nodeReallocationGain]

/-- No rival-appeal change makes the node's reallocation gain zero. -/
@[simp] theorem nodeReallocationGain_same_rival (r a0 a1 b : ℝ) :
    nodeReallocationGain r a0 a1 b b = 0 := by
  simp [nodeReallocationGain]

/-- Equality of the own- and rival-appeal products is the remaining zero boundary in the exact
factorization, provided the four share denominators are nonzero. -/
theorem nodeReallocationGain_zero_of_product_eq (r a0 a1 b0 b1 : ℝ)
    (h00 : a0 + b0 ≠ 0) (h10 : a1 + b0 ≠ 0)
    (h01 : a0 + b1 ≠ 0) (h11 : a1 + b1 ≠ 0)
    (hproduct : a0 * a1 = b0 * b1) :
    nodeReallocationGain r a0 a1 b0 b1 = 0 := by
  rw [nodeReallocationGain_factorization r a0 a1 b0 b1 h00 h10 h01 h11, hproduct,
    sub_self, mul_zero, zero_div]

section FiniteChange

variable {Node : Type*} [Fintype Node]

/-- The finite rational-share reallocation statistic obtained by summing node contributions. -/
noncomputable def finiteReallocationGain
    (revenue ownOld ownNew rivalOld rivalNew : Node → ℝ) : ℝ :=
  ∑ l, nodeReallocationGain (revenue l) (ownOld l) (ownNew l) (rivalOld l) (rivalNew l)

/-- The finite statistic is a negative weighted sum of local own-rival comovement products. -/
theorem finiteReallocationGain_eq_neg_sum
    (revenue ownOld ownNew rivalOld rivalNew : Node → ℝ)
    (h00 : ∀ l, ownOld l + rivalOld l ≠ 0)
    (h10 : ∀ l, ownNew l + rivalOld l ≠ 0)
    (h01 : ∀ l, ownOld l + rivalNew l ≠ 0)
    (h11 : ∀ l, ownNew l + rivalNew l ≠ 0) :
    finiteReallocationGain revenue ownOld ownNew rivalOld rivalNew =
      -∑ l, crowdingCoefficient (revenue l) (ownOld l) (ownNew l) (rivalOld l)
        (rivalNew l) * ((ownNew l - ownOld l) * (rivalNew l - rivalOld l)) := by
  rw [finiteReallocationGain, ← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro l _
  rw [nodeReallocationGain_eq_neg_coefficient_mul _ _ _ _ _ (h00 l) (h10 l) (h01 l)
    (h11 l), neg_mul]

omit [Fintype Node] in
/-- Under the finite rival-dominance conditions, every weight in the aggregate
anti-comovement representation is nonnegative. -/
theorem finiteCrowdingCoefficient_nonneg
    (revenue ownOld ownNew rivalOld rivalNew : Node → ℝ)
    (hrevenue : ∀ l, 0 ≤ revenue l)
    (h00 : ∀ l, 0 < ownOld l + rivalOld l)
    (h10 : ∀ l, 0 < ownNew l + rivalOld l)
    (h01 : ∀ l, 0 < ownOld l + rivalNew l)
    (h11 : ∀ l, 0 < ownNew l + rivalNew l)
    (hdominance : ∀ l, ownOld l * ownNew l ≤ rivalOld l * rivalNew l) :
    ∀ l, 0 ≤ crowdingCoefficient (revenue l) (ownOld l) (ownNew l) (rivalOld l)
      (rivalNew l) := fun l ↦
  crowdingCoefficient_nonneg _ _ _ _ _ (hrevenue l) (h00 l) (h10 l) (h01 l) (h11 l)
    (hdominance l)

/-- Two nodes with positive weights, positive denominators, and strict rival-product dominance
can have aggregate gain zero while the first node's gain is negative.  Thus aggregate
nonnegativity does not imply nodewise nonnegativity. -/
theorem aggregate_nonneg_with_negative_node_example :
    let revenue : Fin 2 → ℝ := fun _ ↦ 1
    let ownOld : Fin 2 → ℝ := fun l ↦ if l = 0 then 0 else 1
    let ownNew : Fin 2 → ℝ := fun l ↦ if l = 0 then 1 else 0
    let rivalOld : Fin 2 → ℝ := fun _ ↦ 1
    let rivalNew : Fin 2 → ℝ := fun _ ↦ 2
    (∀ l, 0 < revenue l ∧
      0 < ownOld l + rivalOld l ∧ 0 < ownNew l + rivalOld l ∧
      0 < ownOld l + rivalNew l ∧ 0 < ownNew l + rivalNew l ∧
      ownOld l * ownNew l < rivalOld l * rivalNew l) ∧
    finiteReallocationGain revenue ownOld ownNew rivalOld rivalNew = 0 ∧
    nodeReallocationGain (revenue 0) (ownOld 0) (ownNew 0) (rivalOld 0) (rivalNew 0) =
      -(1 / 6 : ℝ) := by
  dsimp
  constructor
  · intro l
    fin_cases l <;> norm_num
  constructor
  · norm_num [finiteReallocationGain, nodeReallocationGain, Fin.sum_univ_two]
  · norm_num [nodeReallocationGain]

end FiniteChange

/-! ## Finite fixed-scale rational-share specialization -/

open Finset

variable {κ α : Type*} [Fintype κ] [Fintype α]

/-- A finite normalized composition: nonnegative activity weights with unit total mass. -/
def Composition (α : Type*) [Fintype α] :=
  { weights : α → ℝ // (∀ a, 0 ≤ weights a) ∧ ∑ a, weights a = 1 }

namespace Composition

/-- Convert a normalized composition to capacity at a declared common nonnegative scale. -/
def capacity (scale : NNReal) (composition : Composition α) : α → ℝ :=
  fun a ↦ (scale : ℝ) * composition.1 a

/-- Fixed nonnegative scale preserves coordinatewise nonnegativity of a composition. -/
theorem capacity_nonneg (scale : NNReal) (composition : Composition α) (a : α) :
    0 ≤ capacity scale composition a :=
  mul_nonneg scale.coe_nonneg (composition.2.1 a)

/-- A normalized composition converted at fixed scale has total capacity equal to that scale. -/
theorem sum_capacity (scale : NNReal) (composition : Composition α) :
    ∑ a, capacity scale composition a = (scale : ℝ) := by
  simp only [capacity, ← Finset.mul_sum, composition.2.2, mul_one]

end Composition

/-- A frozen finite rival field with the strictly positive outside-plus-rival appeal required by
the rational-share model. -/
def PositiveRivalField (κ : Type*) [Fintype κ] :=
  { rival : κ → ℝ // ∀ l, 0 < rival l }

/-- Fixed-scale composition appeal plus a positive rival field gives a strictly positive share
denominator at every node. -/
theorem fixedScaleDenominator_pos (N : RationalShare.Nodes κ α) (scale : NNReal)
    (composition : Composition α) (rival : PositiveRivalField κ) (l : κ) :
    0 < N.blockAppeal (Composition.capacity scale composition) l + rival.1 l :=
  add_pos_of_nonneg_of_pos
    (N.blockAppeal_nonneg (Composition.capacity_nonneg scale composition) l) (rival.2 l)

/-- Rational-share operating revenue for a finite normalized composition at a common fixed scale
against a frozen rival-appeal field.

The same `scale` is used for every position compared by the specialization below.  This is an
explicit fixed-scale composition model; it makes no claim that a theorem about unnormalized
capacity survives normalization when scale changes. -/
noncomputable def fixedScaleRevenue (N : RationalShare.Nodes κ α) (scale : NNReal)
    (composition : Composition α) (rival : PositiveRivalField κ) : ℝ :=
  ∑ l, N.revenue l * N.blockAppeal (Composition.capacity scale composition) l /
    (N.blockAppeal (Composition.capacity scale composition) l + rival.1 l)

/-- The generic two-date gain for fixed-scale rational-share revenue is exactly the sum of its
nodewise rational-share gains. -/
theorem fixedScale_reallocationGain_eq_finiteReallocationGain
    (N : RationalShare.Nodes κ α) (scale : NNReal)
    (oldComposition newComposition : Composition α)
    (oldRival newRival : PositiveRivalField κ) :
    reallocationGain (fixedScaleRevenue N scale) oldComposition newComposition
      oldRival newRival =
    finiteReallocationGain N.revenue
      (fun l ↦ N.blockAppeal (Composition.capacity scale oldComposition) l)
      (fun l ↦ N.blockAppeal (Composition.capacity scale newComposition) l)
      oldRival.1 newRival.1 := by
  simp only [reallocationGain, fixedScaleRevenue, finiteReallocationGain, nodeReallocationGain]
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]

/-- Fixed-scale rational-share reallocation is the negative coefficient-weighted sum of local
own-rival appeal comovements. -/
theorem fixedScale_reallocationGain_eq_neg_sum
    (N : RationalShare.Nodes κ α) (scale : NNReal)
    (oldComposition newComposition : Composition α)
    (oldRival newRival : PositiveRivalField κ) :
    reallocationGain (fixedScaleRevenue N scale) oldComposition newComposition
      oldRival newRival =
    -∑ l, crowdingCoefficient (N.revenue l)
      (N.blockAppeal (Composition.capacity scale oldComposition) l)
      (N.blockAppeal (Composition.capacity scale newComposition) l)
      (oldRival.1 l) (newRival.1 l) *
      ((N.blockAppeal (Composition.capacity scale newComposition) l -
          N.blockAppeal (Composition.capacity scale oldComposition) l) *
        (newRival.1 l - oldRival.1 l)) := by
  rw [fixedScale_reallocationGain_eq_finiteReallocationGain]
  apply finiteReallocationGain_eq_neg_sum
  · exact fun l ↦ (fixedScaleDenominator_pos N scale oldComposition oldRival l).ne'
  · exact fun l ↦ (fixedScaleDenominator_pos N scale newComposition oldRival l).ne'
  · exact fun l ↦ (fixedScaleDenominator_pos N scale oldComposition newRival l).ne'
  · exact fun l ↦ (fixedScaleDenominator_pos N scale newComposition newRival l).ne'

/-- If each node is weakly rival-product dominant, every coefficient in the fixed-scale
aggregate anti-comovement identity is nonnegative. -/
theorem fixedScale_crowdingCoefficient_nonneg
    (N : RationalShare.Nodes κ α) (scale : NNReal)
    (oldComposition newComposition : Composition α)
    (oldRival newRival : PositiveRivalField κ)
    (hdominance : ∀ l,
      N.blockAppeal (Composition.capacity scale oldComposition) l *
          N.blockAppeal (Composition.capacity scale newComposition) l ≤
        oldRival.1 l * newRival.1 l) :
    ∀ l, 0 ≤ crowdingCoefficient (N.revenue l)
      (N.blockAppeal (Composition.capacity scale oldComposition) l)
      (N.blockAppeal (Composition.capacity scale newComposition) l)
      (oldRival.1 l) (newRival.1 l) := by
  apply finiteCrowdingCoefficient_nonneg
  · exact N.revenue_nonneg
  · exact fixedScaleDenominator_pos N scale oldComposition oldRival
  · exact fixedScaleDenominator_pos N scale newComposition oldRival
  · exact fixedScaleDenominator_pos N scale oldComposition newRival
  · exact fixedScaleDenominator_pos N scale newComposition newRival
  · exact hdominance

/-- The cost-cancelling restriction specialized to finite fixed-scale rational-share revenue.

The rival fields may differ across dates, while the scale and composition cost remain fixed.
The two displayed inequalities remain maintained optimizing-choice assumptions; a nonnegative
conclusion is compatibility with those assumptions, not identification of strategic rivalry. -/
theorem fixedScale_reallocationGain_nonneg
    (N : RationalShare.Nodes κ α) (scale : NNReal) (cost : Composition α → ℝ)
    (oldComposition newComposition : Composition α)
    (oldRival newRival : PositiveRivalField κ)
    (oldChoice :
      netPayoff (fixedScaleRevenue N scale) cost newComposition oldRival ≤
        netPayoff (fixedScaleRevenue N scale) cost oldComposition oldRival)
    (newChoice :
      netPayoff (fixedScaleRevenue N scale) cost oldComposition newRival ≤
        netPayoff (fixedScaleRevenue N scale) cost newComposition newRival) :
    0 ≤ reallocationGain (fixedScaleRevenue N scale) oldComposition newComposition
      oldRival newRival :=
  reallocationGain_nonneg_of_pairwise_choice (fixedScaleRevenue N scale) cost
    oldComposition newComposition oldRival newRival oldChoice newChoice

end PricingPerspective.Operating.RationalShare.Reallocation.RevealedPreference
