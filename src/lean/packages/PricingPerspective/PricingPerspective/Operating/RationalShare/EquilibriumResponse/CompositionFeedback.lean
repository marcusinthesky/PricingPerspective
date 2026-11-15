import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.ResponseOperator

/-!
# Composition feedback and a many-firm separable cost shock

This module separates two statements used to interpret local equilibrium responses.

First, finite capacity normalization has derivative

`D N_q(h)_a = (h_a - (q_a / S) * sum h) / S`, where `S = sum q > 0`.
Its kernel consists exactly of radial changes proportional to `q`.  Consequently, full and
frozen response derivatives induce the same composition derivative exactly when their capacity
feedback is radial.

Second, the affected-activity Jacobian in a separable many-firm rational-share benchmark has the
rank-one form `diag(a) + c 1ᵀ`.  Under positive own coefficients and strict strategic
substitution, a cost reduction at one firm makes that firm expand and every unshocked rival
contract at the affected activity.  The algebra here is conditional on the displayed local
system.  Identifying that system with an equilibrium derivative still requires an interior,
differentiable equilibrium path with a fixed active set.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse.CompositionFeedback

open PricingPerspective.Operating.VariationalInequality

open Finset

variable {α : Type*} [Fintype α]

/-! ## The kernel of finite-capacity normalization -/

/-- The coordinate derivative of normalization at capacity `q` in direction `h`.

The economic use assumes positive total mass.  The definition remains total so the positivity
premise can be stated on the theorems that use it. -/
noncomputable def compositionDerivative (q h : α → ℝ) (a : α) : ℝ :=
  (h a - q a / (∑ b, q b) * ∑ b, h b) / ∑ b, q b

/-- The existing pathwise normalization result has exactly the coordinate derivative recorded
by `compositionDerivative`. -/
theorem hasDerivAt_normalizedPath {q h : α → ℝ} (hS : 0 < ∑ b, q b) (a : α) :
    HasDerivAt (fun t : ℝ ↦ (q a + t * h a) / ∑ b, (q b + t * h b))
      (compositionDerivative q h a) 0 := by
  simpa only [compositionDerivative] using hasDerivAt_composition hS a

/-- The normalization derivative is linear in its direction argument. -/
theorem compositionDerivative_sub (q hFull hDirect : α → ℝ) (a : α) :
    compositionDerivative q (fun b ↦ hFull b - hDirect b) a =
      compositionDerivative q hFull a - compositionDerivative q hDirect a := by
  simp only [compositionDerivative, Finset.sum_sub_distrib]
  ring

/-- **Exact radial kernel of normalization.**  At positive total capacity, every coordinate of
the normalization derivative vanishes if and only if the capacity direction is proportional to
the initial capacity vector. -/
theorem compositionDerivative_eq_zero_iff_proportional {q h : α → ℝ}
    (hS : 0 < ∑ b, q b) :
    (∀ a, compositionDerivative q h a = 0) ↔
      ∃ lambda : ℝ, ∀ a, h a = lambda * q a := by
  constructor
  · intro hz
    refine ⟨(∑ b, h b) / (∑ b, q b), fun a ↦ ?_⟩
    have hnum : h a - q a / (∑ b, q b) * ∑ b, h b = 0 := by
      rcases div_eq_zero_iff.mp (hz a) with h | h
      · exact h
      · exact (hS.ne' h).elim
    rw [sub_eq_zero] at hnum
    calc
      h a = q a / (∑ b, q b) * ∑ b, h b := hnum
      _ = ((∑ b, h b) / (∑ b, q b)) * q a := by ring
  · rintro ⟨lambda, hradial⟩ a
    have hsum : (∑ b, h b) = lambda * ∑ b, q b := by
      calc
        (∑ b, h b) = ∑ b, lambda * q b :=
          Finset.sum_congr rfl fun b _ ↦ hradial b
        _ = lambda * ∑ b, q b := by rw [Finset.mul_sum]
    rw [compositionDerivative, hradial a, hsum]
    field_simp [hS.ne']
    ring

/-- **Full versus direct composition feedback.**  At positive total capacity, full and direct
capacity responses have identical normalized derivatives exactly when their difference is a
proportional scale change. -/
@[blueprint "eq:proportional-feedback"
  (statement := /--
    Full and direct normalized derivatives coincide exactly when capacity feedback is
    proportional to the initial allocation. Assumptions: Finite activity set and strictly
    positive total initial capacity.
  -/)]
theorem full_eq_direct_iff_feedback_proportional {q hFull hDirect : α → ℝ}
    (hS : 0 < ∑ b, q b) :
    (∀ a, compositionDerivative q hFull a = compositionDerivative q hDirect a) ↔
      ∃ lambda : ℝ, ∀ a, hFull a - hDirect a = lambda * q a := by
  rw [← compositionDerivative_eq_zero_iff_proportional
    (q := q) (h := fun b ↦ hFull b - hDirect b) hS]
  constructor
  · intro heq a
    rw [compositionDerivative_sub, heq a, sub_self]
  · intro hz a
    have ha := hz a
    rw [compositionDerivative_sub] at ha
    linarith

/-- A response confined to one activity coordinate. -/
def singleActivityDirection [DecidableEq α] (affected : α) (u : ℝ) : α → ℝ :=
  fun a ↦ if a = affected then u else 0

omit [Fintype α] in
/-- The isolated response equals its scalar response at the affected activity. -/
@[simp]
theorem singleActivityDirection_same [DecidableEq α] (affected : α) (u : ℝ) :
    singleActivityDirection affected u affected = u := by
  simp [singleActivityDirection]

omit [Fintype α] in
/-- The isolated response vanishes at every unaffected activity. -/
@[simp]
theorem singleActivityDirection_of_ne [DecidableEq α] {affected other : α}
    (h : other ≠ affected) (u : ℝ) :
    singleActivityDirection affected u other = 0 := by
  simp [singleActivityDirection, h]

/-- The total capacity response of an isolated activity change is the scalar response. -/
@[simp]
theorem sum_singleActivityDirection [DecidableEq α] (affected : α) (u : ℝ) :
    ∑ a, singleActivityDirection affected u a = u := by
  simp [singleActivityDirection]

/-- A nonzero response at one activity changes normalized business mix whenever another activity
has positive initial capacity.  This is the scale-versus-composition bridge for a separable
affected activity; no distance-to-peer sign is asserted. -/
theorem compositionDerivative_singleActivity_ne_zero [DecidableEq α]
    {q : α → ℝ} {affected other : α} {u : ℝ}
    (hS : 0 < ∑ a, q a) (hother : 0 < q other) (hne : other ≠ affected) (hu : u ≠ 0) :
    compositionDerivative q (singleActivityDirection affected u) other ≠ 0 := by
  rw [compositionDerivative, singleActivityDirection_of_ne hne,
    sum_singleActivityDirection]
  exact div_ne_zero
    (sub_ne_zero.mpr (mul_ne_zero (div_ne_zero hother.ne' hS.ne') hu).symm) hS.ne'

/-! ## A rank-one many-firm cost-shock system -/

namespace RankOneCostShock

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The sum of rival response weights, excluding the cost-shock recipient. -/
noncomputable def otherWeight (a c : ι → ℝ) (recipient : ι) : ℝ :=
  ∑ i ∈ Finset.univ.erase recipient, c i / a i

/-- The positive denominator in the explicit inverse of `diag(a) + c 1ᵀ`. -/
noncomputable def responseDenominator (a c : ι → ℝ) (recipient : ι) : ℝ :=
  1 + c recipient / a recipient + otherWeight a c recipient

/-- The sum of all firms' affected-activity responses to a cost cut of size `shock`. -/
noncomputable def aggregateResponse (a c : ι → ℝ) (recipient : ι) (shock : ℝ) : ℝ :=
  (shock / a recipient) / responseDenominator a c recipient

/-- Explicit response to a cost cut at `recipient` in the rank-one system.

The recipient formula and the unshocked-firm formula are written separately so their signs are
visible from positive primitives. -/
noncomputable def response (a c : ι → ℝ) (recipient : ι) (shock : ℝ) (i : ι) : ℝ :=
  if i = recipient then
    (shock / a recipient) * (1 + otherWeight a c recipient) /
      responseDenominator a c recipient
  else
    -(c i / a i) * aggregateResponse a c recipient shock

/-- The local affected-activity system `diag(a) u + c 1ᵀ u = shock e_recipient`. -/
def Solves (a c : ι → ℝ) (recipient : ι) (shock : ℝ) (u : ι → ℝ) : Prop :=
  ∀ i, a i * u i + c i * ∑ k, u k = if i = recipient then shock else 0

/-- Positive coefficients imply a nonnegative sum of excluded rival weights. -/
theorem otherWeight_nonneg {a c : ι → ℝ} {recipient : ι}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) :
    0 ≤ otherWeight a c recipient := by
  exact Finset.sum_nonneg fun i hi ↦ div_nonneg (hc i).le (ha i).le

/-- The explicit rank-one inverse has a strictly positive denominator. -/
theorem responseDenominator_pos {a c : ι → ℝ} {recipient : ι}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) :
    0 < responseDenominator a c recipient := by
  rw [responseDenominator]
  have hcdiv : 0 < c recipient / a recipient := div_pos (hc recipient) (ha recipient)
  have hother := otherWeight_nonneg (recipient := recipient) ha hc
  linarith

/-- A positive cost cut raises total capacity at the affected activity. -/
theorem aggregateResponse_pos {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) (hshock : 0 < shock) :
    0 < aggregateResponse a c recipient shock := by
  exact div_pos (div_pos hshock (ha recipient)) (responseDenominator_pos ha hc)

/-- The full sum of the explicit firm responses equals `aggregateResponse`. -/
theorem sum_response {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) :
    ∑ i, response a c recipient shock i = aggregateResponse a c recipient shock := by
  let W := otherWeight a c recipient
  let D := responseDenominator a c recipient
  let T := aggregateResponse a c recipient shock
  have hrec : response a c recipient shock recipient =
      (shock / a recipient) * (1 + W) / D := by
    simp [response, W, D]
  have hother : ∀ i ∈ Finset.univ.erase recipient,
      response a c recipient shock i = -(c i / a i) * T := by
    intro i hi
    have hir : i ≠ recipient := (Finset.mem_erase.mp hi).1
    simp [response, hir, T]
  calc
    ∑ i, response a c recipient shock i =
        (∑ i ∈ Finset.univ.erase recipient, response a c recipient shock i) +
          response a c recipient shock recipient := by
      rw [Finset.sum_erase_add _ _ (Finset.mem_univ recipient)]
    _ = (∑ i ∈ Finset.univ.erase recipient, -(c i / a i) * T) +
          (shock / a recipient) * (1 + W) / D := by
      rw [hrec]
      congr 1
      exact Finset.sum_congr rfl hother
    _ = -W * T + (shock / a recipient) * (1 + W) / D := by
      rw [← Finset.sum_mul, Finset.sum_neg_distrib]
      change -(∑ i ∈ Finset.univ.erase recipient, c i / a i) * T + _ = _
      rw [show (∑ i ∈ Finset.univ.erase recipient, c i / a i) = W by rfl]
    _ = T := by
      have hD : D ≠ 0 := by
        exact (responseDenominator_pos ha hc).ne'
      dsimp [T, aggregateResponse, D, responseDenominator, W, otherWeight]
      field_simp [hD]
      ring

/-- The explicit response solves the rank-one affected-activity system. -/
theorem response_solves {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) :
    Solves a c recipient shock (response a c recipient shock) := by
  intro i
  rw [sum_response ha hc]
  by_cases hir : i = recipient
  · subst i
    simp only [if_pos, response]
    have ha0 := (ha recipient).ne'
    have hD := (responseDenominator_pos (recipient := recipient) ha hc).ne'
    rw [aggregateResponse]
    field_simp [ha0, hD]
    rw [responseDenominator]
    field_simp [ha0]
    ring
  · simp only [if_neg hir, response]
    rw [aggregateResponse]
    field_simp [(ha i).ne']
    ring

/-- The rank-one system has at most one solution under positive coefficients. -/
theorem solves_unique {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i)
    {u v : ι → ℝ} (hu : Solves a c recipient shock u)
    (hv : Solves a c recipient shock v) : u = v := by
  let U : ℝ := (∑ i, u i) - ∑ i, v i
  have hcoordinate : ∀ i, u i - v i = -(c i / a i) * U := by
    intro i
    have hdiff : a i * (u i - v i) + c i * U = 0 := by
      dsimp [U]
      linear_combination hu i - hv i
    calc
      u i - v i = (-c i * U) / a i := by
        rw [eq_div_iff (ha i).ne']
        nlinarith
      _ = -(c i / a i) * U := by ring
  have hweight : (∑ i, c i / a i) =
      c recipient / a recipient + otherWeight a c recipient := by
    rw [otherWeight]
    have h := Finset.sum_erase_add (s := Finset.univ) (f := fun i ↦ c i / a i)
      (Finset.mem_univ recipient)
    linarith
  have hU : U = -(c recipient / a recipient + otherWeight a c recipient) * U := by
    calc
      U = ∑ i, (u i - v i) := by
        rw [Finset.sum_sub_distrib]
      _ = ∑ i, (-(c i / a i) * U) :=
        Finset.sum_congr rfl fun i _ ↦ hcoordinate i
      _ = -(∑ i, c i / a i) * U := by
        rw [← Finset.sum_mul, Finset.sum_neg_distrib]
      _ = -(c recipient / a recipient + otherWeight a c recipient) * U := by
        rw [hweight]
  have hDU : responseDenominator a c recipient * U = 0 := by
    rw [responseDenominator]
    nlinarith
  have hU0 : U = 0 :=
    (mul_eq_zero.mp hDU).resolve_left (responseDenominator_pos ha hc).ne'
  funext i
  have hi := hcoordinate i
  rw [hU0, mul_zero] at hi
  exact sub_eq_zero.mp hi

/-- The displayed response is the unique solution of the rank-one system. -/
theorem solves_iff_eq_response {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) {u : ι → ℝ} :
    Solves a c recipient shock u ↔ u = response a c recipient shock := by
  constructor
  · intro hu
    exact solves_unique ha hc hu (response_solves ha hc)
  · rintro rfl
    exact response_solves ha hc

/-- An unaffected separable activity has zero response: its homogeneous rank-one system has only
the zero solution. -/
theorem solves_zero_iff_eq_zero {a c : ι → ℝ} {recipient : ι}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) {u : ι → ℝ} :
    Solves a c recipient 0 u ↔ u = 0 := by
  rw [solves_iff_eq_response ha hc]
  constructor
  · intro h
    rw [h]
    funext i
    simp [response, aggregateResponse]
  · intro h
    rw [h]
    funext i
    simp [response, aggregateResponse]

/-- A positive cost cut makes the recipient expand at the affected activity. -/
theorem response_recipient_pos {a c : ι → ℝ} {recipient : ι} {shock : ℝ}
    (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i) (hshock : 0 < shock) :
    0 < response a c recipient shock recipient := by
  rw [response, if_pos rfl]
  exact div_pos
    (mul_pos (div_pos hshock (ha recipient))
      (by linarith [otherWeight_nonneg (recipient := recipient) ha hc]))
    (responseDenominator_pos ha hc)

/-- Every unshocked firm contracts at the affected activity under strict substitution. -/
theorem response_unshocked_neg {a c : ι → ℝ} {recipient i : ι} {shock : ℝ}
    (ha : ∀ k, 0 < a k) (hc : ∀ k, 0 < c k) (hshock : 0 < shock)
    (hir : i ≠ recipient) :
    response a c recipient shock i < 0 := by
  rw [response, if_neg hir]
  exact mul_neg_of_neg_of_pos (neg_neg_of_pos (div_pos (hc i) (ha i)))
    (aggregateResponse_pos ha hc hshock)

/-- The recipient's isolated affected-activity expansion changes its normalized business mix
when another activity has positive initial capacity. -/
theorem recipient_composition_changes {β : Type*} [Fintype β] [DecidableEq β]
    {a c : ι → ℝ} {recipient : ι} {shock : ℝ} {q : β → ℝ}
    {affected other : β} (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i)
    (hshock : 0 < shock) (hS : 0 < ∑ b, q b) (hother : 0 < q other)
    (hne : other ≠ affected) :
    compositionDerivative q
      (singleActivityDirection affected (response a c recipient shock recipient)) other ≠ 0 := by
  apply compositionDerivative_singleActivity_ne_zero hS hother hne
  exact (response_recipient_pos ha hc hshock).ne'

/-- Every unshocked rival's isolated affected-activity contraction changes its normalized
business mix when another activity has positive initial capacity. -/
theorem unshocked_composition_changes {β : Type*} [Fintype β] [DecidableEq β]
    {a c : ι → ℝ} {recipient i : ι} {shock : ℝ} {q : β → ℝ}
    {affected other : β} (ha : ∀ k, 0 < a k) (hc : ∀ k, 0 < c k)
    (hshock : 0 < shock) (hir : i ≠ recipient) (hS : 0 < ∑ b, q b)
    (hother : 0 < q other) (hne : other ≠ affected) :
    compositionDerivative q
      (singleActivityDirection affected (response a c recipient shock i)) other ≠ 0 := by
  apply compositionDerivative_singleActivity_ne_zero hS hother hne
  exact (response_unshocked_neg ha hc hshock hir).ne

/-! ### Primitive coefficients -/

/-- The diagonal part `omega_i + r / D²` of the separable affected-activity Jacobian. -/
noncomputable def primitiveDiagonal (omega : ι → ℝ) (revenue denominator : ℝ) (i : ι) : ℝ :=
  omega i + revenue / denominator ^ 2

/-- The rank-one loading `r (B_i - A_i) / D³` under strategic substitution. -/
noncomputable def primitiveLoading (own rival : ι → ℝ) (revenue denominator : ℝ)
    (i : ι) : ℝ :=
  revenue * (rival i - own i) / denominator ^ 3

omit [Fintype ι] [DecidableEq ι] in
/-- Positive curvature and primitives make every diagonal coefficient positive. -/
theorem primitiveDiagonal_pos {omega : ι → ℝ} {revenue denominator : ℝ}
    (homega : ∀ i, 0 < omega i) (hrevenue : 0 < revenue) (hdenominator : 0 < denominator) :
    ∀ i, 0 < primitiveDiagonal omega revenue denominator i := by
  intro i
  exact add_pos (homega i) (div_pos hrevenue (sq_pos_of_pos hdenominator))

omit [Fintype ι] [DecidableEq ι] in
/-- Strict substitution `A_i < B_i` derives, rather than assumes, positive rank-one loadings. -/
theorem primitiveLoading_pos {own rival : ι → ℝ} {revenue denominator : ℝ}
    (hrevenue : 0 < revenue) (hdenominator : 0 < denominator)
    (hsubstitution : ∀ i, own i < rival i) :
    ∀ i, 0 < primitiveLoading own rival revenue denominator i := by
  intro i
  exact div_pos (mul_pos hrevenue (sub_pos.mpr (hsubstitution i)))
    (pow_pos hdenominator 3)

/-- **Many-firm separable cost-shock comparative static.**  With positive curvature, positive
revenue and denominator, and strict substitution at the affected activity, the explicit unique
local response has a positive recipient coordinate and negative coordinates for every unshocked
firm. -/
@[blueprint "prop:composition-feedback"
  (statement := /--
    Conditional many-firm activity-local cost-response signs support the composition benchmark.
    Assumptions: Positive curvature, opportunity value, denominator and cost reduction, strict
    strategic substitution at the affected activity.
  -/)]
theorem primitive_response_signs {omega own rival : ι → ℝ}
    {revenue denominator shock : ℝ} {recipient : ι}
    (homega : ∀ i, 0 < omega i) (hrevenue : 0 < revenue)
    (hdenominator : 0 < denominator) (hsubstitution : ∀ i, own i < rival i)
    (hshock : 0 < shock) :
    0 < response (primitiveDiagonal omega revenue denominator)
        (primitiveLoading own rival revenue denominator) recipient shock recipient ∧
      ∀ i, i ≠ recipient →
        response (primitiveDiagonal omega revenue denominator)
          (primitiveLoading own rival revenue denominator) recipient shock i < 0 := by
  have ha := primitiveDiagonal_pos homega hrevenue hdenominator
  have hc := primitiveLoading_pos hrevenue hdenominator hsubstitution
  exact ⟨response_recipient_pos ha hc hshock,
    fun i hir ↦ response_unshocked_neg ha hc hshock hir⟩

/-- A three-firm unit-coefficient witness: a unit cost cut gives response `3/4` at the recipient
and `-1/4` at either unshocked rival. -/
theorem threeFirm_unit_response :
    response (fun _ : Fin 3 ↦ 1) (fun _ : Fin 3 ↦ 1) 0 1 =
      fun i ↦ if i = 0 then (3 / 4 : ℝ) else (-1 / 4 : ℝ) := by
  funext i
  fin_cases i <;> norm_num [response, aggregateResponse, responseDenominator, otherWeight]

end RankOneCostShock

end PricingPerspective.Operating.RationalShare.EquilibriumResponse.CompositionFeedback
