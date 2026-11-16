import Architect
import PricingPerspective.Operating.RationalShare.SolverCertificates.EquilibriumRegretCertificate

/-!
# Independent-firm equilibrium certificates

The moving-opportunity experiment compares the strategic rational-share game with an
independent-firm counterfactual.  In that counterfactual every firm faces only the strictly
positive outside field: other firms' appeal is removed from its denominator.  This is a
Cartesian product of one-firm rational-share problems, not the multi-firm strategic game.

This module makes that product structure explicit.  Its joint field is exactly the Python
independent-arm operator.  The one-firm response theorems then supply blockwise optimality,
while summing their variational inequalities and strong-monotonicity bounds supplies the joint
equilibrium-distance certificate.  The block residual separately supplies each firm's payoff-
regret certificate.

All statements use exact real arithmetic.  They do not certify a floating-point residual
evaluation.
-/

namespace PricingPerspective.Operating.RationalShare.SolverCertificates.IndependentEquilibriumCertificate

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse
  PricingPerspective.Operating.RationalShare.FirmResponse.Response
  PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse
  PricingPerspective.Operating.RationalShare.EquilibriumResponse
  PricingPerspective.Operating.RationalShare.SolverCertificates.ProjectedResidual

open PricingPerspective.Operating.RationalShare.SolverCertificates.EquilibriumRegretCertificate

open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]
  [DecidableEq ι] [DecidableEq α]

/-- The independent-firm joint pseudo-gradient.

Each firm faces the original node revenues and kernels but only the outside field `N.outside`.
The zero proximal weight and centre make this exactly the unpenalized one-firm operator used by
the Python independent arm. -/
noncomputable def pseudoGradient
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) : Profile ι α :=
  WithLp.toLp 2 fun p ↦
    Response.blockGradient N N.outside N.outside_pos (c p.1) (H p.1) 0 0
      (IncumbentResponse.incumbentBlock x p.1) ((), p.2)

omit [DecidableEq ι] in
/-- Coordinate formula for the independent arm's Python operator.

The rival field is exactly outside appeal.  No other firm's block occurs on the right-hand side.
-/
@[simp]
theorem pseudoGradient_apply
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) (i : ι) (a : α) :
    pseudoGradient N c H x (i, a) =
      (c i a + ∑ b, H i a b * x (i, b))
        - ∑ l, N.revenue l * N.outside l /
            (N.outside l + N.appeal x i l) ^ 2 * N.kernel l a := by
  rw [pseudoGradient, Response.blockGradient_apply]
  simp only [zero_mul, add_zero, IncumbentResponse.incumbentBlock_apply]
  congr 2

omit [DecidableEq ι] in
/-- Extracting one firm from the joint independent field gives its frozen-outside block gradient
exactly.  This is the theorem-to-code bridge used by both the joint and block certificates. -/
theorem firmPseudoGradientAt_eq_blockGradient
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) (i : ι) :
    firmPseudoGradientAt (pseudoGradient N c H) x i =
      Response.blockGradient N N.outside N.outside_pos (c i) (H i) 0 0
        (IncumbentResponse.incumbentBlock x i) := by
  ext ⟨u, a⟩
  cases u
  rfl

/-- Firm `i`'s independent payoff, represented as the payoff of a one-firm rational-share game
whose outside option is the original outside field.  Other blocks of `x` are absent by
construction. -/
noncomputable def independentPayoff
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (i : ι) (x : Profile ι α) : ℝ :=
  RationalShare.payoff (Response.frozenNodes N N.outside N.outside_pos)
    (fun _ ↦ c i) (fun _ ↦ H i) () (IncumbentResponse.incumbentBlock x i)

omit [DecidableEq ι] in
/-- The singleton representation is exactly the unpenalized frozen-outside block objective used
to define the independent-firm counterfactual. -/
theorem independentPayoff_eq_blockPayoff
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (i : ι) (x : Profile ι α) :
    independentPayoff N c H i x =
      Response.blockPayoff N N.outside (c i) (H i) 0 0
        (IncumbentResponse.incumbentBlock x i) := by
  have hc : Response.effectiveCost (c i) 0 0 = fun _ ↦ c i := by
    funext u a
    simp [Response.effectiveCost]
  have hH : Response.effectiveCurvature (H i) 0 = fun _ ↦ H i := by
    funext u a b
    simp [Response.effectiveCurvature]
  symm
  simpa [independentPayoff, hc, hH]
    using Response.blockPayoff_eq_payoff_sub_centerConst
      N N.outside N.outside_pos (c i) (H i) 0 (0 : Response.Block α)
        (IncumbentResponse.incumbentBlock x i)

omit [DecidableEq α] in
/-- Replacing a joint profile's `i` block replaces the associated one-firm profile by the same
activity function. -/
theorem incumbentBlock_replaceFirm_self (x : Profile ι α) (i : ι) (q : α → ℝ) :
    IncumbentResponse.incumbentBlock (replaceFirm x i q) i = Response.ofActivity q := by
  ext ⟨u, a⟩
  cases u
  simp only [IncumbentResponse.incumbentBlock_apply, replaceFirm_self]
  rfl

omit [DecidableEq ι] in
/-- The joint independent block residual is identical to the residual obtained by viewing that
block as a one-firm frozen-outside rational-share game. -/
theorem firmStationarityResidual_eq_singleton
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) (i : ι) :
    firmStationarityResidual (pseudoGradient N c H) x i =
      firmStationarityResidual
        (RationalShare.pseudoGradient
          (Response.frozenNodes N N.outside N.outside_pos)
          (fun _ ↦ c i) (fun _ ↦ H i))
        (IncumbentResponse.incumbentBlock x i) () := by
  ext ⟨u, a⟩
  cases u
  rw [firmStationarityResidual_apply, firmStationarityResidual_apply]
  simp only [IncumbentResponse.incumbentBlock_apply]
  rw [pseudoGradient_apply, RationalShare.pseudoGradient_apply]
  simp only [Response.frozenNodes_rival, Response.frozenNodes_denom,
    Response.frozenNodes_kernel, Response.frozenNodes_revenue,
    IncumbentResponse.incumbentBlock_apply]
  have hactivity : Response.activity (IncumbentResponse.incumbentBlock x i) =
      fun b ↦ x (i, b) := rfl
  rw [hactivity]
  rfl

/-- An independent-firm equilibrium is a profile whose every block globally maximizes the
one-firm rational-share payoff against outside appeal alone. -/
def IsEquilibrium
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) : Prop :=
  x ∈ capacityOrthant ∧ ∀ i,
    Response.IsExactResponse N N.outside (c i) (H i) 0 0
      (IncumbentResponse.incumbentBlock x i)

omit [DecidableEq ι] [DecidableEq α] in
/-- Every feasible joint profile gives every firm a feasible one-firm block. -/
theorem incumbentBlock_mem_orthant {x : Profile ι α} (hx : x ∈ capacityOrthant) (i : ι) :
    IncumbentResponse.incumbentBlock x i ∈ Response.orthant :=
  IncumbentResponse.incumbentBlock_mem_orthant hx i

omit [DecidableEq ι] in
/-- The Cartesian product of the independent one-firm fields is strongly monotone with the
common own-cost modulus.  No cross-firm term is needed: the joint pairing is the sum of the
one-firm pairings. -/
theorem stronglyMonotoneOn_pseudoGradient
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    StronglyMonotoneOn (pseudoGradient N c H) capacityOrthant mu := by
  intro x hx y hy
  rw [RationalShare.norm_sq_eq_sum, RationalShare.inner_eq_sum,
    Fintype.sum_prod_type, Fintype.sum_prod_type, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  have hi := Response.stronglyMonotoneOn_blockGradient
    N N.outside N.outside_pos (c i) (kappa := 0) (0 : Response.Block α) (hH i)
    (IncumbentResponse.incumbentBlock x i) (incumbentBlock_mem_orthant hx i)
    (IncumbentResponse.incumbentBlock y i) (incumbentBlock_mem_orthant hy i)
  rw [RationalShare.norm_sq_eq_sum, RationalShare.inner_eq_sum,
    Fintype.sum_prod_type, Fintype.sum_prod_type] at hi
  simp only [Fintype.sum_unique] at hi
  simpa only [add_zero, PiLp.sub_apply, pseudoGradient,
    IncumbentResponse.incumbentBlock,
    Response.ofActivity,
    show (default : Unit) = () from Subsingleton.elim _ _] using hi

omit [DecidableEq ι] in
/-- A profile of exact independent one-firm responses solves the joint product variational
inequality.  Each firm's block inequality is summed once; no strategic-game Nash bridge is used.
-/
theorem solvesVI_of_isEquilibrium
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    {x : Profile ι α} (hx : IsEquilibrium N c H x) :
    SolvesVI (pseudoGradient N c H) capacityOrthant x := by
  refine ⟨hx.1, fun z hz ↦ ?_⟩
  rw [RationalShare.inner_eq_sum, Fintype.sum_prod_type]
  apply Finset.sum_nonneg
  intro i _
  have hpsd : ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b := fun u ↦
    (mul_nonneg hmu.le (Finset.sum_nonneg fun a _ ↦ sq_nonneg (u a))).trans (hH i u)
  have hvi := (Response.isExactResponse_iff_solvesVI
    N N.outside N.outside_pos (c i) (0 : Response.Block α)
      (IncumbentResponse.incumbentBlock x i) (hsymm i) hpsd (le_refl 0)).mp (hx.2 i)
  have hi := hvi.2 (IncumbentResponse.incumbentBlock z i) (incumbentBlock_mem_orthant hz i)
  rw [RationalShare.inner_eq_sum, Fintype.sum_prod_type] at hi
  simp only [Fintype.sum_unique] at hi
  simpa only [PiLp.sub_apply, pseudoGradient, IncumbentResponse.incumbentBlock,
    Response.ofActivity,
    show (default : Unit) = () from Subsingleton.elim _ _] using hi

omit [DecidableEq ι] in
/-- The product of the one-firm frozen-outside problems has a unique exact equilibrium whenever
every own-cost block has a common positive curvature modulus.

Existence is assembled from the verified one-firm maximizers.  Uniqueness follows from the
product strong-monotonicity theorem and the joint variational-inequality bridge. -/
@[blueprint "app:independent-proof"
  (statement := /--
    Unique equilibrium of the independent product economy. Assumptions: Nonnegative one-firm
    rational-share appeals and positive cost curvature.
  -/)]
theorem existsUnique_isEquilibrium
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    ∃! x, IsEquilibrium N c H x := by
  classical
  have hex : ∀ i, ∃ q,
      Response.IsExactResponse N N.outside (c i) (H i) 0 0 q := fun i ↦
    (Response.existsUnique_isExactResponse N N.outside N.outside_pos (c i)
      (0 : Response.Block α) (hsymm i) hmu (hH i) (le_refl 0)).exists
  choose q hq using hex
  let x : Profile ι α := WithLp.toLp 2 fun p ↦ q p.1 ((), p.2)
  have hxblock (i : ι) : IncumbentResponse.incumbentBlock x i = q i := by
    ext ⟨u, a⟩
    cases u
    rfl
  have hx : IsEquilibrium N c H x := by
    refine ⟨fun p ↦ ?_, fun i ↦ ?_⟩
    · exact (hq p.1).1 ((), p.2)
    · rw [hxblock]
      exact hq i
  refine ⟨x, hx, fun y hy ↦ ?_⟩
  exact eq_of_solvesVI hmu (stronglyMonotoneOn_pseudoGradient N c hH)
    (solvesVI_of_isEquilibrium N c hsymm hmu hH (x := y) hy)
    (solvesVI_of_isEquilibrium N c hsymm hmu hH (x := x) hx)

omit [DecidableEq ι] in
/-- The joint stationarity residual of a feasible computed independent profile, divided by the
common own-cost curvature, bounds its distance from an exact independent equilibrium. -/
@[blueprint "app:independent-distance"
  (statement := /--
    The independent product residual bounds joint equilibrium error. Assumptions: Independent
    one-firm rational-share objectives, feasibility, and positive cost curvature.
  -/)]
theorem dist_equilibrium_le_equilibriumStationarityResidual
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {mu : ℝ} {x y : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : x ∈ capacityOrthant) (hy : IsEquilibrium N c H y) :
    ‖x - y‖ ≤ ‖equilibriumStationarityResidual (pseudoGradient N c H) x‖ / mu := by
  exact dist_exactVI_le_equilibriumStationarityResidual hmu
    (stronglyMonotoneOn_pseudoGradient N c hH) hx
    (solvesVI_of_isEquilibrium N c hsymm hmu hH hy)

/-- Every feasible unilateral independent-payoff gain is bounded by the firm's squared block
stationarity residual divided by twice its own-cost curvature.

The proof applies the verified one-firm rational-share regret theorem to the frozen-outside game
and then uses `firmStationarityResidual_eq_singleton` to identify its residual with the exact
block of the Python joint independent operator. -/
@[blueprint "app:independent-regret"
  (statement := /--
    Independent firm residual bounds unilateral regret. Assumptions: Independent one-firm
    rational-share objectives, feasibility, and positive cost curvature.
  -/)]
theorem unilateralPayoffGain_le_sq_norm_firmStationarityResidual_div
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    (i : ι) (hsymm : ∀ a b, H i a b = H i b a) {mu : ℝ} (hmu : 0 < mu)
    {x : Profile ι α} (hx : x ∈ capacityOrthant) (q : α → ℝ)
    (hq : ∀ a, 0 ≤ q a)
    (hH : ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    independentPayoff N c H i (replaceFirm x i q) - independentPayoff N c H i x ≤
      ‖firmStationarityResidual (pseudoGradient N c H) x i‖ ^ 2 / (2 * mu) := by
  let NF := Response.frozenNodes N N.outside N.outside_pos
  let xb := IncumbentResponse.incumbentBlock x i
  have hxb : xb ∈ (capacityOrthant : Set (Profile Unit α)) := by
    exact incumbentBlock_mem_orthant hx i
  have hbase :=
    EquilibriumRegretCertificate.unilateralPayoffGain_le_sq_norm_firmStationarityResidual_div
      NF (fun _ ↦ c i) (H := fun _ ↦ H i) (fun _ ↦ hsymm) hmu hxb () q hq hH
  rw [← firmStationarityResidual_eq_singleton N c H x i] at hbase
  simpa only [independentPayoff, NF, xb, incumbentBlock_replaceFirm_self,
    Response.replaceFirm_singleton] using hbase

end PricingPerspective.Operating.RationalShare.SolverCertificates.IndependentEquilibriumCertificate
