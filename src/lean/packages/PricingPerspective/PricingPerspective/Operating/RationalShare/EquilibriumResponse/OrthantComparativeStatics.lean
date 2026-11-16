import PricingPerspective.Operating.VariationalInequality.ComparativeStatics
import PricingPerspective.Operating.RationalShare

/-!
# Comparative statics on the uncapped capacity orthant

The rational-share equilibrium is defined on an uncapped nonnegative orthant, while the core
Nash--variational-inequality equivalence and strong-monotonicity theorem are stated on finite
boxes.  This module removes that presentational gap.  For any pair of orthant profiles, an
enlarged finite box contains both profiles and is wide enough for the existing Nash bridge.

The resulting public lemmas compare equilibria of two different rational-share economies.  The
distance bound is evaluated at the old equilibrium and permits simultaneous changes in demand
nodes, linear costs, and quadratic costs.  It assumes a positive curvature modulus for each
economy; the theorem does not itself bound the displayed operator perturbation from primitives.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

variable {κ κ' ι α : Type*} [Fintype κ] [Fintype κ'] [Fintype ι] [Fintype α]

/-- The full nonnegative capacity orthant used by the registered rational-share economy. -/
def capacityOrthant : Set (Profile ι α) := {x | ∀ p, 0 ≤ x p}

/-- Membership in the capacity orthant is coordinatewise nonnegativity. -/
theorem mem_capacityOrthant {x : Profile ι α} :
    x ∈ capacityOrthant ↔ ∀ p, 0 ≤ x p := Iff.rfl

/-- Every orthant Nash equilibrium solves the pseudo-gradient variational inequality on the full
uncapped capacity orthant.

For each test profile, the proof constructs a common finite box containing that profile and the
equilibrium.  A wide-cap witness makes the box Nash-equivalent to the orthant, after which the
existing finite-box Nash--VI equivalence applies. -/
theorem solvesVI_capacityOrthant_of_isNashOrthant
    [DecidableEq ι]
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) {x : Profile ι α}
    (hx : IsNashOrthant N c H x) :
    SolvesVI (pseudoGradient N c H) capacityOrthant x := by
  refine ⟨hx.1, ?_⟩
  intro y hy
  obtain ⟨base, hbase, hbaseCap⟩ := exists_wide_cap (N := N) (ι := ι) hmu
  let hi : ι × α → ℝ := fun p ↦ base p + x p + y p
  have hhi : ∀ p, 0 ≤ hi p := by
    intro p
    dsimp [hi]
    linarith [hbase p, hx.1 p, hy p]
  have hcap : ∀ p, 2 * totalRevenue N ≤ mu * hi p ^ 2 := by
    intro p
    have hle : base p ≤ hi p := by
      dsimp [hi]
      linarith [hx.1 p, hy p]
    have hsq : base p ^ 2 ≤ hi p ^ 2 :=
      (sq_le_sq₀ (hbase p) (hhi p)).2 hle
    exact (hbaseCap p).trans (mul_le_mul_of_nonneg_left hsq hmu.le)
  have hnash : IsNash N c H (fun _ ↦ 0) hi x :=
    (isNash_iff_isNashOrthant N c hmu hc hH hhi hcap).2 hx
  have hvi : SolvesVI (pseudoGradient N c H) (box (fun _ ↦ 0) hi) x :=
    (isNash_iff_solvesVI N c hsymm (psd_of_curvature hmu hH)
      (fun _ ↦ le_refl 0)).1 hnash
  apply hvi.2 y
  intro p
  constructor
  · exact hy p
  · dsimp [hi]
    linarith [hbase p, hx.1 p]

/-- Primitive curvature makes the rational-share pseudo-gradient strongly monotone on the full
uncapped capacity orthant.

The underlying theorem is box-local.  For arbitrary orthant profiles `x` and `y`, the pointwise
maximum of their coordinates supplies a finite box containing both, so the same modulus applies
globally. -/
theorem stronglyMonotoneOn_pseudoGradient_capacityOrthant
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    StronglyMonotoneOn (pseudoGradient N c H) capacityOrthant mu := by
  intro x hx y hy
  let hi : ι × α → ℝ := fun p ↦ max (x p) (y p)
  have hxbox : x ∈ box (fun _ ↦ 0) hi := by
    intro p
    exact ⟨hx p, le_max_left _ _⟩
  have hybox : y ∈ box (fun _ ↦ 0) hi := by
    intro p
    exact ⟨hy p, le_max_right _ _⟩
  exact stronglyMonotoneOn_pseudoGradient N c (lo := fun _ ↦ 0) (hi := hi)
    (fun _ ↦ le_refl 0) hH x hxbox y hybox

/-- Equilibria of two rational-share economies are separated by at most their pseudo-gradient
perturbation at the old equilibrium divided by the new economy's curvature modulus.

Demand nodes, linear costs, and quadratic costs may all differ.  The old economy's positive
curvature is used to turn its orthant Nash equilibrium into a full-orthant VI solution; the new
economy's modulus controls the response distance. -/
theorem orthantEquilibrium_dist_le_operatorPerturbation
    [DecidableEq ι]
    (N : Nodes κ α) (N' : Nodes κ' α)
    (c c' : ι → α → ℝ) {H H' : ι → α → α → ℝ}
    {mu mu' : ℝ} {x y : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hsymm' : ∀ i a b, H' i a b = H' i b a)
    (hmu : 0 < mu) (hmu' : 0 < mu')
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hH' : ∀ i, ∀ u : α → ℝ,
      mu' * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H' i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) (hc' : ∀ i a, 0 ≤ c' i a)
    (hx : IsNashOrthant N c H x) (hy : IsNashOrthant N' c' H' y) :
    ‖y - x‖ ≤
      ‖pseudoGradient N' c' H' x - pseudoGradient N c H x‖ / mu' := by
  have hxvi := solvesVI_capacityOrthant_of_isNashOrthant N c hsymm hmu hH hc hx
  have hyvi := solvesVI_capacityOrthant_of_isNashOrthant N' c' hsymm' hmu' hH' hc' hy
  have hmono := stronglyMonotoneOn_pseudoGradient_capacityOrthant N' c' hH'
  exact response_norm_le hmu' hmono hxvi hyvi

/-- Signed difference between two named firm-activity coordinates. -/
def signedCoordinateDifference (p p' : ι × α) (x : Profile ι α) : ℝ :=
  x p - x p'

/-- A signed coordinate difference is `2`-Lipschitz in the Euclidean profile norm. -/
theorem signedCoordinateDifference_lipschitz (p p' : ι × α)
    (x y : Profile ι α) :
    |signedCoordinateDifference p p' x - signedCoordinateDifference p p' y| ≤
      2 * ‖x - y‖ := by
  have hp : |x p - y p| ≤ ‖x - y‖ := by
    simpa only [Real.norm_eq_abs, PiLp.sub_apply] using PiLp.norm_apply_le (x - y) p
  have hp' : |x p' - y p'| ≤ ‖x - y‖ := by
    simpa only [Real.norm_eq_abs, PiLp.sub_apply] using PiLp.norm_apply_le (x - y) p'
  calc
    |signedCoordinateDifference p p' x - signedCoordinateDifference p p' y|
        = |(x p - y p) - (x p' - y p')| := by
          unfold signedCoordinateDifference
          congr 1
          ring
    _ ≤ |x p - y p| + |x p' - y p'| := by
      simpa [abs_sub_comm (y p') (x p')] using
        abs_sub_le (x p - y p) 0 (x p' - y p')
    _ ≤ ‖x - y‖ + ‖x - y‖ := add_le_add hp hp'
    _ = 2 * ‖x - y‖ := by ring

/-- A positive signed coordinate gap survives any profile perturbation smaller than half its
certified baseline margin. -/
theorem signedCoordinateDifference_pos_of_dist_le
    {x y : Profile ι α} (p p' : ι × α) {delta error : ℝ}
    (hbase : delta ≤ signedCoordinateDifference p p' x)
    (hdist : ‖y - x‖ ≤ error) (hbudget : 2 * error < delta) :
    0 < signedCoordinateDifference p p' y := by
  have hstat := signedCoordinateDifference_lipschitz p p' y x
  have hscaled : 2 * ‖y - x‖ ≤ 2 * error := by nlinarith
  have habs := hstat.trans hscaled
  have hlower := (abs_le.mp habs).1
  linarith

/-- A separately signed equilibrium coordinate gap persists across arbitrary rational-share
primitive changes whenever the evaluated pseudo-gradient perturbation consumes less than half the
baseline margin. -/
theorem orthantSignedCoordinateDifference_pos_of_operatorPerturbation
    [DecidableEq ι]
    (N : Nodes κ α) (N' : Nodes κ' α)
    (c c' : ι → α → ℝ) {H H' : ι → α → α → ℝ}
    {mu mu' delta : ℝ} {x y : Profile ι α} (p p' : ι × α)
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hsymm' : ∀ i a b, H' i a b = H' i b a)
    (hmu : 0 < mu) (hmu' : 0 < mu')
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hH' : ∀ i, ∀ u : α → ℝ,
      mu' * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H' i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) (hc' : ∀ i a, 0 ≤ c' i a)
    (hx : IsNashOrthant N c H x) (hy : IsNashOrthant N' c' H' y)
    (hbase : delta ≤ signedCoordinateDifference p p' x)
    (hbudget :
      2 * (‖pseudoGradient N' c' H' x - pseudoGradient N c H x‖ / mu') < delta) :
    0 < signedCoordinateDifference p p' y := by
  apply signedCoordinateDifference_pos_of_dist_le p p' hbase
    (orthantEquilibrium_dist_le_operatorPerturbation N N' c c' hsymm hsymm'
      hmu hmu' hH hH' hc hc' hx hy)
    hbudget

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
