import PricingPerspective.Connections.SpatialEnergy

/-!
# Metric-agnostic stability of spatial operators

This module gives the algebraic perturbation results in
`Connections/SpatialEnergy.lean` a metric-neutral public surface.  The original
module remains as a compatibility import for historical Paper 2 and Paper 5
claim manifests; new Wasserstein claims should import this module.
-/

namespace PricingPerspective.Connections.SpatialStability

open Matrix PricingPerspective.Discrete.Spatial
open scoped Matrix.Norms.Operator

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Metric-neutral scalar/matrix perturbation inequality. -/
theorem norm_smul_sub_smul_le {E : Type*} [SeminormedAddCommGroup E]
    [NormedSpace ℝ E] (a b : ℝ) (x y : E) :
    ‖a • x - b • y‖ ≤ |a| * ‖x - y‖ + |a - b| * ‖y‖ :=
  PricingPerspective.Connections.SpatialEnergy.norm_smul_sub_smul_le a b x y

/-- Metric-neutral joint perturbation bound for the SAR Leontief inverse. -/
theorem leontief_sub_norm_le_joint {ρ ρ' : ℝ} {W W' : Matrix ι ι ℝ}
    (h : ‖ρ • W‖ < 1) (h' : ‖ρ' • W'‖ < 1) :
    ‖leontief ρ' W' - leontief ρ W‖ ≤
      (|ρ'| * ‖W' - W‖ + |ρ' - ρ| * ‖W‖) /
        ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) :=
  PricingPerspective.Connections.SpatialEnergy.leontief_sub_norm_le_joint h h'

/-- Metric-neutral entrywise covariance perturbation inequality. -/
theorem spatialCov_entry_bracket_joint {ρ ρ' : ℝ} {W W' V : Matrix ι ι ℝ}
    (i j : ι) :
    |spatialCov ρ' W' V i j - spatialCov ρ W V i j| ≤
      ‖leontief ρ' W' - leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W')ᵀ‖ +
        ‖leontief ρ W‖ * ‖V‖ *
          ‖(leontief ρ' W')ᵀ - (leontief ρ W)ᵀ‖ :=
  PricingPerspective.Connections.SpatialEnergy.spatialCov_entry_bracket_joint i j

/-- An absolute row-sum bound controls the matrix `L∞` operator norm. -/
theorem norm_le_of_abs_row_sum_le {M : Matrix ι ι ℝ} {c : ℝ}
    (hc0 : 0 ≤ c) (hc : ∀ i, ∑ j, |M i j| ≤ c) : ‖M‖ ≤ c :=
  PricingPerspective.Connections.SpatialEnergy.norm_le_of_abs_row_sum_le hc0 hc

end PricingPerspective.Connections.SpatialStability
