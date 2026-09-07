import PaperReconstructions.Ross1976.Types
import Mathlib.Data.Finset.Defs
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic

/-!
# Types

Type aliases for portfolio vectors, factor vectors, and continuous model
structures used throughout the PricingPerspective formalization.

`ReturnVec` and `FactorVec` are re-exported from `PaperReconstructions.Ross1976.Types`.
`ExposureMeasure` and `RandomField` are unique to PricingPerspective.
-/

namespace PricingPerspective

open Finset

/-- A vector of portfolio weights indexed by `Fin n`. -/
abbrev ReturnVec (n : ℕ) : Type := PaperReconstructions.ReturnVec n

/-- A vector of factor realizations indexed by `Fin k`. -/
abbrev FactorVec (k : ℕ) : Type := PaperReconstructions.FactorVec k

/-- A measure over exposure indices (for continuous factor models). -/
abbrev ExposureMeasure : Type := MeasureTheory.Measure ℝ

/-- A random field mapping exposures to real-valued outcomes. -/
abbrev RandomField : Type := ℝ → ℝ

end PricingPerspective
