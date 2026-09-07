import Mathlib.Data.Real.Basic

/-!
# Types

Type aliases for portfolio vectors and factor vectors used throughout the
factor-model formalization.
-/

namespace PaperReconstructions

/-- A vector of portfolio weights indexed by `Fin n`. -/
abbrev ReturnVec (n : ℕ) : Type := Fin n → ℝ

/-- A vector of factor realizations indexed by `Fin k`. -/
abbrev FactorVec (k : ℕ) : Type := Fin k → ℝ

end PaperReconstructions
