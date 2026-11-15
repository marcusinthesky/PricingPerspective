import Lake
open Lake DSL

package EnergyStatistics where

require proofs from git
  "https://github.com/marcusinthesky/statadic.git" @
  "213a32399705af3463a3c1d8835409f284025ca1" / "src/lean"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.35.0"

@[default_target]
lean_lib EnergyStatistics where
  roots := #[`EnergyStatistics]
