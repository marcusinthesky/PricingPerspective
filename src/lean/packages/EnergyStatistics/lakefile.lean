import Lake
open Lake DSL

package EnergyStatistics where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.31.0"

@[default_target]
lean_lib EnergyStatistics where
  roots := #[`EnergyStatistics]
