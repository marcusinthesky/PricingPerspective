import Lake
open Lake DSL

package WassersteinGeometry where

require LeanArchitect from git
  "https://github.com/hanwenzhu/LeanArchitect" @
  "9f3bb53916b9d6fa05b7777389e1844f82b1c9a6"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.35.0"

@[default_target]
lean_lib WassersteinGeometry where
  roots := #[`WassersteinGeometry]
  globs := #[.andSubmodules `WassersteinGeometry]
