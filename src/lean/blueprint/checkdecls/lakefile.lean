import Lake
open Lake DSL

package «pricing-perspective-blueprint-checkdecls» where
  -- Reuse the parent workspace's pinned dependency cache rather than cloning a
  -- second mathlib checkout for this tiny declaration-checker adapter.
  packagesDir := "../../.lake/packages"

require checkdecls from git
  "https://github.com/PatrickMassot/checkdecls.git" @
  "3d425859e73fcfbef85b9638c2a91708ef4a22d4"

require LeanArchitect from git
  "https://github.com/hanwenzhu/LeanArchitect" @
  "9f3bb53916b9d6fa05b7777389e1844f82b1c9a6"

require «PricingPerspective» from "../../packages/PricingPerspective"

@[default_target]
lean_lib BlueprintPublication where
  -- The checker imports library roots. Reuse the complete native assembly,
  -- including approved operating claims outside PricingPerspective's root.
  srcDir := "../.."
  roots := #[`Publication]
