# Lean toolchain as a store derivation (GC-safe), replacing imperative `elan`.
# Reuses lean4-nix's `fetchBinaryLean`: it fetches the official prebuilt tarball,
# autoPatchelfs it, swaps in nixpkgs clang/lld, and wraps lean/leanc/lake with
# the C compiler on PATH. Upstream has no v4.31.0 manifest (it tops out at
# v4.30.0, as does nixpkgs' own `lean4`), so the manifest below is inline.
#
# Why a store derivation at all, and the self-contained alternative we did not
# take: ../ARCHITECTURE.md § "Why the Lean toolchain is a store derivation".
#
# BUMPING: when you move the mathlib pin, update `tag` + the per-platform `hash`
#   here IN LOCKSTEP with src/lean/lean-toolchain. Get a hash with:
#     nix store prefetch-file --hash-type sha256 <tarball-url>
{ pkgs, lean4-nix }:
let
  leanManifest = {
    tag = "v4.31.0";
    toolchain = {
      x86_64-linux = {
        url = "https://github.com/leanprover/lean4/releases/download/v4.31.0/lean-4.31.0-linux.tar.zst";
        hash = "sha256-B6YzzI2RUcvAiCXqTN2lDUsCosnLhSwBMbEwRvScrX8=";
      };
      aarch64-linux = {
        url = "https://github.com/leanprover/lean4/releases/download/v4.31.0/lean-4.31.0-linux_aarch64.tar.zst";
        hash = "sha256-sb8dPFhrds9KhiEqWV2Lnt2Z9DikHM6F1XgPqTR8gRs=";
      };
      # macOS: add lean-4.31.0-darwin.tar.zst / lean-4.31.0-darwin_aarch64.tar.zst
      # (+ hashes) here if you ever add a *-darwin system to eachSystem.
    };
  };
in
# `.lean-all` is the bundled toolchain derivation (bin/{lean,lake,leanc} +
# lib/lean/*). We pass `pkgs` explicitly so callPackage always satisfies
# toolchain.nix's `pkgs` formal regardless of the caller's package set.
(pkgs.callPackage (lean4-nix + "/lib/toolchain.nix") { inherit pkgs; }).fetchBinaryLean leanManifest
