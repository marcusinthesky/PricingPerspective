# Packages this flake vendors because nixpkgs has no usable attribute for them,
# one file each. See ./README.md for what earns a file here; each file owns its
# own BUMPING runbook.
#
# CALL CONVENTION — `callPackage` when a file *constructs* a derivation; plain
# `import` when it *selects* an existing package or returns a set, because
# callPackage's makeOverridable wrapper would bolt an `override` onto an attrset
# with nothing to override (lean-toolchain).
{
  pkgs,
  inputs,
}:
{
  bun = pkgs.callPackage ./bun.nix { };
  chktex = pkgs.callPackage ./chktex.nix { };
  dbt-language-server = pkgs.callPackage ./dbt-language-server.nix { };
  frontmatter-cli = pkgs.callPackage ./frontmatter-cli.nix { };
  latexdiff = pkgs.callPackage ./latexdiff.nix { };
  merman-cli = pkgs.callPackage ./merman-cli.nix { };
  pdf-inspector = pkgs.callPackage ./pdf-inspector.nix { };
  pgf-metrics = pkgs.callPackage ./pgf-metrics.nix { };
  schematter = pkgs.callPackage ./schematter.nix { };
  semble-rs = pkgs.callPackage ./semble-rs.nix { };

  lean-toolchain = import ./lean-toolchain.nix {
    inherit pkgs;
    inherit (inputs) lean4-nix;
  };
}
