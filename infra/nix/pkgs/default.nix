{ pkgs, inputs }:
{
  bun = pkgs.callPackage ./bun.nix { };
  chktex = pkgs.callPackage ./chktex.nix { };
  pgf-metrics = pkgs.callPackage ./pgf-metrics.nix { };
  frontmatter-cli = pkgs.callPackage ./frontmatter-cli.nix { };
  tandf-template = pkgs.callPackage ./tandf-template.nix { };
  elsarticle-template = pkgs.callPackage ./elsarticle-template.nix { };
  lean-toolchain = import ./lean-toolchain.nix {
    inherit pkgs;
    inherit (inputs) lean4-nix;
  };
}
