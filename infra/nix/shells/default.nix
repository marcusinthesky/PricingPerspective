{ pkgs, customPkgs }:
let
  inherit (pkgs) lib;
  groups = [
    {
      name = "runtime";
      description = "Locked Python execution, data retrieval, and task orchestration.";
      packages = with pkgs; [
        git
        git-lfs
        curl
        cacert
        just
        uv
        (lib.hiPrio python313)
        dvc
      ];
    }
    {
      name = "proofs";
      description = "Lean 4.35 and mathematical dependency graphs.";
      packages = [
        customPkgs.lean-toolchain.lean-all
        pkgs.graphviz
      ];
    }
    {
      name = "papers";
      description = "arXiv manuscript rendering, source bundles, and bibliography projection.";
      packages = with pkgs; [
        tectonic
        pandoc
        poppler-utils
        imagemagick
        zip
        jq
        yq
        ripgrep
        tex-fmt
        customPkgs.chktex
        customPkgs.pgf-metrics
        customPkgs.frontmatter-cli
      ];
    }
    {
      name = "site";
      description = "The retained paper website and static export.";
      packages = [
        customPkgs.bun
        pkgs.nodejs
      ];
    }
    {
      name = "checks";
      description = "Repository hooks and Nix validation.";
      packages = with pkgs; [
        prek
        nixfmt-rfc-style
        statix
        deadnix
      ];
    }
  ];
  runtime = (builtins.elemAt groups 0).packages;
  proofPackages = (builtins.elemAt groups 1).packages;
  paperPackages = (builtins.elemAt groups 2).packages;
  checkPackages = (builtins.elemAt groups 4).packages;
  packageSets = {
    default = lib.concatMap (group: group.packages) groups;
    lean = lib.filter (package: lib.getName package != "dvc") runtime ++ proofPackages;
    prek = runtime ++ checkPackages ++ [ customPkgs.pgf-metrics ];
    replication = runtime ++ proofPackages ++ paperPackages;
  };
in
{
  inherit packageSets;
  devShells = builtins.mapAttrs (
    name: packages:
    pkgs.mkShell {
      name = "pricing-perspective-${name}";
      inherit packages;
      shellHook = import ./env.nix { inherit pkgs; };
    }
  ) packageSets;
  inventory = map (group: {
    inherit (group) name description;
    tools = map (package: package.meta.mainProgram or (lib.getName package)) group.packages;
  }) groups;
}
