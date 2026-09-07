# The flake's `devShells`. `default` is assembled from ./groups; CI shells stand
# alone under ./ci so an unrelated interactive-tool addition cannot grow either
# runner closure.
{
  pkgs,
  lib,
  customPkgs,
}:
let
  # One set carrying both nixpkgs and ../pkgs, so a group file can just name
  # packages. A shallow merge at this call site, NOT an overlay — nothing inside
  # nixpkgs resolves through it. `chktex` is the one deliberate collision.
  scope = pkgs // customPkgs;

  # The group's name is its filename, by construction — there is no second place
  # to keep them in step.
  group = name: { inherit name; } // import (./groups + "/${name}.nix") { pkgs = scope; };

  # ORDER IS LOAD-BEARING: mkShell hashes `nativeBuildInputs` in list order, and
  # the order decides which package wins on PATH when two ship the same binary
  # (see the chktex/pgf-metrics note in groups/languages.nix). Add a package
  # inside the group that owns it; do not append at the end.
  groups = map group [
    "shell"
    "tools"
    "languages"
    "search"
    "documents"
    "diagnostics"
    "cloud"
  ];
in
{
  default = pkgs.mkShell {
    name = "pricing-perspective-dev";
    packages = lib.concatMap (g: g.packages) groups;
    # `scope`, not `pkgs`: env.nix points cargo at the vendored DuckDB, which
    # lives in ../pkgs and is deliberately not the main nixpkgs pin.
    shellHook = import ./env.nix { pkgs = scope; };
  };

  lean = import ./ci/lean.nix { pkgs = scope; };
  prek = import ./ci/prek.nix { pkgs = scope; };
  replication = import ./ci/replication.nix { pkgs = scope; };

  # Ordered inventory for the generated table in ../README.md (`just nix::tools`).
  # Nothing in the shell reads it — it exists so that table is derived from this
  # list rather than maintained beside it. `mainProgram` first so tools render as
  # the command you type (`tofu`, `rg`) rather than the attribute name.
  inventory = map (g: {
    inherit (g) name description;
    tools = map (p: p.meta.mainProgram or (lib.getName p)) g.packages;
  }) groups;
}
