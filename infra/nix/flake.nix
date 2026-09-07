{
  description = "Pricing Perspective: distributional model of asset pricing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Lean toolchain provider. Pinned as a *source* input (flake = false): we only
    # consume its `lib/toolchain.nix` helper, not its flake outputs, so we avoid
    # pulling in its nixpkgs/flake-parts inputs. See pkgs/lean-toolchain.nix.
    lean4-nix = {
      url = "github:lenianiva/lean4-nix";
      flake = false;
    };

  };

  outputs =
    inputs@{ nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        inherit (nixpkgs) lib;

        # `inputs` is passed whole because lean-toolchain is built from flake
        # inputs, not `pkgs`.
        customPkgs = import ./pkgs { inherit pkgs inputs; };

        shells = import ./shells { inherit pkgs lib customPkgs; };
      in
      {
        # One build handle per vendored derivation, so `nix build .#semble-rs`
        # fails on the package that broke rather than midway through realizing a
        # shell. lean-toolchain is an attrset, so its `.lean-all` member is what
        # is exposed.
        packages = {
          inherit (customPkgs)
            bun
            chktex
            dbt-language-server
            frontmatter-cli
            latexdiff
            merman-cli
            pdf-inspector
            pgf-metrics
            schematter
            semble-rs
            ;
          lean-toolchain = customPkgs.lean-toolchain.lean-all;
        };

        # `default` composes ./shells/groups/*.nix; the other shells are
        # workload-minimal CI slices under ./shells/ci.
        devShells = {
          inherit (shells)
            default
            lean
            prek
            replication
            ;
        };

        # Not a devShell — the ordered group inventory that Insitu renders into
        # README.md's tool_inventory region.
        toolInventory = shells.inventory;
      }
    );
}
