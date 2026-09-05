{
  description = "Pricing Perspective: distributional model of asset pricing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Separately-pinned nixpkgs used ONLY for the duckdb CLI, so it can track the
    # version src/python/apps/harnessme/extensions.lock.yml requires (extension binaries are
    # version-keyed) without moving the rest of the shell. See ARCHITECTURE.md
    # § "Why the SQL toolchain is built from uv.lock".
    nixpkgs-duckdb.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Lean toolchain provider. Pinned as a *source* input (flake = false): we only
    # consume its `lib/toolchain.nix` helper, not its flake outputs, so we avoid
    # pulling in its nixpkgs/flake-parts inputs. See pkgs/lean-toolchain.nix.
    lean4-nix = {
      url = "github:lenianiva/lean4-nix";
      flake = false;
    };

    # uv2nix: builds the dbt/duckdb SQL toolchain as a hermetic Python env from
    # the harnessme member in src/python/uv.lock. See pkgs/dbt-with-duckdb.nix.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    # The Python uv workspace as a source input — only git-tracked files are
    # copied, so .venv / dbt_packages / target are excluded. Editing the shared
    # src/python/uv.lock re-locks this input automatically.
    python-workspace = {
      url = "path:../../src/python";
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

        # `inputs` is passed whole because three of these (duckdb-cli,
        # dbt-with-duckdb, lean-toolchain) are built from flake inputs, not `pkgs`.
        customPkgs = import ./pkgs {
          inherit
            pkgs
            lib
            inputs
            system
            ;
        };

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
            dbt-with-duckdb
            duckdb-cli
            frontmatter-cli
            latexdiff
            merman-cli
            oup-template
            pdf-inspector
            pgf-metrics
            schematter
            semble-rs
            tandf-template
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
