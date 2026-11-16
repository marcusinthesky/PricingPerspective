{
  description = "Pricing Perspective paper reproduction and website toolchain";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
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
        customPkgs = import ./pkgs { inherit pkgs inputs; };
        shells = import ./shells { inherit pkgs customPkgs; };
      in
      {
        packages = customPkgs // {
          lean-toolchain = customPkgs.lean-toolchain.lean-all;
        };
        inherit (shells) devShells;
        shellPackages = shells.packageSets;
        toolInventory = shells.inventory;
      }
    );
}
