{ inputs, pkgs, ... }:
let
  packages = inputs.pricing-flake.shellPackages.${pkgs.stdenv.hostPlatform.system};
  shellEnvironment = import ./infra/nix/shells/env.nix { inherit pkgs; };
in
{
  dotenv.disableHint = true;
  profiles = {
    dev.module = {
      packages = packages.default;
      enterShell = shellEnvironment;
    };
    prek.module = {
      packages = packages.prek;
      enterShell = shellEnvironment;
    };
    lean.module = {
      packages = packages.lean;
      enterShell = shellEnvironment;
    };
    replication.module = {
      packages = packages.replication;
      enterShell = shellEnvironment;
    };
    watch = {
      extends = [ "dev" ];
      module.processes.insitu.exec = "just insitu::watch";
    };
  };
}
