# Descriptor convention: see groups/shell.nix.
{ pkgs }:
{
  description = "Cloud provider CLI: ADC credentials for the OpenTofu google provider, and brownfield resource export.";
  packages = with pkgs; [
    # Separate from the HCL section of languages.nix: it is the one part of the
    # infrastructure toolchain that is not a language tool.
    google-cloud-sdk # `gcloud`: ADC auth for the google provider, and `beta resource-config bulk-export`
  ];
}
