# Latias94/merman — browserless Rust renderer for Mermaid diagrams. DVC emits
# Mermaid natively, so this replaces the retired Typst plugin and the ad-hoc DOT
# path for the root pipeline.png without introducing Node.js or Chromium.
# Not in nixpkgs.
#
# BUMPING: update `version` and the crate `hash`, set `cargoHash` to
#   `lib.fakeHash`, then run `nix build path:./infra/nix#merman-cli` once and
#   paste the reported dependency hash. Verify `just pipeline-graph` twice and
#   compare the resulting pipeline.png hashes.
{
  fetchCrate,
  lib,
  rustPlatform,
}:
rustPlatform.buildRustPackage rec {
  pname = "merman-cli";
  version = "0.8.0-alpha.5";

  src = fetchCrate {
    inherit pname version;
    hash = "sha256-2i3/hIB2aiLFulT93hiDPu8oBNx+E7wb6Epm30vZsyc=";
  };

  cargoHash = "sha256-upWdjCdZMQNoTpt0HnIe9S6gbRDi1eMnnMQgMROETac=";

  # The repository uses the native SVG pipeline plus PNG export only. Avoid
  # pulling Merman's PDF, JPEG, Markdown, network-icon, and terminal surfaces
  # into every development shell.
  buildNoDefaultFeatures = true;
  buildFeatures = [ "png" ];

  meta = {
    description = "Headless Rust renderer for Mermaid diagrams";
    homepage = "https://github.com/Latias94/merman";
    license = with lib.licenses; [
      asl20
      mit
    ];
    mainProgram = "merman-cli";
  };
}
