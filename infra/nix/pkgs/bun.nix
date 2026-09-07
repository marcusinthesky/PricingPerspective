# Bun 1.4.0 — the TypeScript workspace's pinned runtime and package manager.
# nixpkgs currently provides Bun 1.3.x, while the workspace contract is 1.4.0.
# BUMPING: update `version` and both official release hashes together, then run
# `nix build ./infra/nix#bun` and `bun --version` inside the resulting shell.
{
  autoPatchelfHook,
  fetchurl,
  lib,
  openssl,
  stdenvNoCC,
  unzip,
}:
stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "bun";
  version = "1.4.0";

  src =
    {
      "aarch64-linux" = fetchurl {
        url = "https://github.com/oven-sh/bun/releases/download/bun-v${finalAttrs.version}/bun-linux-aarch64.zip";
        hash = "sha256-SxozLuhhmD65O8/m93D/+U4+MbLDiL2uo8jtNeWO7Q4=";
      };
      "x86_64-linux" = fetchurl {
        url = "https://github.com/oven-sh/bun/releases/download/bun-v${finalAttrs.version}/bun-linux-x64-baseline.zip";
        hash = "sha256-GE+0WV8NQBohfPfHjBvEMLqDMU2reouUgFurv3+nCX8=";
      };
    }
    .${stdenvNoCC.hostPlatform.system}
      or (throw "Unsupported system: ${stdenvNoCC.hostPlatform.system}");

  nativeBuildInputs = [
    autoPatchelfHook
    unzip
  ];
  buildInputs = [ openssl ];
  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    install -Dm755 bun $out/bin/bun
    ln -s $out/bin/bun $out/bin/bunx
  '';

  meta = {
    description = "Fast JavaScript runtime, package manager, bundler, and test runner";
    homepage = "https://bun.sh";
    license = with lib.licenses; [
      mit
      lgpl21Only
    ];
    mainProgram = "bun";
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
  };
})
