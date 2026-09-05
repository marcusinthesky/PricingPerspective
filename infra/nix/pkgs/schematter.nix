# iwe-org/schematter — document-schema validator for Markdown frontmatter and body shape.
# The repository uses it only for authored plan/task README bodies; contracts owns
# frontmatter and graph semantics. The release asset is pinned because schematter
# is not in nixpkgs and compiling it in every shell would add a large Rust closure.
# BUMPING: update `version`, the release URL, and the fetch hash from the matching
# release asset; verify `nix build path:./infra/nix#schematter` and `schematter --version`.
{
  fetchurl,
  stdenvNoCC,
}:
stdenvNoCC.mkDerivation {
  pname = "schematter";
  version = "0.1.0";
  src = fetchurl {
    url = "https://github.com/iwe-org/schematter/releases/download/schematter-v0.1.0/schematter-v0.1.0-x86_64-unknown-linux-gnu.tar.gz";
    hash = "sha256-hSEglws2SiqFQh9koaMQZUi0nKAI88QwmzCy5N6hLYM=";
  };
  dontBuild = true;
  unpackPhase = ''
    tar -xzf "$src"
  '';
  installPhase = ''
    install -Dm755 schematter "$out/bin/schematter"
  '';
  meta.mainProgram = "schematter";
}
