# j-clemons/dbt-language-server — Go stdio LSP for dbt. Parses the project
# statically: no manifest.json, no live connection, no warehouse. Navigation
# only, so it complements rather than replaces the sqlfluff/sqlfmt gates.
# Not in nixpkgs.
# BUMPING: update `rev`/`hash`, then `nix build -f <scratch>.nix` to read the new
#   `vendorHash` from the mismatch error (`nix store prefetch-file` for src).
{ buildGoModule, fetchFromGitHub }:
buildGoModule {
  pname = "dbt-language-server";
  version = "0-unstable-2026-06-03";
  src = fetchFromGitHub {
    owner = "j-clemons";
    repo = "dbt-language-server";
    rev = "aa91551d6ffe3f6a186600d1c762c773ffe614ad";
    hash = "sha256-a6yhrhFMiqXj50b0707426GHJZDXpoIuPQgvvSzMy5E=";
  };
  vendorHash = "sha256-g+yaVIx4jxpAQ/+WrGKxhVeliYx7nLQe/zsGpxV4Fn4=";
  # This upstream test compares a slice assembled from Go maps without sorting
  # it. The members are stable, but their iteration order is not: CI observed
  # the same three completion items in a different order and rejected the whole
  # dev-shell closure. Keep the rest of the upstream suite enabled.
  checkFlags = [ "-skip=^TestGetMacroCompletionItems$" ];
  # Upstream ships no main-program metadata; the binary is the module basename.
  meta.mainProgram = "dbt-language-server";
  # Upstream logs next to the running binary, which under Nix is a read-only
  # store path — the server panics with "Did not provide a good file" before it
  # ever speaks LSP. Redirect it to a temp dir.
  postPatch = ''
            substituteInPlace util/logging.go \
              --replace-fail \
                'func executableDirectory() string {' \
                'func executableDirectory() string {
    return os.TempDir()
    // unreachable below, kept so the original body still parses'
  '';
}
