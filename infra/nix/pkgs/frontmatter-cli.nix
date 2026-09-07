# marad/frontmatter — small Go CLI for reading a single YAML frontmatter
# field out of a Markdown file (`frontmatter get description FILE`).
# Replaces the `python-frontmatter` PEP-723 dependency in
# scripts/readme_git_tree with a shared, always-on-PATH binary.
# Not in nixpkgs; vendored directly. BUMPING: update `rev`/`hash` from the
# new tag, then `nix build -f <scratch>.nix` to get the new `vendorHash`
# from the mismatch error (see `nix store prefetch-file` for the src hash).
{ buildGoModule, fetchFromGitHub }:
buildGoModule {
  pname = "frontmatter-cli";
  version = "1.1.0";
  src = fetchFromGitHub {
    owner = "marad";
    repo = "frontmatter";
    rev = "450f8ebfa2f429b03383fd756c52961472c3cbbd"; # v1.1.0
    hash = "sha256-ndQw8tJ6EKUPAcNYhv+KTdCIH7cbcAW7ead6LSqErv8=";
  };
  vendorHash = "sha256-MOGrN9yA8p6L5js4dmue0OtLFd5H8tX8T9xAg5c+DMo=";
  meta.mainProgram = "frontmatter";
}
