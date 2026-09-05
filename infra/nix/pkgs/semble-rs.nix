# semble-rs (johunsang/semble_rs) — hybrid BM25 + Model2Vec code search over
# tree-sitter chunks. CPU-only, no cloud API, pure-Rust TLS. Not in nixpkgs.
# Where it sits against ast-grep, rg, and the DuckDB harness:
# ../ARCHITECTURE.md § "Why several ways to search code?".
# MODEL PIN: upstream defaults to potion-code-16M; this repo wants the successor
#   potion-code-16M-v2. That preference is enforced here rather than in a justfile
#   or a doc, because neither reaches someone typing `semble_rs` directly — they
#   would silently get the older model, with no error and merely worse results.
#   `--model` overrides `SEMBLE_MODEL_PATH`, so `--set-default` keeps an explicit
#   flag winning. Downloads into ~/.cache on first use.
# BUMPING: update `version`/`rev` + src `hash`, set cargoHash to
#   pkgs.lib.fakeHash, `nix build` once, paste the real hash from the
#   mismatch error.
{
  fetchFromGitHub,
  makeWrapper,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  pname = "semble-rs";
  version = "0.9.1";
  src = fetchFromGitHub {
    owner = "johunsang";
    repo = "semble_rs";
    rev = "v0.9.1";
    hash = "sha256-WwVE+PM/fdlwZ7GXiCGiVAXex3a8/a7LCkqG22EKFy0=";
  };
  cargoHash = "sha256-IoZKXg/cEqs6MLgBvufp7n+NtGwhhXSGT+GZGFQ11Wk=";
  # ast-grep is deliberately out of this closure: only `find-pattern` needs it,
  # and we use search/deps/impact. It happens to resolve inside the dev shell,
  # which is incidental — keep semble-rs a standalone binary.

  nativeBuildInputs = [ makeWrapper ];
  postInstall = ''
    wrapProgram $out/bin/semble_rs \
      --set-default SEMBLE_MODEL_PATH minishlab/potion-code-16M-v2
  '';

  meta.mainProgram = "semble_rs";
}
