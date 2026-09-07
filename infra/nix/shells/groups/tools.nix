# Things that *do* something: manage secrets, move data, run gates, transform
# files, render output. A tool whose only output is a measurement belongs in
# diagnostics.nix instead. Descriptor convention: see groups/shell.nix.
{ pkgs }:
{
  description = "Operator tooling: secrets, the DuckDB/dbt data plane, DVC, the hook runner, watch loops, data transformation, and rendering.";
  packages = with pkgs; [
    jq # `jq`: JSON slicing and transformation; prefer a producer's --json over scraping human output
    yq # `yq`: jq-style expressions over YAML/TOML/XML
    sd # `sd FIND REPLACE`: literal-first find-and-replace; `-p` previews. The literal counterpart to `ast-grep`'s syntactic rewrite
    dotenvx # `dotenvx run -- CMD`: encrypted .env management; decrypts the vault into the process env
    libsecret # `secret-tool`: reads/writes the DOTENV_PRIVATE_KEY in the OS keyring for dotenvx; needs a running Secret Service daemon
    dvc-with-remotes # `dvc`: Data Version Control — pipeline + data versioning, GCS remote included
    prek # `prek`: Rust drop-in for pre-commit; runs every .pre-commit-config.yaml in the tree
    pandoc # `pandoc`: universal document converter; used for CSL -> BibTeX reference projections
    watchexec # `watchexec`: rerun a scoped command on file changes — scope with `-e`/`-w`, settle with `--debounce`
    difftastic # `difft OLD NEW`: AST-aware structural diff, for reading a refactor or a reformat
    graphviz # `dot -Tsvg`: renders DOT graphs (SVG unless a bitmap consumer needs `-Tpng`) — gprof2dot and pygraphviz
    merman-cli # `merman-cli render`: browserless Mermaid -> PNG for the collapsed DVC DAG
    ffmpeg # `ffmpeg`: converts generated Kokoro WAV audio into WebM/Opus assets
    espeak-ng # `espeak-ng`: grapheme-to-phoneme for the naming screen (`just naming screen`); ../env.nix points phonemizer at its .so
  ];
}
