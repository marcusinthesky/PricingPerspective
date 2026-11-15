# Elsevier's elsarticle bundle (v3.5, 2026/01/09) is distributed on CTAN as a
# docstrip source rather than a ready class, and the Tectonic subset does not
# carry it. Pin the CTAN archive by its content hash, generate `elsarticle.cls`
# from the pinned `.dtx` at build time, and materialize the class with its
# author-date (`elsarticle-harv`) and numbered bibliography styles for Elsevier
# journal targets such as the Global Finance Journal.
#
# BUMPING: the CTAN URL is unversioned, so an upstream release changes the
#   archive and fails this fetch on hash mismatch. Update `hash` and the version
#   in this comment together, then rebuild `nix build ./infra/nix#elsarticle-template`.
{
  coreutils,
  fetchurl,
  runCommand,
  texliveInfraOnly,
  unzip,
  writeShellApplication,
}:
let
  elsarticleArchive = fetchurl {
    url = "https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip";
    hash = "sha256-8lo+GwBuD4arsJGgLzr6fG1gSRZlx6qJ9wzFYJ4U2dw=";
  };
  # docstrip only needs an engine and `docstrip.tex`; it typesets nothing.
  docstripEnv = texliveInfraOnly.withPackages (ps: [ ps.latex-bin ]);
  elsarticleSource =
    runCommand "elsarticle-template-source"
      {
        nativeBuildInputs = [
          docstripEnv
          unzip
        ];
      }
      ''
        unzip -q "${elsarticleArchive}"
        cd elsarticle
        export HOME="$TMPDIR" TEXMFVAR="$TMPDIR/texmf-var"
        pdflatex -interaction=nonstopmode elsarticle.ins
        test -s elsarticle.cls
        mkdir -p "$out"
        install -m 0644 elsarticle.cls "$out/elsarticle.cls"
        for style in elsarticle-harv.bst elsarticle-num.bst elsarticle-num-names.bst; do
          install -m 0644 "$style" "$out/$style"
        done
      '';
in
writeShellApplication {
  name = "elsarticle-template";
  runtimeInputs = [ coreutils ];
  text = ''
    if [[ $# -ne 1 ]]; then
      echo "usage: elsarticle-template DESTINATION" >&2
      exit 2
    fi
    source_root=${elsarticleSource}
    destination=$1
    mkdir -p "$destination"
    install -m 0644 "$source_root/elsarticle.cls" "$destination/elsarticle.cls"
    for style in elsarticle-harv.bst elsarticle-num.bst elsarticle-num-names.bst; do
      install -m 0644 "$source_root/$style" "$destination/$style"
    done
  '';
}
