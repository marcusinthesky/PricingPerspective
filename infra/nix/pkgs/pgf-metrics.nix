# A pdflatex for Matplotlib's PGF backend — measurement only, renders no
# manuscript artifact. Why an engine is unavoidable and why Tectonic cannot serve
# the role: ../ARCHITECTURE.md § "Why Tectonic is the sole renderer, yet a second
# TeX engine still ships".
#
# CHANGING THE LIST: this is the smallest set that compiles matplotlib's own PGF
#   header plus every math label the manuscripts use — texliveBasic fails.
#   Re-measure against a real `savefig(*.pgf)` before trimming.
#
# `meta` is attached with `//` so the derivation hash is untouched — see chktex.nix.
{ texliveInfraOnly }:
let
  env = texliveInfraOnly.withPackages (ps: [
    ps.latex-bin
    ps.graphics
    ps.epstopdf-pkg # pulled in by matplotlib's PGF header, not by us
    ps.underscore # likewise
  ]);
in
env
// {
  meta = (env.meta or { }) // {
    mainProgram = "pdflatex";
  };
}
