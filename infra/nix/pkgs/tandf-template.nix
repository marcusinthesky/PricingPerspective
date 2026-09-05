# Taylor & Francis distributes both its generic Interact author template and
# Quantitative Finance's journal-specific rQUF package outside TeX Live. Keep
# Interact pinned from the maintained rticles mirror for other manuscripts, and
# pin the official QF archive by content hash. The materializer puts both sets
# beside each manuscript so builds and clean upload bundles have no host
# dependency.
{
  coreutils,
  fetchFromGitHub,
  fetchurl,
  runCommand,
  texlivePackages,
  unzip,
  writeShellApplication,
}:
let
  interactTemplateSource = fetchFromGitHub {
    owner = "rstudio";
    repo = "rticles";
    rev = "2e55e747dacb65c2645cc0273db7f7374a1bb1f2";
    hash = "sha256-ZI1WYHFxElsKalnFDOVgQHW1T85qjnOPZ4/604hYXOU=";
  };
  rqufArchive = fetchurl {
    url = "https://files.taylorandfrancis.com/rqufLatex.zip";
    hash = "sha256-w9qasLPyDn2Pywvp2eka7iNJMCYr8G6BQlPqX7JNTuQ=";
  };
  latexReleaseArchive = fetchurl {
    url = "https://github.com/latex3/latex2e/releases/download/release-2021-11-15-PL1/latex-base.tds.zip";
    hash = "sha256-s8QmZpYDF9U4n7QNGXvsk61WlKYOMI0bPsJnJEz8vBA=";
  };
  amsmathReleaseArchive = fetchurl {
    url = "https://github.com/latex3/latex2e/releases/download/release-2021-11-15-PL1/latex-amsmath.tds.zip";
    hash = "sha256-T2IpSb/oE9G4poZnat3CbknB2wRVIGozpr5Ruopx1/M=";
  };
  graphicsReleaseArchive = fetchurl {
    url = "https://github.com/latex3/latex2e/releases/download/release-2021-11-15-PL1/latex-graphics.tds.zip";
    hash = "sha256-ZMh2vkLnWOV7j8v3OxzjnBb+ES7a36pcwxZ5jyZcV4M=";
  };
  toolsReleaseArchive = fetchurl {
    url = "https://github.com/latex3/latex2e/releases/download/release-2021-11-15-PL1/latex-tools.tds.zip";
    hash = "sha256-kfARr0ul4CbuUrz7nWxEMInUNbmnhRCkqqo86Ltp9t0=";
  };
  l3PackagesReleaseArchive = fetchurl {
    url = "https://github.com/latex3/latex3/releases/download/2022-02-24/l3packages.tds.zip";
    hash = "sha256-qxeeG1wLDc9F74nbAR80aAtNPLzRQlZCWjX5HepiJ5Q=";
  };
  rqufTemplateSource = runCommand "rquf-latex-template" { nativeBuildInputs = [ unzip ]; } ''
    mkdir -p "$out"
    unzip -j "${rqufArchive}" rQUF2e.cls rQUF.bst -d "$out"
    unzip -j "${latexReleaseArchive}" tex/latex/base/latexrelease.sty -d "$out"
    unzip -j "${amsmathReleaseArchive}" tex/latex/amsmath/amsmath-2018-12-01.sty -d "$out"
    unzip -j "${graphicsReleaseArchive}" tex/latex/graphics/graphics-2017-06-25.sty -d "$out"
    unzip -j "${toolsReleaseArchive}" tex/latex/tools/array-2016-10-06.sty -d "$out"
    unzip -j "${toolsReleaseArchive}" tex/latex/tools/longtable-2020-01-07.sty -d "$out"
    unzip -j "${l3PackagesReleaseArchive}" tex/latex/l3packages/xparse/xparse.sty -d "$out"
  '';
in
writeShellApplication {
  name = "tandf-template";
  runtimeInputs = [ coreutils ];
  text = ''
    if [[ $# -ne 1 ]]; then
      echo "usage: tandf-template DESTINATION" >&2
      exit 2
    fi
    interact_root=${interactTemplateSource}/inst/rmarkdown/templates/tf/skeleton
    rquf_root=${rqufTemplateSource}
    destination=$1
    mkdir -p "$destination"
    install -m 0644 "$interact_root/interact.cls" "$destination/interact.cls"
    for style in "$interact_root"/tf*.bst; do
      install -m 0644 "$style" "$destination/$(basename "$style")"
    done
    install -m 0644 "$rquf_root/rQUF2e.cls" "$destination/rQUF2e.cls"
    install -m 0644 "$rquf_root/rQUF.bst" "$destination/rQUF.bst"
    # The QF guide requires latexrelease.sty, which the publisher archive does
    # not ship. Match latexrelease to Tectonic's 2021-11-15 kernel and
    # materialize the remaining rollback closure from pinned TeX distributions
    # so the upload and offline build close over the same dependencies.
    install -m 0644 "$rquf_root/latexrelease.sty" "$destination/latexrelease.sty"
    install -m 0644 \
      "$rquf_root/amsmath-2018-12-01.sty" \
      "$destination/amsmath-2018-12-01.sty"
    install -m 0644 \
      "$rquf_root/graphics-2017-06-25.sty" \
      "$destination/graphics-2017-06-25.sty"
    install -m 0644 \
      "${texlivePackages.tools.outputDrvs.tex}/tex/latex/tools/enumerate.sty" \
      "$destination/enumerate.sty"
    install -m 0644 \
      "${texlivePackages.everyshi.outputDrvs.tex}/tex/latex/everyshi/everyshi.sty" \
      "$destination/everyshi.sty"
    install -m 0644 \
      "${texlivePackages.everyshi.outputDrvs.tex}/tex/latex/everyshi/everyshi-2001-05-15.sty" \
      "$destination/everyshi-2001-05-15.sty"
    install -m 0644 \
      "${texlivePackages.siunitx.outputDrvs.tex}/tex/latex/siunitx/siunitx-v2.sty" \
      "$destination/siunitx-v2.sty"
    for config in siunitx-abbreviations.cfg siunitx-binary.cfg siunitx-version-1.cfg; do
      install -m 0644 \
        "${texlivePackages.siunitx.outputDrvs.tex}/tex/latex/siunitx/$config" \
        "$destination/$config"
    done
    for metric in rm-lmss9.tfm rm-lmsso9.tfm; do
      install -m 0644 \
        "${texlivePackages.lm.outputDrvs.tex}/fonts/tfm/public/lm/$metric" \
        "$destination/$metric"
    done
    install -m 0644 \
      "${texlivePackages.lm.outputDrvs.tex}/fonts/tfm/public/lm/ec-lmcsc10.tfm" \
      "$destination/ec-lmcsc10.tfm"
    install -m 0644 \
      "${texlivePackages.lm.outputDrvs.tex}/fonts/type1/public/lm/lmcsc10.pfb" \
      "$destination/lmcsc10.pfb"
    install -m 0644 \
      "${texlivePackages.amsfonts.outputDrvs.tex}/fonts/type1/public/amsfonts/cm/cmr10.pfb" \
      "$destination/cmr10.pfb"
    unicode_root="${texlivePackages."unicode-data".outputDrvs.tex}/tex/generic/unicode-data"
    for support in load-unicode-xetex-classes.tex LineBreak.txt EastAsianWidth.txt; do
      install -m 0644 "$unicode_root/$support" "$destination/$support"
    done
    install -m 0644 "$rquf_root/xparse.sty" "$destination/xparse.sty"
    install -m 0644 \
      "$rquf_root/array-2016-10-06.sty" \
      "$destination/array-2016-10-06.sty"
    install -m 0644 \
      "$rquf_root/longtable-2020-01-07.sty" \
      "$destination/longtable-2020-01-07.sty"
  '';
}
