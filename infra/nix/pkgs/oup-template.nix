# Oxford University Press's authoring template is distributed as a CTAN source
# bundle rather than through the repository's Tectonic subset. Pin the current
# package archive by its content hash and materialize only the class and
# author-date bibliography style needed by OUP journal targets.
{
  coreutils,
  fetchurl,
  runCommand,
  unzip,
  writeShellApplication,
}:
let
  oupTemplateArchive = fetchurl {
    url = "https://mirrors.ctan.org/macros/latex/contrib/oup-authoring-template.zip";
    hash = "sha256-AlQ09T7TpGuf2E1Qp4xzTDCUdrxHsct6kMODc03zjpc=";
  };
  oupTemplateSource =
    runCommand "oup-authoring-template-source"
      {
        nativeBuildInputs = [ unzip ];
      }
      ''
        mkdir -p "$out"
        unzip -q "${oupTemplateArchive}" -d "$out"
      '';
in
writeShellApplication {
  name = "oup-template";
  runtimeInputs = [ coreutils ];
  text = ''
    if [[ $# -ne 1 ]]; then
      echo "usage: oup-template DESTINATION" >&2
      exit 2
    fi
    source_root=${oupTemplateSource}/oup-authoring-template
    destination=$1
    mkdir -p "$destination"
    # The upstream class invokes the optional external word-count utility via
    # a shell escape. The repository's untrusted/offline build contract forbids
    # that hook, so materialize the otherwise identical class without it.
    sed '/\\immediate\\write18{texcount /d' \
      "$source_root/oup-authoring-template.cls" \
      > "$destination/oup-authoring-template.cls"
    chmod 0644 "$destination/oup-authoring-template.cls"
    install -m 0644 "$source_root/oup-abbrvnat.bst" "$destination/oup-abbrvnat.bst"
  '';
}
