# Ordered as the pipeline runs: classify, then extract, then derive, then
# render, then compare. `pdf-inspector` comes first because it answers the
# question the rest depend on — is there extractable text, or does this need
# OCR? — which was previously inferred from empty `pdftotext` output.
#
# Descriptor convention: see groups/shell.nix, with one exception noted below.
{ pkgs }:
{
  description = "Native PDF/raster primitives used by the visual-document-analysis skill.";
  packages = with pkgs; [
    ## Triage. NOTE: these binaries take a path, not flags — `detect-pdf --help`
    ## fails with an IO error because it reads `--help` as a filename.
    pdf-inspector # `detect-pdf FILE`: text-based vs scanned, no OCR; `pdf2md FILE`: layout-aware Markdown

    ## Extract
    poppler-utils # `pdftotext`, `pdftoppm`, `pdfinfo`: PDF text extraction, rasterization, metadata
    pdfannots # `pdfannots`: extract PDF annotations to Markdown/JSON, page geometry preserved
    qpdf # `qpdf --check`: PDF container/structure inspection and transformation

    ## Derive a searchable copy, when triage says the text is not already there
    ocrmypdf # `ocrmypdf`: add an OCR text layer to a scanned PDF (PDF/A out)
    tesseract # `tesseract`: the OCR engine OCRmyPDF drives; also usable standalone

    ## Render and compare
    mupdf-headless # `mutool`: independent PDF renderer — a second opinion when Poppler's output is disputed
    imagemagick # `magick`: raster identify, compare, and compose
    ## NOTE: `diff-pdf` is a wxGTK binary, so it prints "Unable to initialize GTK+" to stderr on
    ## every headless run and aborts on `--view`/`--help`; the comparison itself and
    ## `--output-diff` still work, and the exit status (0 identical / 1 differing) is the result.
    diff-pdf # `diff-pdf A.pdf B.pdf`: page-aligned visual diff of two PDFs; answers "did the rebuild change anything, and where?"
  ];
}
