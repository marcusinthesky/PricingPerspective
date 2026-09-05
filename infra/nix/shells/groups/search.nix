# Discovery: finding files, text, and code, and building a picture of what is in
# the repository and how it hangs together. The test is whether the tool answers
# "where is / what exists / what is this made of". A tool that changes something
# lives in tools.nix, even when you reached it through a search.
#
# ORDERED BY HOW MUCH THE TOOL UNDERSTANDS OF YOUR QUERY — exact characters, then
# language structure, then meaning. Reading the list top to bottom is the
# escalation path: start at the cheapest tool that can express the question, and
# move down only when it cannot. (Order is also load-bearing for the derivation
# and for PATH; see ../default.nix.)
#
# ast-grep is the deliberate exception to the group's rule: it both queries and
# rewrites, and a single binary cannot sit in two groups. It is here because its
# query modality is one this group would otherwise lack.
#
# Descriptor convention: see groups/shell.nix.
{ pkgs }:
{
  description = "Discovery: find files, text, and code — exact, then structural, then semantic — and size up what the repository contains.";
  packages = with pkgs; [
    ## ── exact characters ──────────────────────────────────────────────────
    ripgrep # `rg`: regex over file contents, gitignore-aware
    fd # `fd`: regex over file names; `-H` includes hidden paths, `-x` runs a command per result

    ## ── approximate characters ────────────────────────────────────────────
    ## Still character-level, just tolerant of gaps. Reach for it when you know
    ## roughly what a path or symbol is called but not exactly. Use `-f/--filter`
    ## to stay non-interactive — `fd … | fzf -f query` composes into a recipe or
    ## an agent step, where the TUI does not.
    fzf # `fzf`: fuzzy filter over stdin; `-f` prints matches and exits, `-e` forces exact

    ## ── language structure ────────────────────────────────────────────────
    ast-grep # `ast-grep`: query (and rewrite) by AST pattern (no `sg` alias, so nothing shadows shadow's `sg`)

    ## ── meaning ───────────────────────────────────────────────────────────
    ## Beyond this point the question is no longer answerable by matching text.
    ## For imports, drift, and blast radius across the whole repo, escalate again
    ## to the DuckDB harness (`just harnessme`), which has no CLI of its own.
    semble-rs # `semble_rs search|deps|impact`: semantic + BM25 search, and transitive blast radius

    ## ── census, not matching ──────────────────────────────────────────────
    ## Off the spectrum above: tokei answers "what is this made of", not "where
    ## is X".
    tokei # `tokei <subtree>`: lines of code by language; `--sort code` to rank
  ];
}
