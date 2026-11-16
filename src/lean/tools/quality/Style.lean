module

public meta import Mathlib.Tactic.Linter.TextBased

/-! Run Mathlib's text linters on explicit paths in the multi-package workspace.
The upstream CLI assumes module names are paths relative to the workspace root.
All rules and fixes here come from Mathlib; this adapter only handles paths and exit status.
-/

open Lean.Linter Mathlib.Linter.TextBased

/-- Check explicit source paths, or apply Mathlib's available fixes with `--fix`.
Unfixable findings still fail, including in fix mode. -/
public meta def main (args : List String) : IO UInt32 := do
  let fix := args.contains "--fix"
  let files := args.filter (· != "--fix")
  if files.isEmpty then
    IO.eprintln "quality_style: no source files supplied"
    return 1
  let opts : LinterOptions := { toOptions := {}, linterSets := {} }
  let mut result := 0
  for file in files do
    unless file.endsWith ".lean" do
      throw <| IO.userError s!"expected a Lean source path: {file}"
    let path := ((file.dropEnd 5).copy.replace "/" ".").toName
    if path.isAnonymous then
      throw <| IO.userError s!"use a path with valid module components: {file}"
    try
      let found ← lintModules opts #[] #[path] .humanReadable fix
      let remaining ← if fix then lintModules opts #[] #[path] .humanReadable false else pure found
      result := max result (min remaining 1)
    catch e =>
      throw <| IO.userError s!"Mathlib text lint failed for {file}: {e}"
  (← IO.getStdout).flush
  (← IO.getStderr).flush
  return result
