import Lean

/-!
# Declaration dump — the Lean plane's extractor

Emits one JSON object per declaration in a module closure, to stdout, as JSONL:
`{name, kind, module, doc?, type?, is_mathlib}`.

## Why this exists

Agents write Lean against a *pinned* mathlib (`v4.31.0` here). The recurring
failure is a hallucinated lemma name or a wrong import path, and the only thing
that currently catches it is a compile cycle. A local, queryable declaration
table turns that into a lookup.

## Why not the published index

`leanprover-community.github.io/mathlib4_docs/declarations/declaration-data.bmp`
is 66 MB of ready-made JSON, but it is built from mathlib **master**. Against a
pinned revision that is precisely the wrong tool: it reports declarations that do
not exist at `v4.31.0`, which is the failure mode we are trying to remove. This
dumps the environment the workspace actually compiles against.

## Why the imports are resolved at run time

`import Mathlib` in this file would make every rebuild of the tool elaborate
mathlib's root module. Importing at run time via `--modules` keeps the executable
a few seconds to build, and makes it general: dump mathlib, dump only this
repo's packages, or dump both.
-/

open Lean

namespace DeclDump

/-- `ConstantInfo`'s constructor, as the kind string the SQL layer groups on. -/
private def kindOf : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

/--
Declarations the elaborator generated rather than a human wrote: `_proof_1`,
`Foo.match_2`, equation lemmas, instance witnesses. They outnumber the real
declarations and none of them is nameable in a proof, so they are noise for
every downstream query.
-/
private def isNoise (env : Environment) (name : Name) : Bool :=
  name.isInternal
    || name.isInternalDetail
    || isPrivateName name
    || (env.find? name).isNone

/--
The module a declaration was imported from.

There is no `Environment.getModuleFor?`; the lookup is two steps — a
`ModuleIdx` (a `Nat` with a `GetElem` instance) indexing `header.moduleNames`.
Returns `none` for a declaration added in the current session rather than
imported, which a pure import closure has none of.
-/
private def moduleOf (env : Environment) (name : Name) : Option Name :=
  (env.getModuleIdxFor? name).map fun idx => env.header.moduleNames[idx]!

structure Config where
  /-- Root modules whose transitive closure is dumped. -/
  modules : Array Name := #[`Mathlib]
  /-- Pretty-print each declaration's type. Costs real time; see `--no-types`. -/
  withTypes : Bool := true
  deriving Inhabited

/--
Pretty-print a type, returning `none` rather than failing the whole dump.

`ppExpr` runs the delaborator, which can throw on pathological terms. One
unprintable declaration out of ~300k must not lose the other 299,999, so the
failure is swallowed per declaration and shows up as a null `type` column that
SQL can count.
-/
private def ppType (ci : ConstantInfo) : MetaM (Option String) := do
  try
    let fmt ← Meta.ppExpr ci.type
    return some (toString fmt)
  catch _ =>
    return none

private def emit (cfg : Config) (env : Environment) (name : Name) (ci : ConstantInfo)
    (out : IO.FS.Stream) : MetaM Unit := do
  let mod := moduleOf env name
  let doc ← findDocString? env name
  let type ← if cfg.withTypes then ppType ci else pure none
  let fields : List (String × Json) :=
    [ ("name", toJson name.toString)
    , ("kind", toJson (kindOf ci))
    , ("module", toJson (mod.map Name.toString))
    , ("doc", toJson doc)
    , ("type", toJson type)
    , ("is_mathlib", toJson (mod.map (·.getRoot == `Mathlib) |>.getD false)) ]
  out.putStrLn (Json.mkObj fields).compress

def dump (cfg : Config) : MetaM Unit := do
  let env ← getEnv
  let out ← IO.getStdout
  -- `map₁` holds the imported constants, `map₂` the ones added since; a dump of
  -- a pure import closure has everything in `map₁`, but iterating both keeps the
  -- tool correct if it is ever invoked after elaborating something.
  for (name, ci) in env.constants.map₁.toList do
    unless isNoise env name do emit cfg env name ci out
  for (name, ci) in env.constants.map₂.toList do
    unless isNoise env name do emit cfg env name ci out

end DeclDump

private def parseArgs (args : List String) : DeclDump.Config :=
  go args {}
where
  go : List String → DeclDump.Config → DeclDump.Config
    | [], cfg => cfg
    | "--no-types" :: rest, cfg => go rest { cfg with withTypes := false }
    | "--modules" :: spec :: rest, cfg =>
      let mods := (spec.splitOn ",").filter (!·.isEmpty) |>.map String.toName
      go rest { cfg with modules := mods.toArray }
    | _ :: rest, cfg => go rest cfg

def main (args : List String) : IO UInt32 := do
  let cfg := parseArgs args
  if cfg.modules.isEmpty then
    IO.eprintln "decl_dump: --modules resolved to an empty list"
    return 1
  initSearchPath (← findSysroot)
  -- Required before importModules (loadExts := true), which refuses to run
  -- otherwise: loading an extension means executing its module initializer, so
  -- the call is `unsafe`. Same idiom as importGraph's MainGraph.lean and
  -- doc-gen4's Load.lean, which need loaded extensions for the same reason.
  unsafe Lean.enableInitializersExecution
  -- loadExts := true is what makes the `type` column readable. Delaborator
  -- unexpanders (the things that print `a = b` instead of `Eq a b`, `∀ x, p x`
  -- with binder notation, and every mathlib-defined notation) live in
  -- environment extensions, and importModules leaves those unloaded by default.
  -- Without it the dump still succeeds and every signature comes out in raw
  -- application form — technically correct and nearly unusable for the lookup
  -- this table exists to serve.
  let env ← importModules (cfg.modules.map ({ module := · })) {} (loadExts := true)
  IO.eprintln s!"decl_dump: imported {cfg.modules.size} root module(s), \
    {env.constants.map₁.size} constants"
  let ctx : Core.Context := { fileName := "<decl_dump>", fileMap := default }
  let state : Core.State := { env }
  discard <| (DeclDump.dump cfg).run'.toIO ctx state
  return 0
