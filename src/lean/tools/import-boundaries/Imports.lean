import Lean

/-! Extract direct imports from compiled module headers. Loading all first-party
modules covers leaves outside roll-ups without transitive-reduction or unused-import
analysis; those presentation transformations would hide actual dependency edges. -/

open Lean

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let names := args.toArray.map String.toName
  let env ← importModules (names.map fun name ↦ { module := name }) {} (loadExts := false)
  let records ← names.mapM fun name ↦ do
    let some index := env.getModuleIdx? name
      | throw <| IO.userError s!"compiled source module missing: {name}"
    let imports := env.header.moduleData[index.toNat]!.imports.map fun item ↦ item.module.toString
    return (name.toString, toJson imports)
  IO.println <| (Json.mkObj records.toList).compress
