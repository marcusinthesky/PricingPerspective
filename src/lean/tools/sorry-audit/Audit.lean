import Lean
import Lean.Util.CollectAxioms

/-! Compiled trust and placeholder census. The Python driver supplies the source-module list.
Trust checks cover every declaration; source metadata only governs placeholder presentation. -/

open Lean Elab Command Meta

namespace SorryAudit

/-- Recognize `True` after unfolding ordinary aliases. -/
def isTrue (e : Expr) : MetaM Bool := do
  return (← withTransparency .all <| whnf e).isConstOf ``True

/-- Classify elaborated statements and Prop-valued definitions, never proof-script tokens. -/
def classify (name : Name) (ci : ConstantInfo) : MetaM (Array String × Array String) := do
  let proof ← isProp ci.type
  withTransparency .all <| (if proof then forallTelescopeReducing else forallTelescope) ci.type fun xs body => do
    let mut flags := #[]
    let mut premises := #[]
    for x in xs do
      if (← isTrue (← inferType x)) then
        premises := premises.push (← x.fvarId!.getDecl).userName.toString
    if proof then
      let body ← whnf body
      if (← isTrue body) then flags := flags.push "true-result"
      if body.isAppOfArity ``Iff 2 then
        let args := body.getAppArgs
        if (← isTrue args[0]!) || (← isTrue args[1]!) then
          flags := flags.push "true-iff"
    else if body.isSort && body.sortLevel!.isZero && ci.isDefinition then
      let value := mkAppN (mkConst name (ci.levelParams.map mkLevelParam)) xs
      if (← isTrue value) then flags := flags.push "true-definition"
    return (flags, premises)

/-- Emit JSON for every requested declaration, including private and generated names.
The current module is supported for integration fixtures. Classifier errors abort the gate. -/
def run (modules : Array Name) (includeCurrent : Bool := false) : TermElabM Unit := do
  let env ← getEnv
  let moduleNames := env.header.moduleNames
  for moduleName in modules do
    unless moduleNames.contains moduleName || (includeCurrent && moduleName == env.header.mainModule) do
      throwError "requested source module was not imported: {moduleName}"
  let owned := modules.foldl (fun (s : NameSet) m ↦ s.insert m) {}
  let mut seen : NameSet := {}
  let mut count := 0
  for (name, ci) in env.constants.map₁.toList ++ env.constants.map₂.toList do
    let source := (env.getModuleIdxFor? name).map fun i ↦ moduleNames[i]!
    if source.isNone && (!includeCurrent || !env.constants.map₂.contains name) then continue
    let source := source.getD env.header.mainModule
    if !owned.contains source then continue
    if seen.contains name then continue
    seen := seen.insert name
    count := count + 1
    let axioms ← Lean.collectAxioms name
    let mut flags := #[]
    let mut premises := #[]
    match ci with
    | .thmInfo _ | .defnInfo _ | .opaqueInfo _ =>
      let result ← withOptions (fun o ↦
        (o.set `maxHeartbeats (200000 : Nat)).set `maxRecDepth (4096 : Nat)) <|
          withCurrHeartbeats <| classify name ci
      flags := result.1
      premises := result.2
    | _ => pure ()
    let range ← findDeclarationRanges? name
    let record := Json.mkObj [
      ("declaration", toJson name.toString),
      ("user_name", toJson (privateToUserName name).toString),
      ("module", toJson source.toString),
      ("axiom", toJson (match ci with | .axiomInfo _ => true | _ => false)),
      ("axioms", toJson (axioms.map Name.toString)),
      ("flags", toJson flags), ("premises", toJson premises),
      ("selection", toJson (range.map fun r ↦
        (r.selectionRange.pos.line, r.selectionRange.pos.column,
         r.selectionRange.endPos.line, r.selectionRange.endPos.column)))]
    logInfo m!"SORRY_AUDIT_JSON {record.compress}"
  logInfo m!"SORRY_AUDIT_COUNT {count}"

end SorryAudit
