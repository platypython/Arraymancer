# Copyright 2026 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Runs native CPU observations and reports required, unobserved Apple coverage.
import std/[json, os, parseopt]
import ./[parity_support, native_cases]
when defined(metal) or defined(coreai):
  import ./native_metal_cases

const parityRoot* = "tests/parity/"
const usage = "Usage: pytorch_parity [--profile=all|cpu|apple|apple-metal|apple-coreai] [--case=ID | --requirement=ID] [--fixtures=PATH] [--output=PATH] [--help]"

type Selection* = object
  profiles*: seq[string]
  caseId*, requirementId*, fixtures*, output*: string

proc parseSelection*(args: seq[string]): Selection =
  result.fixtures = parityRoot & "fixtures.json"
  var profile = "all"
  var explicitCpu = false
  var parser = initOptParser(args)
  for kind, key, value in parser.getopt():
    if kind != cmdLongOption: raise newException(ValueError, "Unexpected argument: " & key)
    case key
    of "profile":
      profile = value
      explicitCpu = value == "cpu"
    of "case":
      if value.len == 0: raise newException(ValueError, "--case requires an ID")
      result.caseId = value
    of "requirement":
      if value.len == 0: raise newException(ValueError, "--requirement requires an ID")
      result.requirementId = value
    of "fixtures":
      if value.len == 0: raise newException(ValueError, "--fixtures requires a path")
      result.fixtures = value
    of "output":
      if value.len == 0: raise newException(ValueError, "--output requires a path")
      result.output = value
    else: raise newException(ValueError, "Unknown option: " & key)
  case profile
  of "all": result.profiles = @["cpu", "apple-metal", "apple-coreai"]
  of "cpu": result.profiles = @["cpu"]
  of "apple": result.profiles = @["apple-metal", "apple-coreai"]
  of "apple-metal", "apple-coreai": result.profiles = @[profile]
  else: raise newException(ValueError, "Unknown profile: " & profile)
  if result.caseId.len > 0 and not explicitCpu:
    raise newException(ValueError, "--case requires explicit --profile=cpu")
  if result.requirementId.len > 0:
    if profile notin ["apple-metal", "apple-coreai"] or result.caseId.len > 0:
      raise newException(ValueError, "--requirement requires a single explicit Apple profile and excludes --case")
    let requirements = loadAppleRequirements(parityRoot & "apple_requirements.json")
    var found = false
    for req in requirements["requirements"]:
      if req["id"].getStr == result.requirementId and req["profile"].getStr == profile: found = true
    if not found: raise newException(ValueError, "Unknown or profile-conflicting requirement: " & result.requirementId)

proc reportExitCode*(report: JsonNode): int =
  if report["counts"]["mismatch"].getInt() > 0 or
      report["counts"]["error"].getInt() > 0 or
      report["apple_counts"]["unobserved"].getInt() > 0 or
      report["apple_counts"]["mismatch"].getInt() > 0 or
      report["apple_counts"]["error"].getInt() > 0: 1
  else: 0

proc collectReport*(selection: Selection): JsonNode =
  let reference = loadReference(parityRoot & "reference.json")
  let inventory = parseFile(parityRoot & "inventory.json")
  let capabilities = parseFile(parityRoot & "capabilities.json")
  let requirements = parseFile(parityRoot & "apple_requirements.json")
  let fixtures = loadFixtures(selection.fixtures)
  validateInventory(inventory, reference)
  validateAppleRequirements(requirements, reference, inventory)
  validateCapabilities(capabilities, inventory, requirements)
  var selected = newJArray()
  for caseData in fixtures["cases"]:
    if selection.caseId.len == 0 or caseData["id"].getStr() == selection.caseId:
      selected.add(caseData)
  if selected.len == 0: raise newException(ValueError, "Unknown or empty case selection: " & selection.caseId)
  var inventoryCounts = %*{"api_area": 0, "aten_schema": 0, "autograd_rule": 0, "mps_export": 0}
  for record in inventory["records"]:
    let kind = record["kind"].getStr()
    inventoryCounts[kind] = %(inventoryCounts[kind].getInt() + 1)
  var capabilityCounts = %*{"partial": 0, "missing": 0, "unreviewed": 0}
  for area in capabilities["areas"]:
    let status = area["status"].getStr()
    capabilityCounts[status] = %(capabilityCounts[status].getInt() + 1)
  var results = newJArray()
  var counts = %*{"pass": 0, "mismatch": 0, "error": 0}
  if "cpu" in selection.profiles:
    for caseData in selected:
      var entry = %*{"id": caseData["id"], "area": caseData["area"], "profile": "cpu",
        "status": "error", "actual": nil, "differences": [], "error": nil}
      try:
        let actual = runCase(caseData)
        let differences = compareObservations(caseData["expected"], actual,
          caseData["atol"].getFloat(), caseData["rtol"].getFloat())
        entry["actual"] = actual
        entry["status"] = %(if differences.len == 0: "pass" else: "mismatch")
        for difference in differences:
          entry["differences"].add(%*{"path": difference.path,
            "expected": difference.expected, "actual": difference.actual})
      except CatchableError, Defect:
        let error = getCurrentException()
        entry["error"] = %*{"type": $error.name, "message": error.msg}
      let status = entry["status"].getStr()
      counts[status] = %(counts[status].getInt() + 1)
      results.add(entry)
  var observations: seq[AppleObservation]
  when defined(metal) or defined(coreai):
    if "apple-metal" in selection.profiles:
      observations = runMetalRequirements(selection.requirementId)
  let appleCoverage = collectAppleCoverage(requirements, capabilities, selection.profiles,
    observations, selection.requirementId)
  var appleCounts = %*{"required": appleCoverage.len, "partial": 0, "missing": 0,
    "unreviewed": 0, "native_executed": 0, "unobserved": 0,
    "pass": 0, "mismatch": 0, "error": 0,
    "gate": (if appleCoverage.len == 0: "not_selected" else: "incomplete")}
  for entry in appleCoverage:
    let status = entry["implementation_status"].getStr()
    appleCounts[status] = %(appleCounts[status].getInt() + 1)
    let execution = entry["execution"].getStr
    let counter = if execution == "not_run": "unobserved" else: execution
    appleCounts[counter] = %(appleCounts[counter].getInt + 1)
    if completedNativeInvocation(entry["evidence"]):
      appleCounts["native_executed"] = %(appleCounts["native_executed"].getInt + 1)
  if appleCoverage.len > 0 and appleCounts["pass"].getInt == appleCoverage.len:
    appleCounts["gate"] = %"pass"
  result = %*{"format": 2, "reference": reference, "scope": "native_apple_conformance",
    "profiles": selection.profiles, "selection": (if selection.requirementId.len > 0: selection.requirementId
      elif selection.caseId.len == 0: "all" else: selection.caseId),
    "inventory_counts": inventoryCounts, "capability_counts": capabilityCounts,
    "results": results, "counts": counts, "apple_coverage": appleCoverage,
    "apple_counts": appleCounts, "notice": "Full feature parity is not established by this run."}

proc main(): int =
  if commandLineParams() == @["--help"]:
    echo usage
    return 0
  if paramCount() == 0: stderr.writeLine(usage)
  let selection = parseSelection(commandLineParams())
  let report = collectReport(selection)
  let text = report.pretty() & "\n"
  if selection.output.len > 0: writeFile(selection.output, text)
  else: stdout.write(text)
  result = reportExitCode(report)

when isMainModule:
  try: quit(main())
  except CatchableError, Defect:
    stderr.writeLine(getCurrentExceptionMsg())
    quit(2)
