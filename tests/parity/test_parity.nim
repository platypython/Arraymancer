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

## Exercises portable conformance comparisons, native matches and strict coverage policy.
import std/[json, unittest]
import ./[parity_support, native_cases, run_parity]

proc duplicate(node: JsonNode): JsonNode = parseJson($node)

proc withoutFirst(node: JsonNode): JsonNode =
  result = newJArray()
  for i in 1 ..< node.len: result.add(node[i])

suite "Pinned PyTorch conformance foundation":
  let reference = loadReference(parityRoot & "reference.json")
  let inventory = parseFile(parityRoot & "inventory.json")
  let capabilities = parseFile(parityRoot & "capabilities.json")
  let requirements = loadAppleRequirements(parityRoot & "apple_requirements.json")

  test "Matching native CPU contracts":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    for caseData in fixtures["cases"]:
      check compareObservations(caseData["expected"], runCase(caseData),
        caseData["atol"].getFloat(), caseData["rtol"].getFloat()).len == 0

  test "Shape differs even when flattened values match":
    let expected = %*{"result": {"shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]}}
    let actual = %*{"result": {"shape": [4], "values": [1.0, 2.0, 3.0, 4.0]}}
    let differences = compareObservations(expected, actual, 0, 0)
    check differences.len > 0
    check differences[0].path == "/result/shape"

  test "Nonfinite values never match finite values or opposite infinities":
    check compareObservations(%*{"values": [1.0]}, %*{"values": ["nan"]}, 1, 1).len == 1
    check compareObservations(%*{"values": ["+inf"]}, %*{"values": ["-inf"]}, 1, 1).len == 1
    check compareObservations(%*{"values": ["nan"]}, %*{"values": ["nan"]}, 0, 0).len == 0

  test "Tolerance boundary is inclusive and elementwise":
    check compareObservations(%*{"values": [1.0]}, %*{"values": [1.25]}, 0.125, 0.125).len == 0
    check compareObservations(%*{"values": [1.0]}, %*{"values": [1.2501]}, 0.125, 0.125).len == 1
    check compareObservations(%*{"numel": 1}, %*{"numel": 2}, 100, 100).len == 1
    check compareObservations(%*{"values": [1.0, 2.0]}, %*{"values": [1.0]}, 0, 0).len == 1

  test "Fixture provenance and complete unique IDs are mandatory":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var malformed = duplicate(fixtures)
    malformed["reference"]["torch"] = %"0.0.0"
    expect ValueError: validateFixtureData(malformed, reference)
    malformed = duplicate(fixtures)
    malformed["cases"] = withoutFirst(malformed["cases"])
    expect ValueError: validateFixtureData(malformed, reference)
    malformed = duplicate(fixtures)
    malformed["cases"][1]["id"] = malformed["cases"][0]["id"]
    expect ValueError: validateFixtureData(malformed, reference)
    malformed = duplicate(fixtures)
    malformed["oracle"]["python"] = %"3.13.0"
    expect ValueError: validateFixtureData(malformed, reference)

  test "Invalid selectors fail before execution":
    expect ValueError: discard parseSelection(@["--profile=unknown"])
    expect ValueError: discard parseSelection(@["--case=tensor.matmul_2d"])
    expect ValueError: discard parseSelection(@["--profile=apple", "--case=tensor.matmul_2d"])
    expect ValueError:
      discard collectReport(parseSelection(@["--profile=cpu", "--case=unknown"]))
    expect ValueError: discard parseSelection(@["--requirement=apple.metal.device_dispatch"])
    expect ValueError: discard parseSelection(@["--profile=apple", "--requirement=apple.metal.device_dispatch"])
    expect ValueError: discard parseSelection(@["--profile=apple-coreai", "--requirement=apple.metal.device_dispatch"])
    expect ValueError: discard parseSelection(@["--profile=apple-metal", "--requirement=unknown"])

  test "Apple profiles remain independent mandatory gates":
    let metal = collectAppleCoverage(requirements, capabilities, @["apple-metal"], [])
    let coreai = collectAppleCoverage(requirements, capabilities, @["apple-coreai"], [])
    check metal.len == 10
    check coreai.len == 8
    check collectAppleCoverage(requirements, capabilities, @["cpu"], []).len == 0
    for entries in [metal, coreai]:
      for entry in entries:
        check entry["execution"].getStr() == "not_run"
        check entry["actual"].kind == JNull
      let report = %*{"counts": {"mismatch": 0, "error": 0},
        "apple_counts": {"unobserved": entries.len, "mismatch": 0, "error": 0}}
      check reportExitCode(report) == 1

  test "Partial annotations cannot establish execution":
    var changed = duplicate(capabilities)
    for i in 0 ..< changed["apple_requirements"].len:
      changed["apple_requirements"][i]["status"] = %"partial"
    validateCapabilities(changed, inventory, requirements)
    let coverage = collectAppleCoverage(requirements, changed, @["apple-metal", "apple-coreai"], [])
    check coverage.len == 18
    for entry in coverage:
      check entry["implementation_status"].getStr() == "partial"
      check entry["execution"].getStr() == "not_run"
      check entry["actual"].kind == JNull
    check reportExitCode(%*{"counts": {"mismatch": 0, "error": 0},
      "apple_counts": {"unobserved": coverage.len, "mismatch": 0, "error": 0}}) == 1

  test "Missing Apple profiles requirements and MPS links are invalid":
    var malformed = duplicate(requirements)
    malformed["profiles"] = withoutFirst(malformed["profiles"])
    expect ValueError: validateAppleRequirements(malformed, reference, inventory)
    malformed = duplicate(requirements)
    malformed["requirements"] = withoutFirst(malformed["requirements"])
    expect ValueError: validateAppleRequirements(malformed, reference, inventory)
    malformed = duplicate(requirements)
    for row in malformed["requirements"]:
      var remaining = newJArray()
      for id in row["reference_ids"]:
        if id.getStr() != "mps:torch.mps.compile_shader": remaining.add(id)
      row["reference_ids"] = remaining
    expect ValueError: validateAppleRequirements(malformed, reference, inventory)

  test "Verified annotations and incomplete area coverage are invalid":
    var malformed = duplicate(capabilities)
    malformed["apple_requirements"][0]["status"] = %"verified"
    expect ValueError: validateCapabilities(malformed, inventory, requirements)
    malformed = duplicate(capabilities)
    malformed["areas"] = withoutFirst(malformed["areas"])
    expect ValueError: validateCapabilities(malformed, inventory, requirements)

  test "Typed native observations distinguish pass mismatch and error":
    let id = "apple.metal.device_dispatch"
    let evidence = %*{"invocation_id": 1, "completed": true,
      "kind": "device_storage_control", "cpu_fallback_count": 0}
    var observation = AppleObservation(id: id, profile: "apple-metal", execution: aePass,
      actual: %*{"observed": true}, evidence: evidence)
    let passed = collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation], id)
    check passed.len == 1
    check passed[0]["execution"].getStr == "pass"
    check completedNativeInvocation(passed[0]["evidence"])
    observation.execution = aeMismatch
    observation.differences = @[Difference(path: "/observed", expected: %true, actual: %false)]
    let mismatch = collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation], id)
    check mismatch[0]["execution"].getStr == "mismatch"
    check mismatch[0]["differences"].len == 1
    observation.execution = aeError
    observation.differences = @[]
    observation.error = %*{"category": "unavailable", "message": "isolated policy probe"}
    let errored = collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation], id)
    check errored[0]["execution"].getStr == "error"
    for execution in ["mismatch", "error"]:
      var counters = %*{"unobserved": 0, "mismatch": 0, "error": 0}
      counters[execution] = %1
      check reportExitCode(%*{"counts": {"mismatch": 0, "error": 0}, "apple_counts": counters}) == 1

  test "Malformed duplicate conflicting and uncompleted observations fail closed":
    let id = "apple.metal.device_dispatch"
    var observation = AppleObservation(id: id, profile: "apple-metal", execution: aePass,
      actual: %*{"observed": true}, evidence: %*{"invocation_id": 1, "completed": true,
        "kind": "device_storage_control", "cpu_fallback_count": 0})
    expect ValueError:
      discard collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation, observation])
    observation.id = "unknown"
    expect ValueError: discard collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation])
    observation.id = id
    observation.profile = "apple-coreai"
    expect ValueError: discard collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation])
    observation.profile = "apple-metal"
    observation.evidence["completed"] = %false
    expect ValueError: discard collectAppleCoverage(requirements, capabilities, ["apple-metal"], [observation])
