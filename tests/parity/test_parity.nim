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


proc isNegativeZero(value: JsonNode): bool =
  value.kind == JFloat and (cast[uint64](value.getFloat) shr 63) == 1'u64
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

  test "Integer scans cover default promotion and exact outputs":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var scanCase = newJNull()
    for caseData in fixtures["cases"]:
      if caseData["id"].getStr == "tensor.integer_scans": scanCase = caseData
    doAssert scanCase.kind == JObject
    check scanCase["atol"].getFloat == 0
    check scanCase["rtol"].getFloat == 0
    for dtype in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
      check scanCase["inputs"].hasKey(dtype)
      for operation in ["cumsum", "cumprod"]:
        let observation = scanCase["expected"][dtype][operation]["-1"]
        check observation["dtype"].getStr == "int64"
        for value in observation["values"]:
          check value.kind == JInt
    check scanCase["inputs"]["uint64"]["words"][2].getBiggestInt == low(int64)
    check scanCase["inputs"]["uint64"]["words"][3].getBiggestInt == -1
    for operation in ["cumsum", "cumprod"]:
      check scanCase["expected"]["scalar"][operation]["-1"]["shape"].len == 0
      check scanCase["expected"]["empty"][operation]["1"]["shape"] == %*[2, 0, 3]
      check scanCase["expected"]["empty"][operation]["1"]["values"].len == 0
    for view in ["transpose", "offset", "reverse", "broadcast", "overflow"]:
      check scanCase["inputs"].hasKey(view)
    var malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.integer_scans":
        let values: JsonNode = malformed["cases"][index]["expected"]["bool"]["cumsum"]["0"]["values"]
        values.elems[0] = %0.0
    expect ValueError: validateFixtureData(malformed, reference)

  test "Integer reductions cover default promotion and exact outputs":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var reductionCase = newJNull()
    for caseData in fixtures["cases"]:
      if caseData["id"].getStr == "tensor.integer_reductions": reductionCase = caseData
    doAssert reductionCase.kind == JObject
    check reductionCase["atol"].getFloat == 0
    check reductionCase["rtol"].getFloat == 0
    for dtype in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
      check reductionCase["inputs"].hasKey(dtype)
      for operation in ["sum", "prod"]:
        let byAxis = reductionCase["expected"][dtype][operation]
        check byAxis.hasKey("all")
        check byAxis.hasKey("0")
        check byAxis.hasKey("1")
        check byAxis.hasKey("-1")
        check byAxis.hasKey("-2")
        for _, observation in byAxis:
          check observation["dtype"].getStr == "int64"
          for value in observation["values"]:
            check value.kind == JInt
    check reductionCase["inputs"]["uint64"]["words"][2].getBiggestInt == low(int64)
    check reductionCase["inputs"]["uint64"]["words"][3].getBiggestInt == -1
    check reductionCase["expected"]["scalar"]["sum"]["all"]["shape"].len == 0
    check reductionCase["expected"]["empty"]["sum"]["all"]["values"][0].getBiggestInt == 0
    check reductionCase["expected"]["empty"]["prod"]["all"]["values"][0].getBiggestInt == 1
    for view in ["overflow", "transpose", "offset", "reverse", "broadcast"]:
      check reductionCase["inputs"].hasKey(view)
    var malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.integer_reductions":
        malformed["cases"][index]["expected"]["uint64"]["prod"]["all"]["dtype"] = %"uint64"
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.integer_reductions":
        malformed["cases"][index]["inputs"]["uint64"] = newJNull()
    expect ValueError: validateFixtureData(malformed, reference)

  test "Reject explicit mean fixtures with invalid scalar, axis or rejection contracts":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.mean_dtype":
        malformed["cases"][index]["expected"]["bool"]["default"]["all"]["error"] = %"success"
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.mean_dtype":
        malformed["cases"][index]["expected"]["int8"]["axes"]["0"]["float32"]["shape"] = %*[3]
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    for index in 0 ..< malformed["cases"].len:
      if malformed["cases"][index]["id"].getStr == "tensor.mean_dtype":
        malformed["cases"][index]["expected"]["float64"]["all"]["float64"]["shape"] = %*[1]
    expect ValueError: validateFixtureData(malformed, reference)


  test "Reject explicit sum fixtures with invalid scalar, axis or dtype contracts":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var sumCaseIndex = -1
    for index in 0 ..< fixtures["cases"].len:
      if fixtures["cases"][index]["id"].getStr == "tensor.sum_dtype":
        sumCaseIndex = index
    doAssert sumCaseIndex >= 0

    var malformed = duplicate(fixtures)
    malformed["cases"][sumCaseIndex]["expected"]["int8"]["all"]["float32"]["shape"] = %*[1]
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    malformed["cases"][sumCaseIndex]["expected"]["int8"]["axes"]["0"]["float32"]["shape"] = %*[3]
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    malformed["cases"][sumCaseIndex]["expected"]["int8"]["all"]["float32"]["dtype"] = %"float64"
    expect ValueError: validateFixtureData(malformed, reference)


  test "Explicit product dtypes cast before multiplying and preserve reduction ordering":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var productCase = newJNull()
    for caseData in fixtures["cases"]:
      if caseData["id"].getStr == "tensor.prod_dtype": productCase = caseData
    doAssert productCase.kind == JObject
    check productCase["atol"].getFloat == 1e-6
    check productCase["rtol"].getFloat == 1e-6
    for name in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64", "float32", "float64", "float64_precision", "uint64_upper", "packed_odd",
        "transpose", "offset", "reverse", "broadcast", "scalar", "empty", "float64_overflow",
        "float64_underflow", "float64_zero_nan", "uint64_upper_zero", "lane_range", "early_zero4",
        "zero_columns", "scalar_negative_zero", "cast_before_product"]:
      check productCase["inputs"].hasKey(name)

    let expected = productCase["expected"]
    let actual = runCase(productCase)
    check expected["float64_overflow"]["all"]["float32"]["values"][0].getStr == "+inf"
    check expected["float64_underflow"]["all"]["float32"]["values"][0].getFloat == 0.0
    check expected["float64_zero_nan"]["all"]["float32"]["values"][0].getStr == "nan"
    check expected["float64_zero_nan"]["all"]["float64"]["values"][0].getFloat == 0.0
    check expected["uint64_upper_zero"]["all"]["float32"]["values"][0].getStr == "nan"
    check expected["uint64_upper_zero"]["all"]["float64"]["values"][0].getFloat == 0.0
    check expected["lane_range"]["all"]["float32"]["values"][0].getFloat == 0.9999946355819702
    check expected["lane_range"]["axes"]["0"]["float32"]["values"][0].getFloat ==
      0.9999946355819702
    check expected["lane_range"]["all"]["float64"]["values"][0].getFloat == 1.0
    check expected["early_zero4"]["all"]["float32"]["values"][0].getFloat == 0.0
    check expected["zero_columns"]["all"]["float32"]["values"][0].getStr == "nan"
    check expected["zero_columns"]["all"]["float64"]["values"][0].getStr == "nan"
    check expected["zero_columns"]["axes"]["0"]["float32"]["values"][0].getStr == "nan"
    check expected["zero_columns"]["axes"]["0"]["float64"]["values"] == %*[0.0, 0.0, 0.0, 0.0, 0.0]
    check expected["cast_before_product"]["all"]["float32"]["values"][0].getFloat == 1.0
    check expected["cast_before_product"]["all"]["float64"]["values"][0].getFloat > 1.0
    for dtype in ["float32", "float64"]:
      for axis in ["0", "-1"]:
        check isNegativeZero(expected["scalar_negative_zero"]["all"][dtype]["values"][0])
        check isNegativeZero(expected["scalar_negative_zero"]["axes"][axis][dtype]["values"][0])
        check isNegativeZero(actual["scalar_negative_zero"]["all"][dtype]["values"][0])
        check isNegativeZero(actual["scalar_negative_zero"]["axes"][axis][dtype]["values"][0])

  test "Reject malformed explicit product dtype fixtures":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var productCaseIndex = -1
    for index in 0 ..< fixtures["cases"].len:
      if fixtures["cases"][index]["id"].getStr == "tensor.prod_dtype":
        productCaseIndex = index
    doAssert productCaseIndex >= 0

    var malformed = duplicate(fixtures)
    malformed["cases"][productCaseIndex]["inputs"].delete("cast_before_product")
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    malformed["cases"][productCaseIndex]["inputs"]["zero_columns"]["axes"] = %*[0, -1]
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    malformed["cases"][productCaseIndex]["expected"]["cast_before_product"]["axes"]["0"]["float32"]["dtype"] = %"float64"
    expect ValueError: validateFixtureData(malformed, reference)

  test "Explicit cumulative dtype scans cover output dtypes, direction and edge values":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var cumsumCase = newJNull()
    var cumprodCase = newJNull()
    for caseData in fixtures["cases"]:
      case caseData["id"].getStr
      of "tensor.cumsum_dtype": cumsumCase = caseData
      of "tensor.cumprod_dtype": cumprodCase = caseData
      else: discard
    doAssert cumsumCase.kind == JObject and cumprodCase.kind == JObject
    for scanCase in [cumsumCase, cumprodCase]:
      for name in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
          "uint64", "float32", "float64", "float32_wide_scan", "float64_overflow",
          "float64_underflow", "nonfinite", "scalar_negative_zero", "vector_negative_zero"]:
        check scanCase["inputs"].hasKey(name)

    let cumsumExpected = cumsumCase["expected"]
    check cumsumExpected["float64_precision"]["axes"]["0"]["float32"]["forward"]["values"][0].getFloat ==
      16777216.0
    check cumsumExpected["float32_wide_scan"]["axes"]["0"]["float64"]["forward"]["values"][1].getFloat ==
      16777217.0
    check cumsumExpected["float32_wide_scan"]["axes"]["0"]["float64"]["forward"]["values"][2].getFloat ==
      1.0
    check cumsumExpected["nonfinite"]["axes"]["0"]["float64"]["forward"]["values"][1].getStr ==
      "+inf"
    check cumsumExpected["nonfinite"]["axes"]["0"]["float64"]["forward"]["values"][2].getStr ==
      "nan"

    let cumsumActual = runCase(cumsumCase)
    check cumsumActual["float32_wide_scan"]["axes"]["0"]["float64"]["forward"]["values"][1].getFloat ==
      16777217.0
    check cumsumActual["float32_wide_scan"]["axes"]["0"]["float64"]["forward"]["values"][2].getFloat ==
      1.0
    check cumsumActual["float64_precision"]["axes"]["0"]["float32"]["forward"]["values"][0].getFloat ==
      16777216.0
    check cumsumActual["float64_precision"]["axes"]["0"]["float32"]["forward"]["values"][1].getFloat ==
      0.0


    let cumprodExpected = cumprodCase["expected"]
    check cumprodExpected["float64_overflow"]["axes"]["0"]["float32"]["forward"]["values"][1].getStr ==
      "+inf"
    check cumprodExpected["float64_overflow"]["axes"]["0"]["float32"]["forward"]["values"][2].getFloat >
      9.9e19
    check cumprodExpected["float64_overflow"]["axes"]["0"]["float32"]["forward"]["values"][2].getFloat <
      1.1e20
    check cumprodExpected["float64_underflow"]["axes"]["0"]["float32"]["forward"]["values"][1].getFloat ==
      0.0
    check runCase(cumprodCase)["float64_underflow"]["axes"]["0"]["float32"]["forward"]["values"][2].getFloat >
      0.0
    check cumprodExpected["float64_underflow"]["axes"]["0"]["float32"]["forward"]["values"][2].getFloat >
      0.0

    for scanCase in [cumsumCase, cumprodCase]:
      let actual = runCase(scanCase)
      let operation = if scanCase["id"].getStr == "tensor.cumsum_dtype": "cumsum" else: "cumprod"
      for dtype in ["float32", "float64"]:
        for axis in ["0", "-1"]:
          let expectedScalar = scanCase["expected"]["scalar_negative_zero"]["axes"][axis][dtype]
          let actualScalar = actual["scalar_negative_zero"]["axes"][axis][dtype]
          check isNegativeZero(expectedScalar["forward"]["values"][0])
          check isNegativeZero(expectedScalar["reverse"]["values"][0])
          check isNegativeZero(actualScalar["forward"]["values"][0])
          check isNegativeZero(actualScalar["reverse"]["values"][0])
          for direction in ["forward", "reverse"]:
            let expectedVector =
              scanCase["expected"]["vector_negative_zero"]["axes"][axis][dtype][direction]["values"][0]
            let actualVector = actual["vector_negative_zero"]["axes"][axis][dtype][direction]["values"][0]
            check expectedVector.getFloat == 0.0
            check actualVector.getFloat == 0.0
            if operation == "cumsum":
              check not isNegativeZero(expectedVector)
              check not isNegativeZero(actualVector)
            else:
              check isNegativeZero(expectedVector)
              check isNegativeZero(actualVector)

  test "Reject malformed explicit cumulative dtype scan fixtures":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    for operation in ["tensor.cumsum_dtype", "tensor.cumprod_dtype"]:
      var scanCaseIndex = -1
      for index in 0 ..< fixtures["cases"].len:
        if fixtures["cases"][index]["id"].getStr == operation:
          scanCaseIndex = index
      doAssert scanCaseIndex >= 0

      var malformed = duplicate(fixtures)
      malformed["cases"][scanCaseIndex]["expected"]["int8"]["axes"]["0"]["float32"]["reverse"] =
        newJNull()
      expect ValueError: validateFixtureData(malformed, reference)

      malformed = duplicate(fixtures)
      malformed["cases"][scanCaseIndex]["expected"]["int8"]["axes"]["0"]["float32"]["forward"]["shape"] =
        %*[3]
      expect ValueError: validateFixtureData(malformed, reference)

      malformed = duplicate(fixtures)
      malformed["cases"][scanCaseIndex]["expected"]["int8"]["axes"]["0"]["float32"]["forward"]["dtype"] =
        %"float64"
      expect ValueError: validateFixtureData(malformed, reference)

      malformed = duplicate(fixtures)
      malformed["cases"][scanCaseIndex]["inputs"]["nonfinite"]["nonfinite"][0]["value"] = %"nan"
      expect ValueError: validateFixtureData(malformed, reference)


  test "Float64 reductions cover CPU layouts, precision, and seeded VJPs":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var float64CaseIndex = -1
    for index in 0 ..< fixtures["cases"].len:
      if fixtures["cases"][index]["id"].getStr == "tensor.float64_reductions":
        float64CaseIndex = index
    doAssert float64CaseIndex >= 0
    let float64Case = fixtures["cases"][float64CaseIndex]
    check float64Case["atol"].getFloat == 1e-12
    check float64Case["rtol"].getFloat == 1e-12
    for view in ["transpose", "offset", "reverse", "broadcast"]:
      check float64Case["inputs"]["reductions"].hasKey(view)
      check float64Case["inputs"]["scans"].hasKey(view)
    for edge in ["scalar", "empty"]:
      check float64Case["inputs"]["reductions"].hasKey(edge)
      check float64Case["inputs"]["scans"].hasKey(edge)
    check float64Case["inputs"]["reductions"]["packed_odd"]["shape"] == %*[3, 5]
    check float64Case["expected"]["reductions"]["packed_odd"]["sum"]["all"]["dtype"].getStr ==
      "float64"
    check float64Case["expected"]["scans"]["scan_precision"]["cumsum"]["0"]["values"].len == 5
    check float64Case["expected"]["reductions"]["empty"]["mean"]["all"]["values"][0].getStr ==
      "nan"
    check float64Case["expected"]["reduction_vjps"]["zero_infinity"][0]["output"]["values"][0].getStr ==
      "nan"
    check float64Case["expected"]["reduction_vjps"]["zero_nan"][0]["output"]["values"][0].getStr ==
      "nan"
    check float64Case["expected"]["reduction_vjps"]["infinity"][0]["output"]["values"][0].getStr ==
      "+inf"
    check float64Case["expected"]["reduction_vjps"]["nan"][0]["output"]["values"][0].getStr ==
      "nan"

    var malformed = duplicate(fixtures)
    let malformedAxes: JsonNode =
      malformed["cases"][float64CaseIndex]["inputs"]["reductions"]["packed_odd"]["axes"]
    malformedAxes.elems[2] = %0
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    let malformedSpecials: JsonNode =
      malformed["cases"][float64CaseIndex]["inputs"]["reduction_vjps"]["zero_nan"]["nonfinite"]
    malformedSpecials.elems[0]["index"] = %3
    expect ValueError: validateFixtureData(malformed, reference)

    malformed = duplicate(fixtures)
    malformed["cases"][float64CaseIndex]["expected"]["scans"]["packed_odd"]["cumprod"][
      "1"]["dtype"] = %"float32"
    expect ValueError: validateFixtureData(malformed, reference)


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

  test "Nonfinite scan seeds reject out-of-bounds writes and invalid values":
    let fixtures = loadFixtures(parityRoot & "fixtures.json")
    var scanCase = -1
    for index in 0 ..< fixtures["cases"].len:
      if fixtures["cases"][index]["id"].getStr == "autograd.cumsum_gradient":
        scanCase = index
    doAssert scanCase >= 0
    let seedSize = fixtures["cases"][scanCase]["inputs"]["nonfinite_seed"][
      "operations"][0]["seed"]["values"].len
    for index in [-1, seedSize]:
      var malformed = duplicate(fixtures)
      let operation = malformed["cases"][scanCase]["inputs"]["nonfinite_seed"]["operations"][0]
      operation["seed_nonfinite"][0]["index"] = %index
      expect ValueError: validateFixtureData(malformed, reference)
    var malformed = duplicate(fixtures)
    let operation = malformed["cases"][scanCase]["inputs"]["nonfinite_seed"]["operations"][0]
    operation["seed_nonfinite"][0]["value"] = %"not-a-number"
    expect ValueError: validateFixtureData(malformed, reference)

  test "Metal reference data requires every subcase and exact tensor metadata":
    let fixtures = loadMetalFixtures(parityRoot & "metal_fixtures.json")
    var malformed = duplicate(fixtures)
    malformed["cases"] = withoutFirst(malformed["cases"])
    expect ValueError: validateMetalFixtures(malformed)
    malformed = duplicate(fixtures)
    malformed["cases"][1]["id"] = malformed["cases"][0]["id"]
    expect ValueError: validateMetalFixtures(malformed)
    malformed = duplicate(fixtures)
    malformed["cases"][0]["atol"] = %(-1.0)
    expect ValueError: validateMetalFixtures(malformed)
    malformed = duplicate(fixtures)
    malformed["cases"][0]["expected"]["base"]["dtype"] = %"float64"
    expect ValueError: validateMetalFixtures(malformed)
    malformed = duplicate(fixtures)
    malformed["cases"][0]["expected"]["base"]["numel"] = %7
    expect ValueError: validateMetalFixtures(malformed)

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
