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

## Validates pinned artifacts and compares native observations offline.
import std/[json, math, sets, strutils, tables]

proc invalid(message: string) {.noreturn.} =
  raise newException(ValueError, message)

proc loadReference*(path = "tests/parity/reference.json"): JsonNode
proc validateAppleRequirements*(data, reference, inventory: JsonNode)

proc loadAppleRequirements*(path: string): JsonNode =
  try:
    result = parseFile(path)
  except CatchableError as e:
    invalid("cannot read Apple requirements " & path & ": " & e.msg)
  let reference = loadReference()
  var inventory: JsonNode
  try:
    inventory = parseFile("tests/parity/inventory.json")
  except CatchableError as e:
    invalid("cannot read inventory while validating Apple requirements: " & e.msg)
  validateAppleRequirements(result, reference, inventory)

type
  Difference* = object
    path*: string
    expected*, actual*: JsonNode
  AppleExecution* = enum
    aePass, aeMismatch, aeError
  AppleObservation* = object
    id*, profile*: string
    execution*: AppleExecution
    actual*, error*, evidence*: JsonNode
    differences*: seq[Difference]

const
  RequiredCaseIds = ["tensor.transpose_alias", "tensor.empty_shape", "tensor.scalar_metadata",
    "tensor.broadcast_same_rank", "tensor.broadcast_leading_dims", "tensor.matmul_2d",
    "autograd.mean_gradient", "loss.mse_mean", "autograd.no_grad_exception",
    "autograd.seeded_vjp", "autograd.retained_vjp", "autograd.independent_graphs",
    "autograd.chunk_seed", "autograd.slice_overlap", "autograd.index_select",
    "autograd.mask_select", "autograd.axis_mean", "autograd.reduction_scalars",
    "autograd.axis_options", "autograd.broadcast_multiply", "autograd.broadcast_divide",
    "loss.mse_mean_broadcast", "loss.mse_sum_broadcast", "loss.mse_none_broadcast",
    "autograd.matmul_ranks", "autograd.singleton_views", "tensor.elementwise_shapes",
    "autograd.unary_math"]
  RequiredCaseAreas = ["api:tensor_view", "api:tensors", "api:tensors", "api:torch", "api:torch",
    "api:torch", "api:autograd", "api:nn.functional", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:nn.functional", "api:nn.functional", "api:nn.functional",
    "api:autograd", "api:autograd", "api:torch", "api:autograd"]
  MetalIds = ["apple.metal.device_dispatch", "apple.metal.dtypes", "apple.metal.tensor_views",
    "apple.metal.tensor_math", "apple.metal.autograd", "apple.metal.optimizer_sgd",
    "apple.metal.queue_lifetime", "apple.metal.memory_profiling", "apple.metal.rng",
    "apple.metal.custom_kernels"]
  CoreIds = ["apple.coreai.model_functions", "apple.coreai.ndarray_descriptors",
    "apple.coreai.strided_layout", "apple.coreai.metal_buffer_interop", "apple.coreai.stream_lifetime",
    "apple.coreai.stateful_execution", "apple.coreai.concurrency", "apple.coreai.compute_placement"]
  AllowedStatuses = ["partial", "missing", "unreviewed"]

proc require(condition: bool, message: string) =
  if not condition: invalid(message)


proc field(node: JsonNode, key: string): JsonNode =
  if node.kind != JObject or not node.hasKey(key): invalid("missing object field: " & key)
  node[key]

proc text(node: JsonNode, name: string): string =
  if node.kind != JString: invalid(name & " must be a string")
  node.getStr

proc arrayIds(node: JsonNode, key: string): HashSet[string] =
  result = initHashSet[string]()
  let items = field(node, key)
  if items.kind != JArray: invalid(key & " must be an array")
  for item in items:
    let id = text(field(item, "id"), key & " id")
    if id in result: invalid("duplicate " & key & " id: " & id)
    result.incl(id)

proc loadReference*(path: string): JsonNode =
  try:
    result = parseFile(path)
  except CatchableError as e:
    invalid("cannot read reference " & path & ": " & e.msg)
  require(result.kind == JObject and result.len == 5 and field(result, "format").kind == JInt and
    field(result, "format").getInt == 1, "unsupported reference format or fields")
  for key in ["format", "torch", "source_commit", "python", "apple"]:
    require(result.hasKey(key), "missing reference field: " & key)
  for key in ["torch", "source_commit", "python"]: discard text(field(result, key), key)
  let apple = field(result, "apple")
  require(apple.kind == JObject and apple.len == 7, "invalid Apple reference fields")
  for key in ["xcode", "xcode_build", "macos_sdk", "swift", "minimum_macos", "architecture", "metal_api"]:
    discard text(field(apple, key), "apple." & key)
  require(result["torch"].getStr == "2.15.0" and result["python"].getStr == "3.13.18" and
    result["source_commit"].getStr == "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457", "reference pin mismatch")
  require(apple == %*{"xcode": "27.1", "xcode_build": "27A9275", "macos_sdk": "27.0",
    "swift": "6.4", "minimum_macos": "27.0", "architecture": "arm64", "metal_api": "4"},
    "Apple reference pin mismatch")

proc requireSameReference(node, reference: JsonNode, label: string) =
  require(field(node, "reference") == reference, label & " reference identity mismatch")

proc validateInventory*(inventory, reference: JsonNode) =
  require(field(inventory, "format").kind == JInt and inventory["format"].getInt == 1, "unsupported inventory format")
  requireSameReference(inventory, reference, "inventory")
  require(text(field(inventory, "granularity"), "inventory.granularity") ==
    "api_areas_aten_schemas_autograd_rules_and_mps_exports", "inventory granularity mismatch")
  let records = field(inventory, "records")
  require(records.kind == JArray and records.len > 0, "inventory records missing")
  var ids = initHashSet[string]()
  var kinds = initHashSet[string]()
  for rec in records:
    let id = text(field(rec, "id"), "inventory id")
    let kind = text(field(rec, "kind"), "inventory kind")
    require(kind in ["api_area", "aten_schema", "autograd_rule", "mps_export"],
      "unknown inventory kind " & kind)
    require(id.len > 0 and id notin ids, "empty or duplicate inventory id: " & id)
    ids.incl(id); kinds.incl(kind)
    discard text(field(rec, "declaration"), "inventory declaration")
    let source = field(rec, "source")
    for key in ["path", "blob"]: discard text(field(source, key), "inventory source." & key)
    require(source["blob"].getStr.len == 40 and field(source, "line").kind == JInt and
      source["line"].getInt > 0, "invalid inventory source location")
  for kind in ["api_area", "aten_schema", "autograd_rule", "mps_export"]:
    require(kind in kinds, "inventory lacks kind " & kind)
proc requiredText(node: JsonNode, keys: openArray[string], label: string): string =
  for key in keys:
    if node.kind == JObject and node.hasKey(key): return text(node[key], label)
  invalid("missing " & label)

proc validateAppleRequirements*(data, reference, inventory: JsonNode) =
  require(field(data, "format").kind == JInt and data["format"].getInt == 1, "unsupported Apple requirements format")
  requireSameReference(data, reference, "Apple requirements")
  let profiles = field(data, "profiles")
  require(profiles.kind == JArray and profiles.len == 2, "exactly two Apple profiles required")
  var profileMap = initTable[string, JsonNode]()
  for profile in profiles:
    let id = text(field(profile, "id"), "profile id")
    require(id in ["apple-metal", "apple-coreai"] and id notin profileMap, "invalid or duplicate Apple profile: " & id)
    profileMap[id] = profile
    require(text(field(profile, "priority"), "profile priority") == "required", "Apple profiles must be required")
    require(field(profile, "baseline") == field(reference, "apple"), "Apple profile baseline mismatch")
  for (id, backend, referenceBackend, role) in [("apple-metal", "metal", "torch.mps", "tensor_training"),
      ("apple-coreai", "coreai", "apple_sdk", "model_inference")]:
    require(profileMap.hasKey(id), "missing Apple profile " & id)
    let p = profileMap[id]
    require(text(field(p, "native_backend"), "native_backend") == backend and
      text(field(p, "reference_backend"), "reference_backend") == referenceBackend and
      text(field(p, "role"), "role") == role, "Apple profile contract mismatch: " & id)
  let reqs = field(data, "requirements")
  require(reqs.kind == JArray and reqs.len == 18, "exactly 18 Apple requirements required")
  var reqIds = initHashSet[string]()
  for req in reqs:
    let id = text(field(req, "id"), "requirement id")
    require(id notin reqIds, "duplicate Apple requirement " & id); reqIds.incl(id)
    let profile = text(field(req, "profile"), "requirement profile")
    require(profile in ["apple-metal", "apple-coreai"], "invalid requirement profile " & profile)
    require((id in MetalIds and profile == "apple-metal") or
      (id in CoreIds and profile == "apple-coreai"), "requirement/profile mismatch: " & id)
    require(text(field(req, "contract"), "contract").len > 0 and text(field(req, "scenario"), "scenario").len > 0,
      "Apple requirement needs contract and concrete scenario: " & id)
    let sources = field(req, "sources")
    require(sources.kind == JArray and sources.len > 0, "Apple requirement sources missing: " & id)
    for source in sources:
      require(source.kind == JObject, "Apple source must be an object: " & id)
      if source.hasKey("sha256") or source.hasKey("digest"):
        discard requiredText(source, ["url", "documentation_url"], "Apple documentation URL")
        discard text(field(source, "symbol"), "Apple source.symbol")
        discard requiredText(source, ["interface", "sdk_interface", "interface_path"], "Apple SDK interface path")
        let digest = requiredText(source, ["sha256", "digest"], "Apple SDK interface digest")
        require(digest.len == 64, "invalid Apple SDK interface digest: " & id)
      else:
        require(text(field(source, "commit"), "PyTorch source.commit") == reference["source_commit"].getStr,
          "PyTorch source commit mismatch: " & id)
        discard text(field(source, "commit"), "PyTorch source.commit")
        discard text(field(source, "path"), "PyTorch source.path")
        discard text(field(source, "blob"), "PyTorch source.blob")
        require(field(source, "line").kind == JInt and source["line"].getInt > 0,
          "invalid PyTorch source line: " & id)
  for id in MetalIds: require(id in reqIds, "missing Apple requirement " & id)
  for id in CoreIds: require(id in reqIds, "missing Apple requirement " & id)
  let exports = arrayIds(inventory, "records")
  var linked = initHashSet[string]()
  for req in reqs:
    for rid in field(req, "reference_ids"):
      let refId = text(rid, "reference id")
      require(refId in exports, "unknown Apple reference id " & refId)
      if refId.startsWith("mps:"): linked.incl(refId)
  for rec in inventory["records"]:
    if field(rec, "kind").getStr == "mps_export":
      require(text(field(rec, "id"), "MPS id") in linked, "unlinked MPS export " & rec["id"].getStr)

proc validateCapabilities*(capabilities, inventory, requirements: JsonNode) =
  require(field(capabilities, "format").kind == JInt and capabilities["format"].getInt == 1,
    "unsupported capabilities format")
  requireSameReference(capabilities, field(inventory, "reference"), "capabilities")
  let apiIds = block:
    var s = initHashSet[string]()
    for rec in inventory["records"]:
      if rec["kind"].getStr == "api_area": s.incl(rec["id"].getStr)
    s
  var rows = initHashSet[string]()
  let areas = field(capabilities, "areas")
  require(areas.kind == JArray, "capabilities areas must be an array")
  for row in areas:
    let id = text(field(row, "id"), "capability id")
    require(id in apiIds and id notin rows, "unknown or duplicate capability area " & id); rows.incl(id)
    let status = text(field(row, "status"), "capability status")
    require(status in AllowedStatuses, "invalid capability status " & status)
    require(field(row, "evidence").kind == JArray and field(row, "gaps").kind == JArray, "invalid capability evidence/gaps")
    for evidence in row["evidence"]:
      discard text(field(evidence, "path"), "capability evidence path")
      discard text(field(evidence, "symbol"), "capability evidence symbol")
    for gap in row["gaps"]: discard text(gap, "capability gap")
  require(rows == apiIds, "capability area coverage incomplete")
  let reqIds = arrayIds(requirements, "requirements")
  let annotations = field(capabilities, "apple_requirements")
  require(annotations.kind == JArray, "apple_requirements annotations must be array")
  var got = initHashSet[string]()
  for row in annotations:
    let id = text(field(row, "id"), "Apple capability id")
    require(id in reqIds and id notin got, "unknown or duplicate Apple capability " & id); got.incl(id)
    require(text(field(row, "status"), "Apple status") in AllowedStatuses, "invalid Apple implementation status")
    require(field(row, "evidence").kind == JArray and field(row, "gaps").kind == JArray,
      "invalid Apple capability evidence/gaps")
    for evidence in row["evidence"]:
      discard text(field(evidence, "path"), "Apple capability evidence path")
      discard text(field(evidence, "symbol"), "Apple capability evidence symbol")
    for gap in row["gaps"]: discard text(gap, "Apple capability gap")
  require(got == reqIds, "Apple capability coverage incomplete")

proc expectTensor(node: JsonNode, metadataOnly: bool) =
  let shape = field(node, "shape")
  require(shape.kind == JArray, "tensor shape must be array")
  var n = 1
  for d in shape:
    require(d.kind == JInt and d.getInt >= 0, "tensor dimensions must be nonnegative integers")
    if d.getInt == 0: n = 0
    elif n != 0:
      require(n <= high(int) div d.getInt, "tensor shape overflows")
      n *= d.getInt
  require(field(node, "numel").kind == JInt and node["numel"].getInt == n, "tensor numel/shape mismatch")
  require(text(field(node, "dtype"), "tensor dtype") == "float64" and text(field(node, "device"), "tensor device") == "cpu",
    "fixture tensors must be float64 CPU")
  if metadataOnly:
    require(not node.hasKey("values"), "metadata-only tensor must omit values")
  else:
    let values = field(node, "values")
    require(values.kind == JArray and values.len == n, "tensor values count does not match shape")
    for value in values:
      require((value.kind in {JInt, JFloat} and
        classify(value.getFloat) notin {fcNan, fcInf, fcNegInf}) or
        (value.kind == JString and value.getStr in ["nan", "+inf", "-inf"]),
        "invalid tensor observation value")


proc expectInputTensor(node: JsonNode) =
  let shape = field(node, "shape")
  require(shape.kind == JArray, "input shape must be array")
  var n = 1
  for dim in shape:
    require(dim.kind == JInt and dim.getInt >= 0, "input dimensions must be nonnegative integers")
    let d = dim.getInt
    if d == 0: n = 0
    elif n != 0:
      require(n <= high(int) div d, "input shape overflows")
      n *= d
  let values = field(node, "values")
  require(values.kind == JArray and values.len == n, "input value count does not match shape")
  for value in node["values"]:
    require(value.kind in {JInt, JFloat} and classify(value.getFloat) notin {fcNan, fcInf, fcNegInf},
      "fixture inputs must contain finite numbers")
proc validateFixtureData*(data, reference: JsonNode) =
  require(field(data, "format").kind == JInt and data["format"].getInt == 1, "unsupported fixtures format")
  requireSameReference(data, reference, "fixtures")
  let oracle = field(data, "oracle")
  let torchVersion = text(field(oracle, "torch"), "oracle.torch")
  let normalized = torchVersion.split('+')[0]
  require(normalized == reference["torch"].getStr and not normalized.contains('a') and not normalized.contains("rc"),
    "fixture Torch version does not match pin")
  require(text(field(oracle, "python"), "oracle.python") == reference["python"].getStr and
    text(field(oracle, "device"), "oracle.device") == "cpu", "fixture oracle provenance mismatch")
  discard text(field(oracle, "git_commit"), "oracle.git_commit")
  let cases = field(data, "cases")
  require(cases.kind == JArray and cases.len == RequiredCaseIds.len, "exactly nine fixtures required")
  var ids = initHashSet[string]()
  for c in cases:
    let id = text(field(c, "id"), "case id")
    let idx = RequiredCaseIds.find(id)
    require(idx >= 0 and id notin ids, "unknown or duplicate fixture case " & id); ids.incl(id)
    require(text(field(c, "area"), "case area") == RequiredCaseAreas[idx], "fixture area mismatch for " & id)
    for key in ["atol", "rtol"]:
      let v = field(c, key)
      require(v.kind in {JInt, JFloat} and v.getFloat >= 0 and classify(v.getFloat) notin {fcNan, fcInf, fcNegInf},
        "invalid fixture tolerance " & key)
    let inputs = field(c, "inputs")
    case id
    of "tensor.transpose_alias":
      expectInputTensor(field(inputs, "a"))
      let mutation = field(inputs, "mutation")
      require(field(mutation, "index") == %*[0, 1], "transpose fixture mutation index must be [0,1]")
      require(field(mutation, "value").kind in {JInt, JFloat}, "transpose mutation value must be numeric")
    of "tensor.empty_shape", "tensor.scalar_metadata":
      let shape = field(inputs, "shape")
      require(shape.kind == JArray, "shape input must be array")
      if id == "tensor.empty_shape":
        require(shape == %*[2, 0, 3], "empty-shape fixture input must be [2,0,3]")
      else:
        require(shape.len == 0, "scalar fixture input must have empty shape")
        require(not inputs.hasKey("values"), "scalar metadata case must not provide storage values")
      var n = 1
      for d in shape:
        require(d.kind == JInt and d.getInt >= 0, "input dimensions must be nonnegative integers")
        if d.getInt == 0: n = 0
        elif n != 0:
          require(n <= high(int) div d.getInt, "input shape overflows")
          n *= d.getInt
    of "tensor.broadcast_same_rank", "tensor.broadcast_leading_dims", "tensor.matmul_2d":
      expectInputTensor(field(inputs, "a")); expectInputTensor(field(inputs, "b"))
    of "autograd.mean_gradient": expectInputTensor(field(inputs, "x"))
    of "loss.mse_mean": expectInputTensor(field(inputs, "prediction")); expectInputTensor(field(inputs, "target"))
    of "autograd.no_grad_exception": expectInputTensor(field(inputs, "x"))
    else: discard
    let expected = field(c, "expected")
    case id
    of "tensor.transpose_alias": expectTensor(field(expected, "base"), false); expectTensor(field(expected, "view"), false)
    of "tensor.empty_shape": expectTensor(field(expected, "result"), false)
    of "tensor.scalar_metadata": expectTensor(field(expected, "result"), true)
    of "tensor.broadcast_same_rank", "tensor.broadcast_leading_dims", "tensor.matmul_2d": expectTensor(field(expected, "result"), false)
    of "autograd.mean_gradient": expectTensor(field(expected, "grad"), false)
    of "loss.mse_mean": expectTensor(field(expected, "loss"), false); expectTensor(field(expected, "grad"), false)
    of "autograd.no_grad_exception": require(field(expected, "tracking_after_exception").kind == JBool, "no-grad expected value must be bool")
    else: discard
  require(ids.len == RequiredCaseIds.len, "missing required fixture case")

proc loadFixtures*(path: string): JsonNode =
  try:
    result = parseFile(path)
  except CatchableError as e:
    invalid("cannot read fixtures " & path & ": " & e.msg)
  validateFixtureData(result, loadReference())

proc validateMetalTensor(node: JsonNode) =
  let shape = field(node, "shape")
  require(shape.kind == JArray, "Metal fixture shape must be an array")
  var size = 1
  for dimension in shape:
    require(dimension.kind == JInt and dimension.getInt >= 0, "Invalid Metal dimension")
    let dim = dimension.getInt
    require(dim == 0 or size <= high(int) div dim, "Metal fixture shape overflow")
    size *= dim
  require(field(node, "numel").kind == JInt and node["numel"].getInt == size,
    "Metal fixture numel mismatch")
  require(field(node, "dtype").getStr == "float32" and field(node, "device").getStr in ["mps", "mps:0"],
    "Metal reference observations must be float32 on MPS")
  let values = field(node, "values")
  require(values.kind == JArray and values.len == size, "Metal fixture values mismatch")
  for value in values:
    require((value.kind in {JInt, JFloat} and classify(value.getFloat) notin {fcNan, fcInf, fcNegInf}) or
      (value.kind == JString and value.getStr in ["nan", "+inf", "-inf"]), "Invalid Metal fixture value")

proc validateMetalFixtures*(data: JsonNode) =
  require(field(data, "format") == %1, "Unsupported Metal fixture format")
  let reference = loadReference()
  requireSameReference(data, reference, "Metal fixtures")
  let oracle = field(data, "oracle")
  require(field(oracle, "torch").getStr.split('+')[0] == reference["torch"].getStr and
    field(oracle, "python").getStr == reference["python"].getStr and
    field(oracle, "device").getStr == "mps", "Metal oracle provenance mismatch")
  let ids = ["tensor.transpose_alias", "tensor.broadcast_same_rank", "tensor.broadcast_leading_dims",
    "tensor.matmul_2d", "autograd.mean_gradient", "loss.mse_mean", "optim.sgd_step"]
  let cases = field(data, "cases")
  require(cases.kind == JArray and cases.len == ids.len, "Exactly seven Metal reference cases required")
  var seen = initHashSet[string]()
  for item in cases:
    let id = text(field(item, "id"), "Metal case id")
    require(id in ids and id notin seen, "Unknown or duplicate Metal case " & id)
    seen.incl(id)
    for key in ["atol", "rtol"]:
      let tolerance = field(item, key)
      require(tolerance.kind in {JInt, JFloat} and tolerance.getFloat >= 0 and
        classify(tolerance.getFloat) notin {fcNan, fcInf, fcNegInf}, "Invalid Metal tolerance")
    let inputs = field(item, "inputs")
    for key, value in inputs:
      if value.kind == JObject and value.hasKey("shape"): expectInputTensor(value)
    let expected = field(item, "expected")
    let outputs = case id
      of "tensor.transpose_alias": @["base", "view"]
      of "autograd.mean_gradient", "loss.mse_mean": @["loss", "grad"]
      of "optim.sgd_step": @["prediction", "loss", "grad", "weights_after_step", "grad_after_step", "grad_after_zero"]
      else: @["result"]
    require(expected.kind == JObject and expected.len == outputs.len, "Metal expected outputs incomplete")
    for key in outputs: validateMetalTensor(field(expected, key))

proc loadMetalFixtures*(path = "tests/parity/metal_fixtures.json"): JsonNode =
  try: result = parseFile(path)
  except CatchableError as error: invalid("Cannot read Metal fixtures: " & error.msg)
  validateMetalFixtures(result)

proc compareValue(expected, actual: JsonNode, path: string, atol, rtol: float64, diffs: var seq[Difference]) =
  template addDiff(e, a: JsonNode) = diffs.add Difference(path: path, expected: e, actual: a)
  if expected.kind != actual.kind:
    addDiff(expected, actual); return
  case expected.kind
  of JObject:
    var keys = initHashSet[string]()
    if expected.hasKey("shape"):
      keys.incl("shape")
      if actual.hasKey("shape"):
        compareValue(expected["shape"], actual["shape"], path & "/shape", 0, 0, diffs)
      else:
        diffs.add Difference(path: path & "/shape", expected: expected["shape"], actual: newJNull())
    for key, value in expected:
      if key == "shape": continue
      keys.incl(key)
      if not actual.hasKey(key):
        diffs.add Difference(path: path & "/" & key, expected: value, actual: newJNull())
      else: compareValue(value, actual[key], path & "/" & key, atol, rtol, diffs)
    for key, value in actual:
      if key notin keys: diffs.add Difference(path: path & "/" & key, expected: newJNull(), actual: value)
  of JArray:
    if expected.len != actual.len: addDiff(expected, actual)
    else:
      for i in 0 ..< expected.len: compareValue(expected[i], actual[i], path & "/" & $i, atol, rtol, diffs)
  of JInt:
    if expected != actual: addDiff(expected, actual)
  of JFloat:
    let e = expected.getFloat; let a = actual.getFloat
    if classify(e) in {fcNan, fcInf, fcNegInf} or classify(a) in {fcNan, fcInf, fcNegInf}:
      if e != a: addDiff(expected, actual)
    elif abs(a - e) > atol + rtol * abs(e): addDiff(expected, actual)
  else:
    if expected != actual: addDiff(expected, actual)

proc compareObservations*(expected, actual: JsonNode, atol, rtol: float64): seq[Difference] =
  compareValue(expected, actual, "", atol, rtol, result)
proc completedNativeInvocation*(evidence: JsonNode): bool =
  if evidence == nil or evidence.kind != JObject: return false
  if not evidence.hasKey("invocation_id") or evidence["invocation_id"].kind != JInt or
      evidence["invocation_id"].getBiggestInt <= 0: return false
  if not evidence.hasKey("completed") or evidence["completed"].kind != JBool or
      not evidence["completed"].getBool: return false
  evidence.hasKey("kind") and evidence["kind"].getStr in
    ["device_storage_control", "gpu_command", "coreai_inference"]

proc collectAppleCoverage*(requirements, capabilities: JsonNode;
                           profiles: openArray[string];
                           observations: openArray[AppleObservation];
                           requirementId = ""): JsonNode =
  result = newJArray()
  var known = initTable[string, string]()
  for req in field(requirements, "requirements"):
    known[text(field(req, "id"), "requirement id")] = text(field(req, "profile"), "profile")
  if requirementId.len > 0:
    require(known.hasKey(requirementId), "Unknown Apple requirement " & requirementId)
    require(profiles.len == 1 and profiles[0] == known[requirementId],
      "Requirement needs its explicit single Apple profile")
  var observed = initTable[string, AppleObservation]()
  for observation in observations:
    require(known.hasKey(observation.id), "Unknown Apple observation " & observation.id)
    require(observation.id notin observed, "Duplicate Apple observation " & observation.id)
    require(known[observation.id] == observation.profile and observation.profile in profiles,
      "Apple observation/profile conflict")
    require(requirementId.len == 0 or observation.id == requirementId, "Unselected Apple observation")
    if observation.execution == aePass:
      require(observation.actual != nil and observation.actual.kind != JNull and
        observation.differences.len == 0 and (observation.error == nil or observation.error.kind == JNull),
        "Pass requires actual values without differences/errors")
      require(completedNativeInvocation(observation.evidence), "Pass requires completed native invocation")
      let kind = observation.evidence["kind"].getStr
      if observation.profile == "apple-metal":
        require(kind == "gpu_command" or observation.id == "apple.metal.device_dispatch",
          "Metal computation requires completed GPU command")
        require(observation.evidence.hasKey("cpu_fallback_count") and
          observation.evidence["cpu_fallback_count"].getInt == 0, "Metal fallback cannot pass")
      else:
        require(kind == "coreai_inference", "Core AI requires completed inference")
    if observation.execution == aeMismatch:
      require(observation.differences.len > 0, "Mismatch requires differences")
    if observation.execution == aeError:
      require(observation.error != nil and observation.error.kind == JObject, "Error requires diagnostic")
    observed[observation.id] = observation
  for req in field(requirements, "requirements"):
    let profile = text(field(req, "profile"), "requirement profile")
    let id = text(field(req, "id"), "requirement id")
    if profile notin profiles or (requirementId.len > 0 and id != requirementId): continue
    var status = ""
    for annotation in field(capabilities, "apple_requirements"):
      if annotation["id"].getStr == id: status = text(field(annotation, "status"), "status")
    require(status in AllowedStatuses, "Missing or invalid Apple annotation " & id)
    var entry = %*{"id": id, "profile": profile, "implementation_status": status,
      "execution": "not_run", "actual": nil, "reason": "No native conformance observation.",
      "differences": [], "error": nil, "evidence": nil}
    if observed.hasKey(id):
      let observation = observed[id]
      entry["execution"] = %(case observation.execution
        of aePass: "pass"
        of aeMismatch: "mismatch"
        of aeError: "error")
      entry["reason"] = newJNull()
      entry["actual"] = if observation.actual == nil: newJNull() else: observation.actual
      entry["error"] = if observation.error == nil: newJNull() else: observation.error
      entry["evidence"] = if observation.evidence == nil: newJNull() else: observation.evidence
      for difference in observation.differences:
        entry["differences"].add(%*{"path": difference.path,
          "expected": difference.expected, "actual": difference.actual})
    result.add entry
