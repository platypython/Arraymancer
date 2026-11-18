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
    "autograd.unary_math", "autograd.activation_math", "tensor.reduction_edges",
    "autograd.product_gradient", "autograd.cumsum_gradient", "autograd.cumprod_gradient",
    "tensor.integer_scans", "tensor.integer_reductions", "tensor.float64_reductions",
    "tensor.mean_dtype"]
  RequiredCaseAreas = ["api:tensor_view", "api:tensors", "api:tensors", "api:torch", "api:torch",
    "api:torch", "api:autograd", "api:nn.functional", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:nn.functional", "api:nn.functional", "api:nn.functional",
    "api:autograd", "api:autograd", "api:torch", "api:autograd", "api:autograd", "api:torch",
    "api:autograd", "api:autograd", "api:autograd", "api:torch", "api:torch", "api:autograd",
    "api:torch"]
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

proc expectTensor(node: JsonNode, metadataOnly: bool, expectedDtype = "float64") =
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
  require(text(field(node, "dtype"), "tensor dtype") == expectedDtype and text(field(node, "device"), "tensor device") == "cpu",
    "fixture tensor dtype/device mismatch")
  if metadataOnly:
    require(not node.hasKey("values"), "metadata-only tensor must omit values")
  else:
    let values = field(node, "values")
    require(values.kind == JArray and values.len == n, "tensor values count does not match shape")
    for value in values:
      if expectedDtype == "int64":
        require(value.kind == JInt, "int64 fixture tensors require exact integer values")
      else:
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
proc expectIntegerWordsInput(node: JsonNode, label: string, allowWhole: bool) =
  let allowedDtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
  require(text(field(node, "dtype"), label & " dtype") in allowedDtypes,
    "unsupported " & label & " dtype")
  let shapeNode = field(node, "shape")
  require(shapeNode.kind == JArray, label & " shape must be an array")
  var shape: seq[int]
  var n = 1
  for dim in shapeNode:
    require(dim.kind == JInt and dim.getInt >= 0, label & " dimensions must be nonnegative integers")
    let size = dim.getInt
    shape.add size
    if size == 0: n = 0
    elif n != 0:
      require(n <= high(int) div size, label & " input shape overflows")
      n *= size
  let words = field(node, "words")
  require(words.kind == JArray and words.len == n, label & " words do not match shape")
  for word in words:
    require(word.kind == JInt, label & " words must be signed int64 JSON integers")
  let layout = text(field(node, "layout"), label & " layout")
  case layout
  of "identity": discard
  of "transpose": require(shape.len == 2, "transpose " & label & " input must have rank 2")
  of "offset": require(shape.len == 2 and shape[0] > 1, "offset " & label & " input shape is invalid")
  of "reverse": require(shape.len == 1, "reverse " & label & " input must have rank 1")
  of "broadcast": require(shape == @[1, 3], "broadcast " & label & " input must have shape [1,3]")
  else: invalid("unknown " & label & " layout")
  let axes = field(node, "axes")
  require(axes.kind == JArray and axes.len > 0, label & " axes are required")
  let dimensions = max(1, shape.len)
  var seen = initHashSet[string]()
  var hasWhole = false
  var hasPositiveAxis = false
  var hasNegativeAxis = false
  for axis in axes:
    if allowWhole and axis.kind == JNull:
      require(not hasWhole, label & " whole-reduction axis is duplicated")
      hasWhole = true
    else:
      require(axis.kind == JInt and axis.getInt >= -dimensions and axis.getInt < dimensions,
        label & " axis is out of bounds")
      let axisKey = $axis.getInt
      require(axisKey notin seen, label & " axis is duplicated")
      seen.incl(axisKey)
      hasPositiveAxis = hasPositiveAxis or axis.getInt > 0
      hasNegativeAxis = hasNegativeAxis or axis.getInt < 0
  if allowWhole:
    require(hasWhole, label & " whole-reduction axis is required")
    if shape.len > 1:
      require(hasPositiveAxis and hasNegativeAxis, label & " positive and negative axes are required")
    elif shape.len == 1:
      require(hasNegativeAxis and "0" in seen, label & " rank-one axes must include 0 and -1")

proc expectIntegerScanInput(node: JsonNode) =
  expectIntegerWordsInput(node, "integer scan", false)

proc expectIntegerReductionInput(node: JsonNode) =
  expectIntegerWordsInput(node, "integer reduction", true)

proc jsonShape(shape: openArray[int]): JsonNode =
  result = newJArray()
  for dimension in shape: result.add newJInt(dimension)

proc expectFloat64Layout(node: JsonNode, layout, label: string) =
  expectInputTensor(node)
  let shape = field(node, "shape")
  case layout
  of "identity": discard
  of "transpose":
    require(shape.len == 2, label & " transpose input must have rank 2")
  of "offset":
    require(shape.len == 2 and shape[0].getInt > 1,
      label & " offset input must have rank 2 and a sliceable first dimension")
  of "reverse":
    require(shape.len == 1, label & " reverse input must have rank 1")
  of "broadcast":
    require(shape == %*[1, 3], label & " broadcast input must have shape [1,3]")
  else:
    invalid("unknown " & label & " layout")

proc float64LogicalShape(node: JsonNode, layout: string): seq[int] =
  for dimension in field(node, "shape"): result.add dimension.getInt
  case layout
  of "transpose":
    swap(result[0], result[1])
  of "offset":
    let first = result[0]
    result[0] = result[1]
    result[1] = first - 1
  of "reverse": discard
  of "broadcast": result = @[4, 3]
  else: discard

proc expectFloat64Axes(node: JsonNode, allowWhole: bool, label: string) =
  let shape = field(node, "shape")
  let axes = field(node, "axes")
  require(axes.kind == JArray and axes.len > 0, label & " axes are required")
  let dimensions = max(1, shape.len)
  var seen = initHashSet[string]()
  var hasWhole = false
  var hasNonnegativeAxis = false
  var hasNegativeAxis = false
  for axis in axes:
    if allowWhole and axis.kind == JNull:
      require(not hasWhole, label & " whole-reduction axis is duplicated")
      hasWhole = true
    else:
      require(axis.kind == JInt and axis.getInt >= -dimensions and axis.getInt < dimensions,
        label & " axis is out of bounds")
      let key = $axis.getInt
      require(key notin seen, label & " axis is duplicated")
      seen.incl(key)
      hasNonnegativeAxis = hasNonnegativeAxis or axis.getInt >= 0
      hasNegativeAxis = hasNegativeAxis or axis.getInt < 0
  if allowWhole:
    require(hasWhole, label & " whole-reduction axis is required")
  if shape.len > 1:
    require(hasNonnegativeAxis and hasNegativeAxis,
      label & " must include nonnegative and negative axes")
  elif shape.len == 1:
    require("0" in seen and "-1" in seen, label & " rank-one axes must include 0 and -1")
  elif not allowWhole:
    require("0" in seen and "-1" in seen, label & " scalar scan axes must include 0 and -1")
proc expectNonUnitFloat64Seed(seed: JsonNode, label: string) =
  expectInputTensor(seed)
  var nonUnit = false
  for value in seed["values"]:
    nonUnit = nonUnit or (value.getFloat != 1.0 and value.getFloat != -1.0)
  require(nonUnit, label & " seed must contain a nonunit value")


proc float64ReductionShape(shape: seq[int], axis: JsonNode, keepdim: bool): seq[int] =
  if axis.kind == JNull or shape.len == 0: return @[]
  var normalized = axis.getInt
  if normalized < 0: normalized += max(1, shape.len)
  if keepdim:
    for dimension in shape: result.add dimension
    result[normalized] = 1
  else:
    for index, dimension in shape:
      if index != normalized: result.add dimension

proc expectFloat64Observation(node: JsonNode, shape: seq[int], label: string) =
  expectTensor(node, false)
  require(field(node, "shape") == jsonShape(shape), label & " shape mismatch")

proc expectFloat64Specials(specials: JsonNode, valueCount: int, label: string) =
  require(specials.kind == JArray, label & " nonfinite values must be an array")
  var seen = initHashSet[int]()
  for special in specials:
    let index = field(special, "index")
    require(index.kind == JInt and index.getInt >= 0 and index.getInt < valueCount,
      label & " nonfinite index out of bounds")
    require(index.getInt notin seen, label & " nonfinite index is duplicated")
    seen.incl(index.getInt)
    require(text(field(special, "value"), label & " nonfinite value") in ["nan", "+inf", "-inf"],
      label & " invalid nonfinite value")

proc expectFloat64ReductionInputs(inputs: JsonNode) =
  let reductions = field(inputs, "reductions")
  let reductionNames = ["packed_odd", "transpose", "offset", "reverse", "broadcast", "scalar", "empty"]
  require(reductions.kind == JObject and reductions.len == reductionNames.len,
    "float64 reduction inputs do not match required cases")
  for name in reductionNames:
    require(reductions.hasKey(name), "missing float64 reduction input " & name)
    let spec = reductions[name]
    let expectedShape = case name
      of "packed_odd": @[3, 5]
      of "transpose": @[2, 3]
      of "offset": @[3, 4]
      of "reverse": @[5]
      of "broadcast": @[1, 3]
      of "empty": @[2, 0, 3]
      else: @[]
    let expectedLayout = if name in ["transpose", "offset", "reverse", "broadcast"]: name
      else: "identity"
    require(field(spec, "shape") == jsonShape(expectedShape),
      "float64 reduction shape mismatch for " & name)
    require(text(field(spec, "layout"), "float64 reduction layout") == expectedLayout,
      "float64 reduction layout mismatch for " & name)
    expectFloat64Layout(spec, expectedLayout, "float64 reduction " & name)
    expectFloat64Axes(spec, true, "float64 reduction " & name)

  let scans = field(inputs, "scans")
  let scanNames = ["packed_odd", "scan_precision", "transpose", "offset",
    "reverse", "broadcast", "scalar", "empty"]
  require(scans.kind == JObject and scans.len == scanNames.len,
    "float64 scan inputs do not match required cases")
  for name in scanNames:
    require(scans.hasKey(name), "missing float64 scan input " & name)
    let spec = scans[name]
    let expectedShape = case name
      of "packed_odd": @[3, 5]
      of "scan_precision", "reverse": @[5]
      of "transpose": @[2, 3]
      of "offset": @[3, 4]
      of "broadcast": @[1, 3]
      of "empty": @[2, 0, 3]
      else: @[]
    let expectedLayout = if name in ["transpose", "offset", "reverse", "broadcast"]: name
      else: "identity"
    require(field(spec, "shape") == jsonShape(expectedShape),
      "float64 scan shape mismatch for " & name)
    require(text(field(spec, "layout"), "float64 scan layout") == expectedLayout,
      "float64 scan layout mismatch for " & name)
    expectFloat64Layout(spec, expectedLayout, "float64 scan " & name)
    expectFloat64Axes(spec, false, "float64 scan " & name)

  let reductionVjps = field(inputs, "reduction_vjps")
  let reductionVjpNames = ["offset", "one_zero", "multiple_zeros", "infinity", "nan",
    "zero_infinity", "zero_nan"]
  require(reductionVjps.kind == JObject and reductionVjps.len == reductionVjpNames.len,
    "float64 reduction VJP inputs do not match required cases")
  for name in reductionVjpNames:
    require(reductionVjps.hasKey(name), "missing float64 reduction VJP input " & name)
    let spec = reductionVjps[name]
    let layout = text(field(spec, "layout"), "float64 VJP layout")
    expectFloat64Layout(field(spec, "x"), layout, "float64 VJP " & name)
    let shape = float64LogicalShape(spec["x"], layout)
    expectFloat64Specials(field(spec, "nonfinite"), spec["x"]["values"].len,
      "float64 VJP " & name)
    let operations = field(spec, "operations")
    require(operations.kind == JArray and operations.len > 0,
      "float64 reduction VJP operations are missing")
    for operation in operations:
      let kind = text(field(operation, "operation"), "float64 reduction VJP operation")
      require(kind in ["sum", "prod", "mean"], "invalid float64 reduction VJP operation")
      let axis = field(operation, "axis")
      require(axis.kind in {JNull, JInt}, "invalid float64 reduction VJP axis")
      if axis.kind == JInt:
        require(axis.getInt >= -max(1, shape.len) and axis.getInt < max(1, shape.len),
          "float64 reduction VJP axis is out of bounds")
      require(field(operation, "keepdim").kind == JBool,
        "float64 reduction VJP keepdim must be boolean")
      let seed = field(operation, "seed")
      expectNonUnitFloat64Seed(seed, "float64 reduction VJP")
      let expectedSeedShape = float64ReductionShape(shape, axis, operation["keepdim"].getBool)
      require(seed["shape"] == jsonShape(expectedSeedShape),
        "float64 reduction VJP seed shape mismatch")

  let scanVjps = field(inputs, "scan_vjps")
  let scanVjpNames = ["packed_odd", "broadcast"]
  require(scanVjps.kind == JObject and scanVjps.len == scanVjpNames.len,
    "float64 scan VJP inputs do not match required cases")
  for name in scanVjpNames:
    require(scanVjps.hasKey(name), "missing float64 scan VJP input " & name)
    let spec = scanVjps[name]
    let layout = text(field(spec, "layout"), "float64 scan VJP layout")
    expectFloat64Layout(field(spec, "x"), layout, "float64 scan VJP " & name)
    let shape = float64LogicalShape(spec["x"], layout)
    expectFloat64Specials(field(spec, "nonfinite"), spec["x"]["values"].len,
      "float64 scan VJP " & name)
    let operations = field(spec, "operations")
    require(operations.kind == JArray and operations.len > 0,
      "float64 scan VJP operations are missing")
    for operation in operations:
      require(text(field(operation, "operation"), "float64 scan VJP operation") in ["cumsum", "cumprod"],
        "invalid float64 scan VJP operation")
      let axis = field(operation, "axis")
      require(axis.kind == JInt and axis.getInt >= -max(1, shape.len) and
        axis.getInt < max(1, shape.len), "float64 scan VJP axis is out of bounds")
      let seed = field(operation, "seed")
      expectNonUnitFloat64Seed(seed, "float64 scan VJP")
      require(seed["shape"] == jsonShape(shape),
        "float64 scan VJP seed shape mismatch")

proc expectFloat64Reductions(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == 4,
    "float64 reduction expected groups are incomplete")
  let reductions = field(inputs, "reductions")
  let expectedReductions = field(expected, "reductions")
  require(expectedReductions.kind == JObject and expectedReductions.len == reductions.len,
    "float64 reduction observations do not match inputs")
  for name, spec in reductions:
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let observations = field(expectedReductions, name)
    require(observations.kind == JObject and observations.len == 3,
      "float64 reduction operation observations are incomplete for " & name)
    for operation in ["sum", "prod", "mean"]:
      let byAxis = field(observations, operation)
      require(byAxis.kind == JObject and byAxis.len == spec["axes"].len,
        "float64 reduction axis observations are incomplete for " & name & "." & operation)
      for axis in spec["axes"]:
        let key = if axis.kind == JNull: "all" else: $axis.getInt
        let outputShape = float64ReductionShape(logicalShape, axis, true)
        expectFloat64Observation(field(byAxis, key), outputShape,
          "float64 reduction " & name & "." & operation & "." & key)

  let scans = field(inputs, "scans")
  let expectedScans = field(expected, "scans")
  require(expectedScans.kind == JObject and expectedScans.len == scans.len,
    "float64 scan observations do not match inputs")
  for name, spec in scans:
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let observations = field(expectedScans, name)
    require(observations.kind == JObject and observations.len == 2,
      "float64 scan operation observations are incomplete for " & name)
    for operation in ["cumsum", "cumprod"]:
      let byAxis = field(observations, operation)
      require(byAxis.kind == JObject and byAxis.len == spec["axes"].len,
        "float64 scan axis observations are incomplete for " & name & "." & operation)
      for axis in spec["axes"]:
        let key = $axis.getInt
        expectFloat64Observation(field(byAxis, key), logicalShape,
          "float64 scan " & name & "." & operation & "." & key)

  let reductionVjps = field(inputs, "reduction_vjps")
  let expectedReductionVjps = field(expected, "reduction_vjps")
  require(expectedReductionVjps.kind == JObject and expectedReductionVjps.len == reductionVjps.len,
    "float64 reduction VJP observations do not match inputs")
  for name, spec in reductionVjps:
    let logicalShape = float64LogicalShape(spec["x"], spec["layout"].getStr)
    let observations = field(expectedReductionVjps, name)
    require(observations.kind == JArray and observations.len == spec["operations"].len,
      "float64 reduction VJP observations are incomplete for " & name)
    for index in 0 ..< spec["operations"].len:
      let operation = spec["operations"][index]
      let axis = field(operation, "axis")
      let outputShape = float64ReductionShape(logicalShape, axis,
        operation["keepdim"].getBool)
      expectFloat64Observation(field(observations[index], "output"), outputShape,
        "float64 reduction VJP output " & name)
      expectFloat64Observation(field(observations[index], "grad"), logicalShape,
        "float64 reduction VJP gradient " & name)

  let scanVjps = field(inputs, "scan_vjps")
  let expectedScanVjps = field(expected, "scan_vjps")
  require(expectedScanVjps.kind == JObject and expectedScanVjps.len == scanVjps.len,
    "float64 scan VJP observations do not match inputs")
  for name, spec in scanVjps:
    let logicalShape = float64LogicalShape(spec["x"], spec["layout"].getStr)
    let observations = field(expectedScanVjps, name)
    require(observations.kind == JArray and observations.len == spec["operations"].len,
      "float64 scan VJP observations are incomplete for " & name)
    for observation in observations:
      expectFloat64Observation(field(observation, "output"), logicalShape,
        "float64 scan VJP output " & name)
      expectFloat64Observation(field(observation, "grad"), logicalShape,
        "float64 scan VJP gradient " & name)

proc meanDtypeOutputShape(shape: seq[int], axis: int): seq[int] =
  if shape.len == 0: return @[]
  result = shape
  var normalized = axis
  if normalized < 0: normalized += shape.len
  result[normalized] = 1

proc expectMeanDtypeInputs(inputs: JsonNode) =
  let names = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float32", "float64", "float64_precision", "uint64_upper", "packed_odd",
    "transpose", "offset", "reverse", "broadcast", "scalar", "empty"]
  let integerDtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
  require(inputs.kind == JObject and inputs.len == names.len,
    "mean dtype inputs do not match required cases")
  for name in names:
    require(inputs.hasKey(name), "missing mean dtype input " & name)
    let spec = inputs[name]
    require(spec.kind == JObject and spec.len == 5, "mean dtype input fields are incomplete for " & name)
    let expectedDtype = case name
      of "float32": "float32"
      of "float64", "float64_precision": "float64"
      of "uint64_upper", "empty": "uint64"
      of "packed_odd": "int16"
      of "transpose": "uint16"
      of "offset": "uint32"
      of "reverse", "scalar": "int64"
      of "broadcast": "uint8"
      else: name
    let dtype = text(field(spec, "dtype"), "mean dtype input dtype")
    require(dtype == expectedDtype, "mean dtype mismatch for " & name)
    let expectedShape = case name
      of "offset": @[3, 4]
      of "reverse": @[5]
      of "float64_precision", "uint64_upper": @[2]
      of "packed_odd": @[3, 5]
      of "broadcast": @[1, 3]
      of "scalar": @[]
      of "empty": @[2, 0, 3]
      else: @[2, 3]
    require(field(spec, "shape") == jsonShape(expectedShape),
      "mean dtype input shape mismatch for " & name)
    let expectedLayout = if name == "float64": "transpose"
      elif name in ["transpose", "offset", "reverse", "broadcast"]: name
      else: "identity"
    require(text(field(spec, "layout"), "mean dtype input layout") == expectedLayout,
      "mean dtype input layout mismatch for " & name)
    let expectedAxes = case name
      of "empty": @[0, 1, 2, -1]
      of "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
          "uint64", "float32", "float64", "packed_odd", "broadcast": @[0, 1, -1]
      else: @[0, -1]
    var axes = newJArray()
    for axis in expectedAxes: axes.add newJInt(axis)
    require(field(spec, "axes") == axes, "mean dtype axis coverage mismatch for " & name)
    if dtype in integerDtypes:
      expectIntegerWordsInput(spec, "mean dtype " & name, false)
    else:
      expectInputTensor(spec)

proc expectMeanDtypeError(node: JsonNode, label: string) =
  require(node.kind == JObject and node.len == 1 and
    text(field(node, "error"), label & " error") == "default_dtype_rejected",
    label & " must record an observed default-dtype rejection")

proc expectMeanDtypeExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "mean dtype observations do not match inputs")
  for name, spec in inputs:
    let dtype = spec["dtype"].getStr
    let integerInput = dtype in ["bool", "int8", "int16", "int32", "int64",
      "uint8", "uint16", "uint32", "uint64"]
    let observations = field(expected, name)
    require(observations.kind == JObject and observations.len == (if integerInput: 3 else: 2),
      "mean dtype observation groups are incomplete for " & name)
    let all = field(observations, "all")
    require(all.kind == JObject and all.len == 2, "mean dtype whole observations are incomplete for " & name)
    for outputDtype in ["float32", "float64"]:
      expectTensor(field(all, outputDtype), false, outputDtype)
      require(field(all[outputDtype], "shape") == newJArray(),
        "mean dtype whole output must be scalar for " & name)

    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let axes = field(observations, "axes")
    require(axes.kind == JObject and axes.len == spec["axes"].len,
      "mean dtype axis observations are incomplete for " & name)
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      let byDtype = field(axes, $axis)
      require(byDtype.kind == JObject and byDtype.len == 2,
        "mean dtype result dtypes are incomplete for " & name & "." & $axis)
      let outputShape = meanDtypeOutputShape(logicalShape, axis)
      for outputDtype in ["float32", "float64"]:
        expectTensor(field(byDtype, outputDtype), false, outputDtype)
        require(field(byDtype[outputDtype], "shape") == jsonShape(outputShape),
          "mean dtype axis output must retain reduced dimension for " & name & "." & $axis)

    if integerInput:
      let defaults = field(observations, "default")
      require(defaults.kind == JObject and defaults.len == 2,
        "mean dtype default rejection observations are incomplete for " & name)
      expectMeanDtypeError(field(defaults, "all"), "mean dtype " & name & " whole")
      let defaultAxes = field(defaults, "axes")
      require(defaultAxes.kind == JObject and defaultAxes.len == spec["axes"].len,
        "mean dtype default axis rejections are incomplete for " & name)
      for axisNode in spec["axes"]:
        let axis = $axisNode.getInt
        expectMeanDtypeError(field(defaultAxes, axis), "mean dtype " & name & "." & axis)
    else:
      require(not observations.hasKey("default"), "floating mean dtype input has default rejection data")

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
  require(cases.kind == JArray and cases.len == RequiredCaseIds.len, "fixture case count mismatch")
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
    of "tensor.integer_scans":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "integer scan fixtures require exact-value tolerances")
      require(inputs.kind == JObject and inputs.len > 0, "integer scan inputs missing")
      for _, spec in inputs: expectIntegerScanInput(spec)
    of "tensor.integer_reductions":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "integer reduction fixtures require exact-value tolerances")
      let requiredNames = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64", "scalar", "empty", "overflow", "transpose", "offset", "reverse", "broadcast"]
      require(inputs.kind == JObject and inputs.len == requiredNames.len,
        "integer reduction inputs do not match required cases")
      for name in requiredNames:
        require(inputs.hasKey(name), "missing integer reduction input " & name)
        let spec = inputs[name]
        expectIntegerReductionInput(spec)
        let expectedDtype = if name in ["bool", "int8", "int16", "int32", "int64", "uint8",
            "uint16", "uint32", "uint64"]: name
          elif name == "empty": "uint64"
          elif name == "offset": "uint16"
          elif name == "broadcast": "uint8"
          else: "int64"
        require(text(field(spec, "dtype"), "integer reduction dtype") == expectedDtype,
          "integer reduction dtype mismatch for " & name)
        let expectedShape = case name
          of "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
              "uint64", "transpose": %*[2, 3]
          of "empty": %*[2, 0, 3]
          of "overflow", "reverse": %*[5]
          of "offset": %*[3, 4]
          of "broadcast": %*[1, 3]
          else: %*[]
        require(field(spec, "shape") == expectedShape, "integer reduction shape mismatch for " & name)
        var requiredAxes = newJArray()
        requiredAxes.add newJNull()
        let axisCount = max(1, expectedShape.len)
        for axis in 0 ..< axisCount: requiredAxes.add newJInt(axis)
        for axis in 1 .. axisCount: requiredAxes.add newJInt(-axis)
        require(field(spec, "axes") == requiredAxes,
          "integer reduction axes do not match required coverage for " & name)
        let expectedLayout = if name in ["transpose", "offset", "reverse", "broadcast"]: name
          else: "identity"
        require(text(field(spec, "layout"), "integer reduction layout") == expectedLayout,
          "integer reduction layout mismatch for " & name)
    of "tensor.mean_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "mean dtype fixtures require 1e-6 tolerances")
      expectMeanDtypeInputs(inputs)
    of "tensor.float64_reductions":
      require(c["atol"].getFloat == 1e-12 and c["rtol"].getFloat == 1e-12,
        "float64 reduction fixtures require 1e-12 tolerances")
      require(inputs.kind == JObject and inputs.len == 4,
        "float64 reduction input groups are incomplete")
      expectFloat64ReductionInputs(inputs)
    of "autograd.product_gradient", "autograd.cumsum_gradient", "autograd.cumprod_gradient":
      let product = id == "autograd.product_gradient"
      require(inputs.kind == JObject and inputs.len > 0, "reduction gradient inputs missing")
      for _, spec in inputs:
        expectInputTensor(field(spec, "x"))
        require(text(field(spec, "layout"), "reduction layout") in
          ["identity", "transpose", "offset", "reverse", "broadcast"], "invalid reduction layout")
        require(field(spec, "nonfinite").kind == JArray, "nonfinite values must be an array")
        for special in spec["nonfinite"]:
          let index = field(special, "index")
          require(index.kind == JInt and index.getInt >= 0 and
            index.getInt < spec["x"]["values"].len, "nonfinite index out of bounds")
          let value = text(field(special, "value"), "nonfinite value")
          require(value in ["nan", "+inf"] or (not product and value == "-inf"),
            "invalid nonfinite value")
        let operations = field(spec, "operations")
        require(operations.kind == JArray and operations.len > 0, "reduction operations missing")
        for operation in operations:
          if product:
            require(field(operation, "axis").kind in {JNull, JInt}, "invalid product axis")
            require(field(operation, "keepdim").kind == JBool, "invalid product keepdim")
          else:
            require(field(operation, "axis").kind == JInt, "invalid cumulative scan axis")
          expectInputTensor(field(operation, "seed"))
          if not product:
            require(field(operation, "seed_nonfinite").kind == JArray,
              "nonfinite seed values must be an array")
            for special in operation["seed_nonfinite"]:
              let index = field(special, "index")
              require(index.kind == JInt and index.getInt >= 0 and
                index.getInt < operation["seed"]["values"].len, "nonfinite seed index out of bounds")
              require(text(field(special, "value"), "nonfinite seed value") in
                ["nan", "+inf", "-inf"], "invalid nonfinite seed value")
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
    of "tensor.integer_scans":
      require(expected.kind == JObject and expected.len == inputs.len,
        "integer scan expected cases must match inputs")
      for name, spec in inputs:
        let byOperation = field(expected, name)
        require(byOperation.kind == JObject and byOperation.len == 2,
          "integer scan must include cumsum and cumprod observations")
        for operation in ["cumsum", "cumprod"]:
          let byAxis = field(byOperation, operation)
          require(byAxis.kind == JObject and byAxis.len == spec["axes"].len,
            "integer scan axis observations are incomplete")
          for axis in spec["axes"]:
            let key = $axis.getInt
            expectTensor(field(byAxis, key), false, "int64")
    of "tensor.integer_reductions":
      require(expected.kind == JObject and expected.len == inputs.len,
        "integer reduction expected cases must match inputs")
      for name, spec in inputs:
        let byOperation = field(expected, name)
        require(byOperation.kind == JObject and byOperation.len == 2,
          "integer reduction must include sum and prod observations")
        for operation in ["sum", "prod"]:
          let byAxis = field(byOperation, operation)
          require(byAxis.kind == JObject and byAxis.len == spec["axes"].len,
            "integer reduction axis observations are incomplete")
          for axis in spec["axes"]:
            let key = if axis.kind == JNull: "all" else: $axis.getInt
            expectTensor(field(byAxis, key), false, "int64")
    of "tensor.float64_reductions":
      expectFloat64Reductions(expected, inputs)
    of "tensor.mean_dtype":
      expectMeanDtypeExpected(expected, inputs)
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
