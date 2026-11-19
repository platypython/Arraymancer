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
    "tensor.mean_dtype", "tensor.sum_dtype", "tensor.cumsum_dtype", "tensor.cumprod_dtype",
    "tensor.prod_dtype", "tensor.integer_output_dtype", "tensor.complex_scans",
    "tensor.complex_scan_dtype", "tensor.complex_mean_dtype", "tensor.complex_sum_dtype",
    "tensor.complex_prod_dtype", "tensor.extrema", "autograd.extrema", "tensor.indexed_extrema",
    "tensor.paired_extrema", "autograd.paired_extrema", "tensor.aminmax"]
  RequiredCaseAreas: array[RequiredCaseIds.len, string] = [
    "api:tensor_view", "api:tensors", "api:tensors", "api:torch", "api:torch",
    "api:torch", "api:autograd", "api:nn.functional", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:autograd", "api:autograd", "api:autograd", "api:autograd",
    "api:autograd", "api:nn.functional", "api:nn.functional", "api:nn.functional", "api:autograd",
    "api:autograd", "api:torch", "api:autograd", "api:autograd", "api:torch",
    "api:autograd", "api:autograd", "api:autograd", "api:torch", "api:torch",
    "api:autograd", "api:torch", "api:torch", "api:torch", "api:torch",
    "api:torch", "api:torch", "api:torch", "api:torch", "api:torch",
    "api:torch", "api:torch", "api:torch", "api:autograd", "api:torch", "api:torch",
    "api:autograd", "api:torch"]
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

proc realDtypeOutputShape(shape: seq[int], axis: int): seq[int] =
  if shape.len == 0: return @[]
  result = shape
  var normalized = axis
  if normalized < 0: normalized += shape.len
  result[normalized] = 1

const
  RealDtypeInputNames = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float32", "float64", "float64_precision", "uint64_upper", "packed_odd",
    "transpose", "offset", "reverse", "broadcast", "scalar", "empty"]
  RealDtypeScanExtraNames = ["float32_wide_scan", "float64_overflow", "float64_underflow",
    "nonfinite", "scalar_negative_zero", "vector_negative_zero"]
  RealDtypeProductExtraNames = ["float64_overflow", "float64_underflow", "float64_zero_nan",
    "uint64_upper_zero", "lane_range", "early_zero4", "zero_columns", "scalar_negative_zero",
    "cast_before_product"]

proc expectRealDtypeInput(spec: JsonNode, name: string) =
  require(spec.kind == JObject and spec.len == (if name == "nonfinite": 6 else: 5),
    "real dtype input fields are incomplete for " & name)
  let integerDtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
  let expectedDtype = case name
    of "float32", "float32_wide_scan": "float32"
    of "float64", "float64_precision", "float64_overflow", "float64_underflow",
        "float64_zero_nan", "lane_range", "early_zero4", "zero_columns",
        "cast_before_product", "nonfinite", "scalar_negative_zero", "vector_negative_zero": "float64"
    of "uint64_upper", "uint64_upper_zero", "empty": "uint64"
    of "packed_odd": "int16"
    of "transpose": "uint16"
    of "offset": "uint32"
    of "reverse", "scalar": "int64"
    of "broadcast": "uint8"
    else: name
  let dtype = text(field(spec, "dtype"), "real dtype input dtype")
  require(dtype == expectedDtype, "real dtype mismatch for " & name)
  let expectedShape = case name
    of "offset": @[3, 4]
    of "reverse": @[5]
    of "float64_precision", "uint64_upper": @[2]
    of "float64_overflow", "float64_underflow", "float64_zero_nan", "uint64_upper_zero": @[3]
    of "lane_range": @[16]
    of "cast_before_product": @[64]
    of "early_zero4": @[4]
    of "zero_columns": @[4, 5]
    of "packed_odd": @[3, 5]
    of "broadcast": @[1, 3]
    of "scalar", "scalar_negative_zero": @[]
    of "empty": @[2, 0, 3]
    of "float32_wide_scan": @[3]
    of "nonfinite": @[4]
    of "vector_negative_zero": @[1]
    else: @[2, 3]
  require(field(spec, "shape") == jsonShape(expectedShape),
    "real dtype input shape mismatch for " & name)
  let expectedLayout = if name == "float64": "transpose"
    elif name in ["transpose", "offset", "reverse", "broadcast"]: name
    else: "identity"
  require(text(field(spec, "layout"), "real dtype input layout") == expectedLayout,
    "real dtype input layout mismatch for " & name)
  let expectedAxes = case name
    of "empty": @[0, 1, 2, -1]
    of "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64", "float32", "float64", "packed_odd", "broadcast", "zero_columns": @[0, 1, -1]
    else: @[0, -1]
  var axes = newJArray()
  for axis in expectedAxes: axes.add newJInt(axis)
  require(field(spec, "axes") == axes, "real dtype axis coverage mismatch for " & name)
  if dtype in integerDtypes:
    expectIntegerWordsInput(spec, "real dtype " & name, false)
  else:
    expectInputTensor(spec)
  if name == "nonfinite":
    let specials = field(spec, "nonfinite")
    require(specials.kind == JArray and specials.len == 2 and
      specials[0] == %*{"index": 1, "value": "+inf"} and
      specials[1] == %*{"index": 2, "value": "-inf"},
      "nonfinite real dtype scan specials do not match the required source values")
  else:
    require(not spec.hasKey("nonfinite"), "unexpected nonfinite real dtype input " & name)

proc expectRealDtypeInputs(inputs: JsonNode, includeScanExtras = false,
                           includeProductExtras = false) =
  let expectedCount = RealDtypeInputNames.len +
    (if includeScanExtras: RealDtypeScanExtraNames.len else: 0) +
    (if includeProductExtras: RealDtypeProductExtraNames.len else: 0)
  require(inputs.kind == JObject and inputs.len == expectedCount,
    "real dtype inputs do not match required cases")
  for name in RealDtypeInputNames:
    require(inputs.hasKey(name), "missing real dtype input " & name)
    expectRealDtypeInput(inputs[name], name)
  if includeScanExtras:
    for name in RealDtypeScanExtraNames:
      require(inputs.hasKey(name), "missing real dtype scan input " & name)
      expectRealDtypeInput(inputs[name], name)
  if includeProductExtras:
    for name in RealDtypeProductExtraNames:
      require(inputs.hasKey(name), "missing real dtype product input " & name)
      expectRealDtypeInput(inputs[name], name)

proc expectMeanDtypeError(node: JsonNode, label: string) =
  require(node.kind == JObject and node.len == 1 and
    text(field(node, "error"), label & " error") == "default_dtype_rejected",
    label & " must record an observed default-dtype rejection")

proc expectRealDtypeExpected(expected, inputs: JsonNode, includeMeanRejections: bool) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "real dtype observations do not match inputs")
  for name, spec in inputs:
    let dtype = spec["dtype"].getStr
    let integerInput = dtype in ["bool", "int8", "int16", "int32", "int64",
      "uint8", "uint16", "uint32", "uint64"]
    let hasRejections = includeMeanRejections and integerInput
    let observations = field(expected, name)
    require(observations.kind == JObject and observations.len == (if hasRejections: 3 else: 2),
      "real dtype observation groups are incomplete for " & name)
    let all = field(observations, "all")
    require(all.kind == JObject and all.len == 2, "real dtype whole observations are incomplete for " & name)
    for outputDtype in ["float32", "float64"]:
      expectTensor(field(all, outputDtype), false, outputDtype)
      require(field(all[outputDtype], "shape") == newJArray(),
        "real dtype whole output must be scalar for " & name)

    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let axes = field(observations, "axes")
    require(axes.kind == JObject and axes.len == spec["axes"].len,
      "real dtype axis observations are incomplete for " & name)
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      let byDtype = field(axes, $axis)
      require(byDtype.kind == JObject and byDtype.len == 2,
        "real dtype result dtypes are incomplete for " & name & "." & $axis)
      let outputShape = realDtypeOutputShape(logicalShape, axis)
      for outputDtype in ["float32", "float64"]:
        expectTensor(field(byDtype, outputDtype), false, outputDtype)
        require(field(byDtype[outputDtype], "shape") == jsonShape(outputShape),
          "real dtype axis output must retain reduced dimension for " & name & "." & $axis)

    if hasRejections:
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
      require(not observations.hasKey("default"), "real dtype observations must not contain default rejection data")

proc expectRealDtypeScanExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "real dtype scan observations do not match inputs")
  for name, spec in inputs:
    let observations = field(expected, name)
    require(observations.kind == JObject and observations.len == 1,
      "real dtype scan observation groups are incomplete for " & name)
    let axes = field(observations, "axes")
    require(axes.kind == JObject and axes.len == spec["axes"].len,
      "real dtype scan axis observations are incomplete for " & name)
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      let byDtype = field(axes, $axis)
      require(byDtype.kind == JObject and byDtype.len == 2,
        "real dtype scan outputs are incomplete for " & name & "." & $axis)
      for outputDtype in ["float32", "float64"]:
        let directions = field(byDtype, outputDtype)
        require(directions.kind == JObject and directions.len == 2,
          "real dtype scan directions are incomplete for " & name & "." & $axis & "." & outputDtype)
        for direction in ["forward", "reverse"]:
          let observation = field(directions, direction)
          expectTensor(observation, false, outputDtype)
          require(field(observation, "shape") == jsonShape(logicalShape),
            "real dtype scan output shape must match input for " & name & "." & $axis & "." & direction)


const ComplexScanInputNames = ["complex64", "complex128", "transpose", "offset",
  "reverse", "broadcast", "scalar", "empty", "singleton", "precision_cancellation",
  "overflow_recovery", "underflow_recovery", "signed_zero", "nonfinite"]

proc complexScanNumber(node: JsonNode): bool =
  node.kind in {JInt, JFloat} and
    classify(node.getFloat) notin {fcNan, fcInf, fcNegInf}

proc complexScanFiniteFloat(node: JsonNode): bool =
  node.kind == JFloat and classify(node.getFloat) notin {fcNan, fcInf, fcNegInf}

proc complexSafeIntegerCast(node: JsonNode, dtype: string): bool =
  if node.kind == JInt:
    let value = node.getBiggestInt
    case dtype
    of "uint8": value >= 0 and value <= high(uint8).int64
    of "int8": value >= low(int8).int64 and value <= high(int8).int64
    of "int16": value >= low(int16).int64 and value <= high(int16).int64
    of "int32": value >= low(int32).int64 and value <= high(int32).int64
    of "int64": true
    else: false
  elif node.kind == JFloat:
    let value = trunc(node.getFloat)
    case dtype
    of "uint8": value >= 0.0 and value <= high(uint8).float64
    of "int8": value >= low(int8).float64 and value <= high(int8).float64
    of "int16": value >= low(int16).float64 and value <= high(int16).float64
    of "int32": value >= low(int32).float64 and value <= high(int32).float64
    of "int64": value >= low(int64).float64 and value < 9223372036854775808.0
    else: false
  else:
    false


proc expectComplexScanInput(spec: JsonNode, name: string) =
  let hasNonfinite = name == "nonfinite"
  require(spec.kind == JObject and spec.len == (if hasNonfinite: 7 else: 6),
    "complex scan input fields are incomplete for " & name)
  let dtype = case name
    of "complex128", "offset", "broadcast", "scalar", "signed_zero", "nonfinite": "complex128"
    else: "complex64"
  require(text(field(spec, "dtype"), "complex scan dtype") == dtype,
    "complex scan dtype mismatch for " & name)
  let expectedShape = case name
    of "complex64", "complex128", "transpose": @[2, 3]
    of "offset": @[3, 4]
    of "reverse": @[5]
    of "broadcast": @[1, 3]
    of "empty": @[2, 0, 3]
    of "singleton": @[1]
    of "precision_cancellation", "overflow_recovery", "underflow_recovery": @[3]
    of "signed_zero": @[2]
    of "nonfinite": @[5]
    else: @[]
  let shape = field(spec, "shape")
  require(shape == jsonShape(expectedShape), "complex scan shape mismatch for " & name)
  var size = 1
  for dimension in expectedShape:
    if dimension == 0: size = 0
    elif size != 0:
      require(size <= high(int) div dimension, "complex scan input shape overflows")
      size *= dimension
  let values = field(spec, "values")
  require(values.kind == JArray and values.len == size,
    "complex scan component count does not match shape for " & name)
  for pair in values:
    require(pair.kind == JArray and pair.len == 2,
      "complex scan values must be real/imaginary pairs for " & name)
    for component in pair:
      require(complexScanNumber(component),
        "complex scan inputs must contain finite components for " & name)

  let expectedLayout = if name in ["transpose", "offset", "reverse", "broadcast"]: name
                       else: "identity"
  require(text(field(spec, "layout"), "complex scan layout") == expectedLayout,
    "complex scan layout mismatch for " & name)
  let expectedAxes = case name
    of "complex64", "complex128", "transpose", "offset", "broadcast": @[0, 1, -1]
    of "empty": @[0, 1, 2, -1]
    else: @[0, -1]
  var axes = newJArray()
  for axis in expectedAxes: axes.add newJInt(axis)
  require(field(spec, "axes") == axes, "complex scan axes mismatch for " & name)
  let dimensions = max(1, expectedShape.len)
  let invalidAxes = field(spec, "invalid_axes")
  require(invalidAxes == %*[dimensions, -dimensions - 1],
    "complex scan invalid axes mismatch for " & name)

  if hasNonfinite:
    let specials = field(spec, "nonfinite")
    require(specials.kind == JArray and specials.len == 3 and
      specials[0] == %*{"index": 1, "component": "real", "value": "+inf"} and
      specials[1] == %*{"index": 2, "component": "real", "value": "-inf"} and
      specials[2] == %*{"index": 4, "component": "imag", "value": "nan"},
      "complex scan nonfinite source components do not match the required values")
  else:
    require(not spec.hasKey("nonfinite"), "unexpected complex scan nonfinite input " & name)

proc complexNegativeZero(node: JsonNode): bool =
  node.kind == JFloat and node.getFloat == 0.0 and
    (cast[uint64](node.getFloat) shr 63) == 1'u64

proc complexNegativeZeroIndices(values: JsonNode, component: string): JsonNode =
  result = newJArray()
  for index in 0 ..< values.len:
    if complexNegativeZero(field(values[index], component)):
      result.add newJInt(index)

proc expectComplexObservation(node: JsonNode, shape: seq[int], dtype, label: string) =
  require(node.kind == JObject and node.len == 6,
    label & " complex observation fields are incomplete")
  require(field(node, "shape") == jsonShape(shape), label & " shape mismatch")
  let count = block:
    var n = 1
    for dimension in shape:
      if dimension == 0: n = 0
      elif n != 0:
        require(n <= high(int) div dimension, label & " shape overflows")
        n *= dimension
    n
  require(field(node, "numel").kind == JInt and node["numel"].getInt == count,
    label & " numel/shape mismatch")
  require(text(field(node, "dtype"), label & " dtype") == dtype and
    text(field(node, "device"), label & " device") == "cpu",
    label & " complex observation dtype/device mismatch")
  let values = field(node, "values")
  require(values.kind == JArray and values.len == count,
    label & " complex values count does not match shape")
  for value in values:
    require(value.kind == JObject and value.len == 2,
      label & " complex values must be real/imaginary objects")
    for component in ["real", "imag"]:
      let scalar = field(value, component)
      require(complexScanFiniteFloat(scalar) or
        (scalar.kind == JString and scalar.getStr in ["nan", "+inf", "-inf"]),
        label & " has an invalid complex component")
  let negativeZero = field(node, "negative_zero")
  require(negativeZero.kind == JObject and negativeZero.len == 2,
    label & " signed-zero components are incomplete")
  for component in ["real", "imag"]:
    let indices = field(negativeZero, component)
    require(indices.kind == JArray and
      indices == complexNegativeZeroIndices(values, component),
      label & " signed-zero indices do not match component values")

proc expectComplexInvalidAxis(node: JsonNode, label: string) =
  require(node.kind == JObject and node.len == 1 and
    text(field(node, "error"), label & " error") == "invalid_axis",
    label & " must record the native invalid-axis rejection")

proc expectComplexScanExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "complex scan observations do not match inputs")
  for name, spec in inputs:
    let observations = field(expected, name)
    require(observations.kind == JObject and observations.len == 2,
      "complex scan operation observations are incomplete for " & name)
    let shape = float64LogicalShape(spec, spec["layout"].getStr)
    let dtype = spec["dtype"].getStr
    for operation in ["cumsum", "cumprod"]:
      let byOperation = field(observations, operation)
      require(byOperation.kind == JObject and byOperation.len == 2,
        "complex scan direction/rejection groups are incomplete for " & name & "." & operation)
      let axes = field(byOperation, "axes")
      require(axes.kind == JObject and axes.len == spec["axes"].len,
        "complex scan axes are incomplete for " & name & "." & operation)
      for axisNode in spec["axes"]:
        let axis = $axisNode.getInt
        let directions = field(axes, axis)
        require(directions.kind == JObject and directions.len == 2,
          "complex scan directions are incomplete for " & name & "." & operation & "." & axis)
        for direction in ["forward", "reverse"]:
          expectComplexObservation(field(directions, direction), shape, dtype,
            "complex scan " & name & "." & operation & "." & axis & "." & direction)
      let invalidAxes = field(byOperation, "invalid_axes")
      require(invalidAxes.kind == JObject and invalidAxes.len == spec["invalid_axes"].len,
        "complex scan invalid-axis observations are incomplete for " & name & "." & operation)
      for axisNode in spec["invalid_axes"]:
        let axis = $axisNode.getInt
        expectComplexInvalidAxis(field(invalidAxes, axis),
          "complex scan " & name & "." & operation & "." & axis)

const
  IntegerOutputDtypeNames = ["bool", "uint8", "int8", "int16", "int32", "int64",
    "uint16", "uint32", "uint64"]
  IntegerOutputSourceNames = ["bool", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "float32", "float64", "transpose", "offset",
    "reverse", "broadcast", "wrap", "scalar", "empty", "singleton", "bool_nan",
    "float_signed"]

proc integerOutputAxes(values: openArray[int]): JsonNode =
  result = newJArray()
  for value in values: result.add newJInt(value)

proc integerOutputWords(values: openArray[int64]): JsonNode =
  result = newJArray()
  for value in values: result.add newJInt(value)

proc integerOutputNumel(shape: openArray[int]): int =
  result = 1
  for dimension in shape:
    if dimension == 0: return 0
    result *= dimension

proc expectIntegerOutputInput(spec: JsonNode, name: string) =
  let expectedDtype = case name
    of "transpose": "uint16"
    of "offset": "uint32"
    of "reverse": "int64"
    of "broadcast": "uint8"
    of "wrap": "int64"
    of "scalar", "empty": "uint64"
    of "singleton": "int32"
    of "bool_nan", "float_signed": "float64"
    else: name
  let expectedShape = case name
    of "offset": @[3, 4]
    of "reverse": @[5]
    of "broadcast": @[1, 3]
    of "scalar": @[]
    of "empty": @[2, 0, 3]
    of "singleton": @[1]
    of "bool_nan": @[3]
    of "float_signed": @[4]
    else: @[2, 3]
  let expectedLayout = case name
    of "float64": "transpose"
    of "transpose", "offset", "reverse", "broadcast": name
    else: "identity"
  let expectedAxes = case name
    of "transpose", "offset", "reverse", "scalar", "singleton", "bool_nan",
        "float_signed": @[0, -1]
    of "broadcast", "wrap": @[0, 1, -1]
    of "empty": @[0, 1, 2, -1, -2, -3]
    else: @[0, 1, -1]
  let extras = (if name in ["scalar", "empty"]: 1 else: 0) +
    (if name == "bool_nan": 2 else: 0) +
    (if name == "float_signed": 1 else: 0)
  require(spec.kind == JObject and spec.len == 5 + extras,
    "integer output input fields are incomplete for " & name)
  require(text(field(spec, "dtype"), "integer output input dtype") == expectedDtype,
    "integer output input dtype mismatch for " & name)
  require(field(spec, "shape") == jsonShape(expectedShape),
    "integer output input shape mismatch for " & name)
  require(text(field(spec, "layout"), "integer output input layout") == expectedLayout,
    "integer output input layout mismatch for " & name)
  require(field(spec, "axes") == integerOutputAxes(expectedAxes),
    "integer output axes mismatch for " & name)
  if expectedDtype in ["bool", "int8", "int16", "int32", "int64",
      "uint8", "uint16", "uint32", "uint64"]:
    expectIntegerWordsInput(spec, "integer output " & name, false)
  else:
    expectInputTensor(spec)
  if name in ["scalar", "empty"]:
    let expectedInvalid = if name == "scalar": @[1] else: @[3]
    require(field(spec, "invalid_axes") == integerOutputAxes(expectedInvalid),
      "integer output invalid-axis fixture mismatch for " & name)
  else:
    require(not spec.hasKey("invalid_axes"),
      "unexpected integer output invalid axes for " & name)
  if name == "bool_nan":
    require(field(spec, "output_dtypes") == %*["bool"] and
      field(spec, "nonfinite") == %*[{"index": 1, "value": "nan"}],
      "bool integer output nonfinite fixture mismatch")
  elif name == "float_signed":
    require(field(spec, "output_dtypes") == %*["int8", "int16", "int32", "int64"] and
      not spec.hasKey("nonfinite"), "signed float output target subset mismatch")
  else:
    require(not spec.hasKey("output_dtypes") and not spec.hasKey("nonfinite"),
      "unexpected restricted or nonfinite integer output input " & name)

proc expectIntegerOutputInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == IntegerOutputSourceNames.len,
    "integer output inputs do not match required source and shape cases")
  for name in IntegerOutputSourceNames:
    require(inputs.hasKey(name), "missing integer output input " & name)
    expectIntegerOutputInput(inputs[name], name)
  require(inputs["uint64"]["words"] ==
    integerOutputWords([0'i64, high(int64), low(int64), -1, -2, 1]) and
    inputs["int64"]["words"] ==
    integerOutputWords([low(int64), high(int64), -1, 1, 2, 3]) and
    inputs["wrap"]["words"] == %*[127, 3, 255, 2, -128, -1],
    "integer output fixtures must retain signed/unsigned high-bit and wrapping words")
  require(inputs["float32"]["values"] == %*[1.9, 2.9, 3.4, 4.1, 0.25, 5.8] and
    inputs["float64"]["values"] == %*[0.1, 1.5, 2.0, 3.25, 4.5, 0.5] and
    inputs["float_signed"]["values"] == %*[-128.9, 127.9, -1.9, 2.9],
    "integer output float fixtures must keep unsigned casts in-range and exercise signed truncation")

proc expectIntegerOutputTensor(node: JsonNode, expectedShape: seq[int], dtype: string) =
  let expectedFields = if dtype == "uint64": 6 else: 5
  require(node.kind == JObject and node.len == expectedFields,
    "integer output tensor observation fields are incomplete")
  require(field(node, "shape") == jsonShape(expectedShape),
    "integer output tensor shape mismatch")
  require(field(node, "numel").kind == JInt and
    node["numel"].getInt == integerOutputNumel(expectedShape),
    "integer output tensor numel mismatch")
  require(text(field(node, "dtype"), "integer output tensor dtype") == dtype and
    text(field(node, "device"), "integer output tensor device") == "cpu",
    "integer output tensor dtype/device mismatch")
  if dtype == "uint64":
    require(text(field(node, "word_encoding"), "uint64 word encoding") ==
      "signed_twos_complement", "uint64 output words must preserve exact bit patterns")
  else:
    require(not node.hasKey("word_encoding"), "word encoding is only valid for uint64 outputs")
  let values = field(node, "values")
  require(values.kind == JArray and values.len == integerOutputNumel(expectedShape),
    "integer output tensor values do not match shape")
  for value in values:
    if dtype == "bool":
      require(value.kind == JBool, "bool output tensors require boolean values")
    else:
      require(value.kind == JInt, "integer output tensors require exact integer words")
      let word = value.getBiggestInt
      case dtype
      of "int8": require(word >= low(int8).int64 and word <= high(int8).int64, "int8 output is out of range")
      of "int16": require(word >= low(int16).int64 and word <= high(int16).int64, "int16 output is out of range")
      of "int32": require(word >= low(int32).int64 and word <= high(int32).int64, "int32 output is out of range")
      of "uint8": require(word >= 0 and word <= high(uint8).int64, "uint8 output is out of range")
      of "uint16": require(word >= 0 and word <= high(uint16).int64, "uint16 output is out of range")
      of "uint32": require(word >= 0 and word <= high(uint32).int64, "uint32 output is out of range")
      of "int64", "uint64": discard
      else: invalid("unknown integer output dtype " & dtype)

proc expectIntegerOutputResult(node: JsonNode, shape: seq[int], dtype: string,
                               rejected: bool, errorCode, label: string) =
  if rejected:
    require(node.kind == JObject and node.len == 1 and
      text(field(node, "error"), label & " error") == errorCode,
      label & " must record the expected normalized error")
  else:
    expectIntegerOutputTensor(node, shape, dtype)

proc integerOutputTargets(name: string): seq[string] =
  if name == "bool_nan":
    result = @["bool"]
  elif name == "float_signed":
    result = @["int8", "int16", "int32", "int64"]
  else:
    for dtype in IntegerOutputDtypeNames: result.add dtype

proc expectIntegerOutputExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "integer output observations do not match inputs")
  for name, spec in inputs:
    let observations = field(expected, name)
    let hasInvalidAxes = spec.hasKey("invalid_axes")
    require(observations.kind == JObject and observations.len == (if hasInvalidAxes: 4 else: 3),
      "integer output observation groups are incomplete for " & name)
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let nonempty = integerOutputNumel(logicalShape) > 0
    let scanRejectsNonempty = nonempty and logicalShape.len > 0
    let targets = integerOutputTargets(name)

    let reductions = field(observations, "reductions")
    require(reductions.kind == JObject and reductions.len == 2,
      "integer output reductions are incomplete for " & name)
    for operation in ["sum", "prod"]:
      let byOperation = field(reductions, operation)
      require(byOperation.kind == JObject and byOperation.len == 2,
        "integer output reduction groups are incomplete for " & name & "." & operation)
      let whole = field(byOperation, "all")
      require(whole.kind == JObject and whole.len == targets.len,
        "integer output whole reductions are incomplete for " & name & "." & operation)
      for dtype in targets:
        let rejected = dtype in ["uint16", "uint32", "uint64"] and nonempty
        expectIntegerOutputResult(field(whole, dtype), @[], dtype, rejected,
          "unsupported_output_dtype", name & "." & operation & ".all." & dtype)
      let axes = field(byOperation, "axes")
      require(axes.kind == JObject and axes.len == spec["axes"].len,
        "integer output axis reductions are incomplete for " & name & "." & operation)
      for axisNode in spec["axes"]:
        let axis = axisNode.getInt
        let byDtype = field(axes, $axis)
        require(byDtype.kind == JObject and byDtype.len == targets.len,
          "integer output reduction dtypes are incomplete for " & name & "." & operation & "." & $axis)
        let shape = realDtypeOutputShape(logicalShape, axis)
        for dtype in targets:
          let rejected = dtype in ["uint16", "uint32", "uint64"] and nonempty
          expectIntegerOutputResult(field(byDtype, dtype), shape, dtype, rejected,
            "unsupported_output_dtype", name & "." & operation & "." & $axis & "." & dtype)

    let scans = field(observations, "scans")
    require(scans.kind == JObject and scans.len == 2,
      "integer output scans are incomplete for " & name)
    for operation in ["cumsum", "cumprod"]:
      let byOperation = field(scans, operation)
      require(byOperation.kind == JObject and byOperation.len == 1,
        "integer output scan groups are incomplete for " & name & "." & operation)
      let axes = field(byOperation, "axes")
      require(axes.kind == JObject and axes.len == spec["axes"].len,
        "integer output scan axes are incomplete for " & name & "." & operation)
      for axisNode in spec["axes"]:
        let axis = axisNode.getInt
        let byDtype = field(axes, $axis)
        require(byDtype.kind == JObject and byDtype.len == targets.len,
          "integer output scan dtypes are incomplete for " & name & "." & operation & "." & $axis)
        for dtype in targets:
          let directions = field(byDtype, dtype)
          require(directions.kind == JObject and directions.len == 2,
            "integer output scan directions are incomplete for " & name & "." & operation & "." & $axis)
          let rejected = dtype in ["bool", "uint16", "uint32", "uint64"] and scanRejectsNonempty
          for direction in ["forward", "reverse"]:
            expectIntegerOutputResult(field(directions, direction), logicalShape, dtype,
              rejected, "unsupported_output_dtype",
              name & "." & operation & "." & $axis & "." & dtype & "." & direction)

    let mean = field(observations, "mean")
    require(mean.kind == JObject and mean.len == 2,
      "integer output mean rejection groups are incomplete for " & name)
    let meanAll = field(mean, "all")
    require(meanAll.kind == JObject and meanAll.len == targets.len,
      "integer output whole mean rejections are incomplete for " & name)
    for dtype in targets:
      expectIntegerOutputResult(field(meanAll, dtype), @[], dtype, true,
        "integer_mean_dtype_rejected", name & ".mean.all." & dtype)
    let meanAxes = field(mean, "axes")
    require(meanAxes.kind == JObject and meanAxes.len == spec["axes"].len,
      "integer output axis mean rejections are incomplete for " & name)
    for axisNode in spec["axes"]:
      let axis = $axisNode.getInt
      let byDtype = field(meanAxes, axis)
      require(byDtype.kind == JObject and byDtype.len == targets.len,
        "integer output axis mean dtypes are incomplete for " & name & "." & axis)
      for dtype in targets:
        expectIntegerOutputResult(field(byDtype, dtype), @[], dtype, true,
          "integer_mean_dtype_rejected", name & ".mean." & axis & "." & dtype)

    if hasInvalidAxes:
      let invalidAxes = field(observations, "invalid_scan_axes")
      require(invalidAxes.kind == JObject and invalidAxes.len == 2,
        "integer output invalid scan-axis observations are incomplete for " & name)
      for operation in ["cumsum", "cumprod"]:
        let byOperation = field(invalidAxes, operation)
        require(byOperation.kind == JObject and byOperation.len == spec["invalid_axes"].len,
          "integer output invalid scan axes are incomplete for " & name & "." & operation)
        for axisNode in spec["invalid_axes"]:
          let axis = $axisNode.getInt
          let byDtype = field(byOperation, axis)
          require(byDtype.kind == JObject and byDtype.len == targets.len,
            "integer output invalid-axis dtypes are incomplete for " & name & "." & operation)
          for dtype in targets:
            expectIntegerOutputResult(field(byDtype, dtype), @[], dtype, true,
              "invalid_axis", name & "." & operation & "." & axis & "." & dtype)


const ComplexScanDtypeInputNames = [
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
  "float32", "float64", "complex64", "complex128", "transpose", "offset", "reverse",
  "broadcast", "scalar", "scalar_imaginary", "scalar_negative_zero", "empty",
  "precision", "overflow_recovery", "underflow_recovery", "nonfinite",
  "complex_negative", "uint64_precision"]
const ComplexReductionDtypeInputNames = [
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
  "float32", "float64", "complex64", "complex128", "transpose", "offset", "reverse",
  "broadcast", "scalar", "scalar_imaginary", "scalar_negative_zero", "empty",
  "precision", "overflow_recovery", "underflow_recovery", "nonfinite",
  "complex_negative", "uint64_precision", "rounding", "complex64_cancellation",
  "double_precision", "packed_columns_cancellation"]
const ComplexScanDtypeRealSources = [
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
  "float32", "float64"]

proc complexScanDtypeTargets(name: string): seq[string] =
  if name in ComplexScanDtypeRealSources or name == "uint64_precision":
    return @["complex64", "complex128"]
  if name == "precision":
    return @["complex64", "complex128", "float32", "float64", "int32", "int64"]
  if name in ["overflow_recovery", "underflow_recovery", "nonfinite"]:
    return @["complex64", "complex128", "float32", "float64"]
  if name == "complex_negative":
    return @["complex64", "complex128", "float32", "float64", "int8", "int16", "int32", "int64"]
  if name == "complex64_cancellation" or name == "packed_columns_cancellation":
    return @["complex64", "complex128", "float32", "float64", "int32", "int64"]
  if name == "double_precision":
    return @["complex64", "complex128", "float32", "float64", "int64"]
  @["bool", "uint8", "int8", "int16", "int32", "int64", "uint16", "uint32", "uint64",
    "float32", "float64", "complex64", "complex128"]

proc complexSumDtypeTargets(name: string): seq[string] =
  result = complexScanDtypeTargets(name)
  if name notin ComplexScanDtypeRealSources and name != "uint64_precision":
    result.add "default"

proc expectComplexScanDtypeInput(spec: JsonNode, name: string;
                                 mean: static[bool] = false,
                                 sum: static[bool] = false,
                                 prod: static[bool] = false) =
  let nonfinite = name == "nonfinite"
  require(spec.kind == JObject and spec.len == (if nonfinite: 8 else: 7),
    "complex scan dtype input fields are incomplete for " & name)
  let sourceDtype =
    if name in ComplexScanDtypeRealSources: name
    elif name == "uint64_precision": "uint64"
    elif name in ["complex64", "transpose", "reverse", "scalar_negative_zero",
        "rounding", "complex64_cancellation", "packed_columns_cancellation"]: "complex64"
    else: "complex128"
  require(text(field(spec, "dtype"), "complex scan source dtype") == sourceDtype,
    "complex scan source dtype mismatch for " & name)
  let shape = case name
    of "scalar", "scalar_imaginary", "scalar_negative_zero": @[]
    of "empty": @[2, 0, 3]
    of "offset": @[3, 4]
    of "reverse": @[5]
    of "broadcast": @[1, 3]
    of "precision": @[2]
    of "rounding", "complex64_cancellation", "double_precision": @[3]
    of "packed_columns_cancellation": @[5, 17]
    of "overflow_recovery", "underflow_recovery", "nonfinite", "complex_negative",
        "uint64_precision": @[3]
    else: @[2, 3]
  require(field(spec, "shape") == jsonShape(shape), "complex scan source shape mismatch for " & name)
  let count = integerOutputNumel(shape)
  let integerSource = sourceDtype notin ["float32", "float64", "complex64", "complex128"]
  let values = field(spec, if integerSource: "words" else: "values")
  require(values.kind == JArray and values.len == count,
    "complex scan source values do not match shape for " & name)
  for value in values:
    if integerSource:
      require(value.kind == JInt, "complex scan source words must be exact integers")
    elif sourceDtype in ["complex64", "complex128"]:
      require(value.kind == JArray and value.len == 2,
        "complex scan source values must be component pairs")
      for component in value:
        require(complexScanNumber(component), "complex scan source component must be finite")
    else:
      require(complexScanNumber(value), "real scan source value must be finite")
  let layout = if name in ["transpose", "offset", "reverse", "broadcast"]: name else: "identity"
  require(text(field(spec, "layout"), "complex scan layout") == layout,
    "complex scan source layout mismatch for " & name)
  let axes =
    if shape.len == 2: @[0, 1, -1]
    elif shape.len == 3: @[0, 1, 2, -1]
    else: @[0, -1]
  require(field(spec, "axes") == jsonShape(axes), "complex scan axes mismatch for " & name)
  let dimensions = max(1, shape.len)
  require(field(spec, "invalid_axes") == %*[dimensions, -dimensions - 1],
    "complex scan invalid axes mismatch for " & name)
  let outputs = field(spec, "output_dtypes")
  let targets =
    when mean:
      if sourceDtype in ["complex64", "complex128"]:
        @["bool", "uint8", "int8", "int16", "int32", "int64", "uint16", "uint32", "uint64",
          "float32", "float64", "complex64", "complex128", "default"]
      else: @["complex64", "complex128"]
    elif sum:
      complexSumDtypeTargets(name)
    elif prod:
      complexSumDtypeTargets(name)
    else:
      complexScanDtypeTargets(name)
  require(outputs.kind == JArray and outputs.len == targets.len,
    "complex scan output dtype coverage mismatch for " & name)
  var seen = initHashSet[string]()
  for output in outputs:
    let dtype = text(output, "complex scan output dtype")
    require(dtype in targets and dtype notin seen, "unexpected or duplicate complex scan output dtype")
    seen.incl(dtype)
  if (sum or prod) and sourceDtype in ["complex64", "complex128"]:
    for output in outputs:
      let dtype = output.getStr
      if dtype in ["uint8", "int8", "int16", "int32", "int64"]:
        for value in values:
          require(complexSafeIntegerCast(value[0], dtype),
            "complex source exceeds the safe " & dtype & " cast domain for " & name)
  case name
  of "rounding":
    require(values == %*[[1.9, 2.0], [2.9, -3.0], [0.0, 4.0]],
      "complex sum rounding source values changed")
  of "complex64_cancellation":
    require(values == %*[[16777216.0, 0.0], [1.0, 0.0], [-16777216.0, 0.0]],
      "complex sum cancellation source values changed")
  of "double_precision":
    require(values == %*[
      [9007199254740992, 9007199254740992],
      [1, 1],
      [-9007199254740992, -9007199254740992]],
      "complex sum double-precision source values changed")
  of "packed_columns_cancellation":
    var expectedPackedValues = newJArray()
    for value in [0, 0, 16777216, 1, -16777216]:
      for _ in 0 ..< 17:
        expectedPackedValues.add %*[value, value]
    require(values == expectedPackedValues, "complex sum packed source values changed")
  else: discard
  if nonfinite:
    let specials = field(spec, "nonfinite")
    require(specials.kind == JArray and specials.len > 0, "complex nonfinite source overrides are missing")
    var overridden = initHashSet[string]()
    for special in specials:
      require(special.kind == JObject and special.len == 3,
        "complex nonfinite source fields are incomplete")
      let index = field(special, "index")
      require(index.kind == JInt and index.getInt >= 0 and index.getInt < count,
        "complex nonfinite source index is out of bounds")
      let component = text(field(special, "component"), "nonfinite component")
      require(component in ["real", "imag"], "unknown complex nonfinite component")
      require(text(field(special, "value"), "nonfinite value") in ["nan", "+inf", "-inf"],
        "unknown complex nonfinite value")
      let key = $index.getInt & ":" & component
      require(key notin overridden, "duplicate complex nonfinite source override")
      overridden.incl(key)

proc expectComplexScanDtypeInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == ComplexScanDtypeInputNames.len,
    "complex scan dtype sources are incomplete")
  for name in ComplexScanDtypeInputNames:
    expectComplexScanDtypeInput(field(inputs, name), name)

proc expectComplexScanDtypeTensor(node: JsonNode, shape: seq[int], dtype, label: string) =
  if dtype in ["complex64", "complex128"]:
    expectComplexObservation(node, shape, dtype, label)
  elif dtype in ["float32", "float64"]:
    require(node.kind == JObject and node.len == 6, label & " real tensor fields are incomplete")
    expectTensor(node, false, dtype)
    require(node["shape"] == jsonShape(shape), label & " real tensor shape mismatch")
    let values = node["values"]
    var negativeZero = newJArray()
    for index in 0 ..< values.len:
      let value = values[index]
      require(complexScanFiniteFloat(value) or
        (value.kind == JString and value.getStr in ["nan", "+inf", "-inf"]),
        label & " real tensor component is invalid")
      if complexNegativeZero(value): negativeZero.add newJInt(index)
    require(field(node, "negative_zero") == negativeZero,
      label & " real signed-zero indices do not match values")
  else:
    expectIntegerOutputTensor(node, shape, dtype)

proc expectComplexScanDtypeExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "complex scan dtype observations do not match sources")
  for name, spec in inputs:
    let byDtype = field(expected, name)
    let targets = spec["output_dtypes"]
    require(byDtype.kind == JObject and byDtype.len == targets.len,
      "complex scan output dtype observations are incomplete for " & name)
    let shape = float64LogicalShape(spec, spec["layout"].getStr)
    let count = integerOutputNumel(shape)
    for dtypeNode in targets:
      let dtype = dtypeNode.getStr
      let operations = field(byDtype, dtype)
      require(operations.kind == JObject and operations.len == 2,
        "complex scan operations are incomplete")
      let rejected = shape.len > 0 and count > 0 and
        dtype in ["bool", "uint16", "uint32", "uint64"]
      for operation in ["cumsum", "cumprod"]:
        let observed = field(operations, operation)
        require(observed.kind == JObject and observed.len == 2,
          "complex scan valid/invalid axis groups are incomplete")
        let axes = field(observed, "axes")
        require(axes.kind == JObject and axes.len == spec["axes"].len,
          "complex scan dtype axis observations are incomplete")
        for axisNode in spec["axes"]:
          let directions = field(axes, $axisNode.getInt)
          require(directions.kind == JObject and directions.len == 2,
            "complex scan dtype directions are incomplete")
          for direction in ["forward", "reverse"]:
            let node = field(directions, direction)
            if rejected:
              expectIntegerOutputResult(node, shape, dtype, true,
                "unsupported_output_dtype", "complex scan " & name & "." & dtype)
            else:
              expectComplexScanDtypeTensor(node, shape, dtype,
                "complex scan " & name & "." & dtype & "." & operation)
        let invalidAxes = field(observed, "invalid_axes")
        require(invalidAxes.kind == JObject and invalidAxes.len == spec["invalid_axes"].len,
          "complex scan dtype invalid axis observations are incomplete")
        for axisNode in spec["invalid_axes"]:
          expectComplexInvalidAxis(field(invalidAxes, $axisNode.getInt),
            "complex scan dtype " & name & "." & dtype & "." & operation)

proc expectComplexMeanDtypeInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == ComplexScanDtypeInputNames.len + 4,
    "complex mean dtype sources are incomplete")
  for name in ComplexScanDtypeInputNames:
    expectComplexScanDtypeInput(field(inputs, name), name, true)
  for name in ["rounding", "complex64_cancellation", "double_precision", "packed_columns_cancellation"]:
    expectComplexScanDtypeInput(field(inputs, name), name, true)

proc expectComplexSumDtypeInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == ComplexReductionDtypeInputNames.len,
    "complex sum dtype sources are incomplete")
  for name in ComplexReductionDtypeInputNames:
    expectComplexScanDtypeInput(field(inputs, name), name, sum = true)

proc expectComplexProdDtypeInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == ComplexReductionDtypeInputNames.len,
    "complex product dtype sources are incomplete")
  for name in ComplexReductionDtypeInputNames:
    expectComplexScanDtypeInput(field(inputs, name), name, prod = true)

proc expectComplexMeanResult(node: JsonNode, shape: seq[int], dtype: string,
                             rejected: bool, label: string) =
  if rejected:
    expectIntegerOutputResult(node, shape, dtype, true, "integer_mean_dtype_rejected", label)
  else:
    expectComplexScanDtypeTensor(node, shape, dtype, label)

proc expectComplexMeanDtypeExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "complex mean dtype observations do not match sources")
  for name, spec in inputs:
    let outputs = field(expected, name)
    require(outputs.kind == JObject and outputs.len == spec["output_dtypes"].len,
      "complex mean output dtype observations are incomplete for " & name)
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    for targetNode in spec["output_dtypes"]:
      let target = targetNode.getStr
      let dtype = if target == "default": spec["dtype"].getStr else: target
      let rejected = target in IntegerOutputDtypeNames
      let observed = field(outputs, target)
      require(observed.kind == JObject and observed.len == 3,
        "complex mean whole/valid/invalid axis observations are incomplete")
      let label = "complex mean " & name & "." & target
      expectComplexMeanResult(field(observed, "all"), @[], dtype, rejected, label)
      let axes = field(observed, "axes")
      require(axes.kind == JObject and axes.len == spec["axes"].len,
        "complex mean axis observations are incomplete")
      for axisNode in spec["axes"]:
        var shape = logicalShape
        if shape.len > 0:
          let axis = if axisNode.getInt < 0: axisNode.getInt + shape.len else: axisNode.getInt
          shape[axis] = 1
        expectComplexMeanResult(field(axes, $axisNode.getInt), shape, dtype, rejected, label)
      let invalidAxes = field(observed, "invalid_axes")
      require(invalidAxes.kind == JObject and invalidAxes.len == spec["invalid_axes"].len,
        "complex mean invalid axis observations are incomplete")
      for axisNode in spec["invalid_axes"]:
        let node = field(invalidAxes, $axisNode.getInt)
        if rejected:
          expectComplexMeanResult(node, @[], dtype, true, label)
        else:
          expectComplexInvalidAxis(node, label)

proc expectComplexReductionResult(node: JsonNode, shape: seq[int], dtype: string,
                                  rejected: bool, label: string) =
  if rejected:
    expectIntegerOutputResult(node, shape, dtype, true, "unsupported_output_dtype", label)
  else:
    expectComplexScanDtypeTensor(node, shape, dtype, label)

proc expectComplexReductionDtypeExpected(expected, inputs: JsonNode, operation: string) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "complex " & operation & " dtype observations do not match sources")
  for name, spec in inputs:
    let outputs = field(expected, name)
    require(outputs.kind == JObject and outputs.len == spec["output_dtypes"].len,
      "complex " & operation & " output dtype observations are incomplete for " & name)
    let logicalShape = float64LogicalShape(spec, spec["layout"].getStr)
    let count = integerOutputNumel(logicalShape)
    for targetNode in spec["output_dtypes"]:
      let target = targetNode.getStr
      let dtype = if target == "default": spec["dtype"].getStr else: target
      let rejected = target in ["uint16", "uint32", "uint64"] and count > 0
      let observed = field(outputs, target)
      require(observed.kind == JObject and observed.len == 3,
        "complex " & operation & " whole/valid/invalid axis observations are incomplete")
      let label = "complex " & operation & " " & name & "." & target
      expectComplexReductionResult(field(observed, "all"), @[], dtype, rejected, label)

      let axes = field(observed, "axes")
      require(axes.kind == JObject and axes.len == spec["axes"].len,
        "complex " & operation & " axis observations are incomplete")
      for axisNode in spec["axes"]:
        var shape = logicalShape
        if shape.len > 0:
          let axis = if axisNode.getInt < 0: axisNode.getInt + shape.len else: axisNode.getInt
          shape[axis] = 1
        expectComplexReductionResult(field(axes, $axisNode.getInt), shape, dtype, rejected, label)

      let invalidAxes = field(observed, "invalid_axes")
      require(invalidAxes.kind == JObject and invalidAxes.len == spec["invalid_axes"].len,
        "complex " & operation & " invalid axis observations are incomplete")
      for axisNode in spec["invalid_axes"]:
        expectComplexInvalidAxis(
          field(invalidAxes, $axisNode.getInt), label & ".invalid_axis." & $axisNode.getInt)

proc expectComplexSumDtypeExpected(expected, inputs: JsonNode) =
  expectComplexReductionDtypeExpected(expected, inputs, "sum")

proc expectComplexProdDtypeExpected(expected, inputs: JsonNode) =
  expectComplexReductionDtypeExpected(expected, inputs, "product")

proc extremaRequestIds(inputName: string): seq[string] =
  case inputName
  of "bool", "int8", "int16", "int32", "int64", "uint8", "native_int", "float32",
      "float64", "native_float", "reject_uint16_matrix", "transpose", "offset", "broadcast":
    @[
      "amin_all_drop", "amin_all_keep", "amin_axes_empty_drop", "amin_axes_empty_keep",
      "amin_axis_0_drop", "amin_axis_0_keep", "amin_axis_neg1_drop", "amin_axes_0_1_drop",
      "amin_axes_1_0_keep", "amax_all_drop", "amax_all_keep", "amax_axes_empty_drop",
      "amax_axes_empty_keep", "amax_axis_0_drop", "amax_axis_0_keep", "amax_axis_neg1_drop",
      "amax_axes_0_1_drop", "amax_axes_1_0_keep", "amin_bad_axis_drop", "amax_duplicate_axis_drop"]
  of "reverse", "nan", "infinities", "vector_zero_lane_f32", "vector_zero_tail_f32",
      "vector_zero_lane_f64", "vector_zero_tail_f64":
    @[
      "amin_all_drop", "amax_all_drop", "amin_all_keep", "amax_all_keep",
      "amin_axes_empty_drop", "amax_axes_empty_keep", "amin_axis_0_drop",
      "amax_axis_neg1_keep", "amin_bad_axis_drop", "amax_duplicate_axis_drop"]
  of "scalar", "reject_uint16_scalar", "reject_uint32_scalar", "reject_uint64_scalar",
      "reject_complex64_scalar", "reject_complex128_scalar":
    @[
      "amin_all_drop", "amax_all_keep", "amin_axes_empty_drop", "amax_axes_empty_keep",
      "amin_axis_0_drop", "amax_axis_neg1_keep", "amin_bad_axis_drop", "amax_duplicate_axis_drop"]
  of "multi_axes":
    @[
      "amin_axes_0_2_drop", "amax_axes_2_0_keep", "amin_axes_0_neg1_drop",
      "amax_axes_neg1_0_keep", "amin_axes_empty_drop", "amax_axes_empty_keep",
      "amin_bad_axis_drop", "amax_duplicate_axis_drop"]
  of "empty":
    @[
      "amin_all_drop", "amax_axes_empty_keep", "amin_zero_axis_drop", "amax_axis_0_drop",
      "amin_axes_0_2_drop", "amax_axes_2_0_keep", "amin_duplicate_axis_drop",
      "amax_zero_axis_duplicate_drop", "amin_bad_axis_drop"]
  of "empty_uint16", "empty_uint32", "empty_uint64", "empty_complex64", "empty_complex128":
    @[
      "amin_all_drop", "amax_axes_empty_drop", "amax_axis_1_drop", "amax_axis_1_keep",
      "amin_bad_axis_drop", "amax_duplicate_zero_axis_drop"]
  of "coalesced_axes_f32", "coalesced_axes_f64":
    @["amin_axes_0_1_drop", "amax_axes_0_1_drop"]
  of "packed_columns_f32", "packed_columns_f64":
    @["amin_axis_0_drop", "amax_axis_0_drop", "amin_axis_0_keep", "amax_axis_0_keep"]
  of "tie_zeros", "tie_zeros_reversed":
    @["amin_all_drop", "amax_all_drop"]
  else:
    raise newException(ValueError, "unknown extrema fixture input " & inputName)


proc extremaRequestDimensions(requestId: string, rank: int): JsonNode =
  case requestId
  of "amin_all_drop", "amax_all_drop", "amin_all_keep", "amax_all_keep":
    result = newJNull()
  of "amin_axes_empty_drop", "amax_axes_empty_drop", "amin_axes_empty_keep", "amax_axes_empty_keep":
    result = newJArray()
  of "amin_axis_0_drop", "amax_axis_0_drop", "amin_axis_0_keep", "amax_axis_0_keep":
    result = newJInt(0)
  of "amin_axis_neg1_drop", "amax_axis_neg1_drop", "amin_axis_neg1_keep", "amax_axis_neg1_keep":
    result = newJInt(-1)
  of "amin_axes_0_1_drop", "amax_axes_0_1_drop":
    result = %*[0, 1]
  of "amin_axes_1_0_keep", "amax_axes_1_0_keep":
    result = %*[1, 0]
  of "amin_axes_0_2_drop":
    result = %*[0, 2]
  of "amax_axes_2_0_keep":
    result = %*[2, 0]
  of "amin_axes_0_neg1_drop":
    result = %*[0, -1]
  of "amax_axes_neg1_0_keep":
    result = %*[-1, 0]
  of "amin_bad_axis_drop":
    result = newJInt(if rank == 0: 1 else: rank)
  of "amax_duplicate_axis_drop":
    result = newJArray()
    result.add newJInt(0)
    result.add newJInt(-max(1, rank))
  of "amin_zero_axis_drop", "amax_axis_1_drop", "amax_axis_1_keep":
    result = newJInt(1)
  of "amin_duplicate_axis_drop":
    result = %*[0, -3]
  of "amax_zero_axis_duplicate_drop":
    result = %*[1, 1]
  of "amax_duplicate_zero_axis_drop":
    result = %*[0, 0]
  else:
    raise newException(ValueError, "unknown extrema request " & requestId)


proc extremaInputDtype(inputName: string): string =
  case inputName
  of "bool": "bool"
  of "int8": "int8"
  of "int16": "int16"
  of "int32": "int32"
  of "int64", "native_int": "int64"
  of "uint8": "uint8"
  of "float32", "nan", "vector_zero_lane_f32", "vector_zero_tail_f32",
      "packed_columns_f32", "coalesced_axes_f32": "float32"
  of "float64", "native_float", "transpose", "infinities", "vector_zero_lane_f64",
      "vector_zero_tail_f64", "packed_columns_f64", "coalesced_axes_f64", "tie_zeros",
      "tie_zeros_reversed":
    "float64"
  of "reject_uint16_scalar", "reject_uint16_matrix", "empty_uint16": "uint16"
  of "reject_uint32_scalar", "empty_uint32": "uint32"
  of "reject_uint64_scalar", "empty_uint64": "uint64"
  of "reject_complex64_scalar", "empty_complex64": "complex64"
  of "reject_complex128_scalar", "empty_complex128": "complex128"
  of "offset": "int32"
  of "reverse": "float32"
  of "broadcast": "uint8"
  of "scalar": "int64"
  of "multi_axes": "float32"
  of "empty": "int8"
  else: raise newException(ValueError, "unknown extrema fixture input " & inputName)


proc extremaInputShape(inputName: string): seq[int] =
  case inputName
  of "bool", "int8", "int16", "int32", "int64", "uint8", "native_int", "float32",
      "float64", "native_float", "reject_uint16_matrix", "transpose":
    @[2, 3]
  of "offset": @[3, 4]
  of "broadcast": @[1, 3]
  of "reverse": @[5]
  of "scalar", "reject_uint16_scalar", "reject_uint32_scalar", "reject_uint64_scalar",
      "reject_complex64_scalar", "reject_complex128_scalar":
    @[]
  of "multi_axes": @[2, 3, 2]
  of "empty": @[2, 0, 3]
  of "empty_uint16", "empty_uint32", "empty_uint64", "empty_complex64", "empty_complex128":
    @[0, 3]
  of "nan", "infinities": @[3]
  of "vector_zero_lane_f32": @[16]
  of "vector_zero_tail_f32": @[17]
  of "vector_zero_lane_f64": @[8]
  of "vector_zero_tail_f64": @[9]
  of "coalesced_axes_f32", "coalesced_axes_f64": @[4, 8, 3]
  of "packed_columns_f32": @[2, 17]
  of "packed_columns_f64": @[2, 9]
  of "tie_zeros", "tie_zeros_reversed": @[2]
  else: raise newException(ValueError, "unknown extrema fixture input " & inputName)


proc extremaInputLayout(inputName: string): string =
  case inputName
  of "transpose", "offset", "reverse", "broadcast": inputName
  of "tie_zeros_reversed": "reverse"
  of "coalesced_axes_f32", "coalesced_axes_f64": "permute_210"
  else: "identity"


proc extremaLogicalShape(inputName: string): seq[int] =
  result = extremaInputShape(inputName)
  case extremaInputLayout(inputName)
  of "transpose":
    swap(result[0], result[1])
  of "offset":
    result = @[result[1], result[0] - 1]
  of "broadcast":
    result = @[4, 3]
  of "permute_210":
    result = @[result[2], result[1], result[0]]
  else: discard


proc extremaNumel(shape: seq[int]): int =
  result = 1
  for dimension in shape:
    if dimension == 0:
      return 0
    require(result <= high(int) div dimension, "extrema input shape overflows")
    result *= dimension


proc expectExtremaFields(node: JsonNode, keys: openArray[string], label: string) =
  require(node.kind == JObject and node.len == keys.len, label & " fields mismatch")
  for key in keys:
    require(node.hasKey(key), label & " missing field " & key)


proc expectExtremaRequests(spec: JsonNode, inputName: string) =
  let requests = field(spec, "requests")
  let required = extremaRequestIds(inputName)
  require(requests.kind == JArray and requests.len == required.len,
    "extrema requests do not match required coverage for " & inputName)
  var seen = initHashSet[string]()
  let rank = extremaInputShape(inputName).len
  for request in requests:
    let label = "extrema request " & inputName
    expectExtremaFields(request, ["id", "operation", "dimensions", "keepdim"], label)
    let id = text(field(request, "id"), label & " id")
    require(id in required and id notin seen, "unknown or duplicate extrema request " & id)
    seen.incl(id)
    let operation = if id.startsWith("amin_"): "amin" else: "amax"
    require(text(field(request, "operation"), label & " operation") == operation,
      "extrema request operation mismatch for " & id)
    require(field(request, "dimensions") == extremaRequestDimensions(id, rank),
      "extrema request dimensions mismatch for " & id)
    let keepdim = id.endsWith("_keep")
    require(field(request, "keepdim").kind == JBool and field(request, "keepdim").getBool == keepdim,
      "extrema request keepdim mismatch for " & id)


proc extremaFiniteNumber(node: JsonNode): bool =
  node.kind in {JInt, JFloat} and
    classify(node.getFloat) notin {fcNan, fcInf, fcNegInf}


proc expectExtremaInputs(inputs: JsonNode) =
  const Names = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "native_int", "float32", "float64",
    "native_float", "reject_uint16_scalar", "reject_uint32_scalar", "reject_uint64_scalar",
    "reject_complex64_scalar", "reject_complex128_scalar", "reject_uint16_matrix", "transpose",
    "offset", "reverse", "broadcast", "scalar", "multi_axes", "empty", "empty_uint16",
    "empty_uint32", "empty_uint64", "empty_complex64", "empty_complex128", "nan", "infinities",
    "vector_zero_lane_f32", "vector_zero_tail_f32", "vector_zero_lane_f64",
    "vector_zero_tail_f64", "packed_columns_f32", "packed_columns_f64",
    "coalesced_axes_f32", "coalesced_axes_f64", "tie_zeros", "tie_zeros_reversed"]
  require(inputs.kind == JObject and inputs.len == Names.len,
    "extrema inputs do not match required cases")
  for name in Names:
    require(inputs.hasKey(name), "missing extrema input " & name)
    let spec = inputs[name]
    let dtype = extremaInputDtype(name)
    let payloadKey = if dtype in ["bool", "int8", "int16", "int32", "int64", "uint8",
        "uint16", "uint32", "uint64"]: "words" else: "values"
    expectExtremaFields(spec, ["dtype", "shape", "layout", payloadKey, "specials", "requests"],
      "extrema input " & name)
    require(text(field(spec, "dtype"), "extrema dtype") == dtype,
      "extrema dtype mismatch for " & name)
    let shape = extremaInputShape(name)
    require(field(spec, "shape") == jsonShape(shape), "extrema input shape mismatch for " & name)
    require(text(field(spec, "layout"), "extrema layout") == extremaInputLayout(name),
      "extrema layout mismatch for " & name)
    let count = extremaNumel(shape)
    let payload = field(spec, payloadKey)
    require(payload.kind == JArray and payload.len == count, "extrema payload length mismatch for " & name)
    if payloadKey == "words":
      for wordNode in payload:
        require(wordNode.kind == JInt, "extrema integer payload must use exact integer words")
        let word = wordNode.getInt
        case dtype
        of "bool": require(word in [0, 1], "extrema bool input words must be zero or one")
        of "int8": require(word >= low(int8).int and word <= high(int8).int, "extrema int8 word out of range")
        of "int16": require(word >= low(int16).int and word <= high(int16).int, "extrema int16 word out of range")
        of "int32": require(word >= low(int32).int and word <= high(int32).int, "extrema int32 word out of range")
        of "uint8": require(word >= 0 and word <= high(uint8).int, "extrema uint8 word out of range")
        of "uint16": require(word >= 0 and word <= high(uint16).int, "extrema uint16 word out of range")
        of "uint32": require(word >= 0 and word <= high(uint32).int, "extrema uint32 word out of range")
        else: discard
    elif dtype in ["float32", "float64"]:
      for value in payload:
        require(extremaFiniteNumber(value), "extrema float input values must be finite numbers")
    else:
      for pair in payload:
        require(pair.kind == JArray and pair.len == 2 and
          extremaFiniteNumber(pair[0]) and extremaFiniteNumber(pair[1]),
          "extrema complex input values must be real/imaginary pairs")
    let specials = field(spec, "specials")
    require(specials.kind == JArray, "extrema nonfinite values must be an array")
    var seenSpecials = initHashSet[int]()
    for special in specials:
      expectExtremaFields(special, ["index", "value"], "extrema nonfinite value")
      let indexNode = field(special, "index")
      require(indexNode.kind == JInt and indexNode.getInt >= 0 and indexNode.getInt < count,
        "extrema nonfinite value index out of bounds")
      let index = indexNode.getInt
      require(index notin seenSpecials, "duplicate extrema nonfinite value index")
      seenSpecials.incl(index)
      require(dtype in ["float32", "float64"] and
        text(field(special, "value"), "extrema nonfinite value") in ["nan", "+inf", "-inf"],
        "invalid extrema nonfinite value")
    if name == "nan":
      require(specials == %*[{"index": 1, "value": "nan"}], "nan extrema input must carry its pinned NaN")
    elif name == "infinities":
      require(specials == %*[{"index": 0, "value": "+inf"}, {"index": 2, "value": "-inf"}],
        "infinity extrema input must carry pinned infinities")
    else:
      require(specials.len == 0, "unexpected extrema nonfinite values for " & name)
    expectExtremaRequests(spec, name)


proc extremaExpectedError(inputName: string, spec, request: JsonNode): string =
  let shape = extremaLogicalShape(inputName)
  let dimensions = field(request, "dimensions")
  var axes: seq[int]
  let allAxes = dimensions.kind == JNull or (dimensions.kind == JArray and dimensions.len == 0)
  if dimensions.kind == JInt:
    axes.add dimensions.getInt
  elif dimensions.kind == JArray:
    for axis in dimensions: axes.add axis.getInt
  elif dimensions.kind != JNull:
    raise newException(ValueError, "invalid extrema expected dimensions")

  var empty = false
  for dimension in shape:
    if dimension == 0: empty = true
  if empty:
    if allAxes: return "empty_reduction"
  let axisLimit = max(1, shape.len)
  for axis in axes:
    if axis < -axisLimit or axis >= axisLimit:
      return "axis_out_of_range"
    if empty and shape.len > 0:
      let normalized = if axis < 0: axis + axisLimit else: axis
      if shape[normalized] == 0:
        return "empty_reduced_axis"
  var seen = initHashSet[int]()
  for axis in axes:
    let normalized = if axis < 0: axis + axisLimit else: axis
    if normalized in seen:
      return "duplicate_axes"
    seen.incl(normalized)
  let dtype = spec["dtype"].getStr
  if not empty and dtype in ["complex64", "complex128", "uint16", "uint32", "uint64"]:
    return "unsupported_dtype"


proc extremaOutputShape(shape: seq[int], dimensions: JsonNode, keepdim: bool): seq[int] =
  var reduced = newSeq[bool](shape.len)
  if dimensions.kind == JNull or (dimensions.kind == JArray and dimensions.len == 0):
    for axis in 0 ..< shape.len: reduced[axis] = true
  elif dimensions.kind == JInt:
    if shape.len > 0:
      let axis = if dimensions.getInt < 0: dimensions.getInt + shape.len else: dimensions.getInt
      reduced[axis] = true
  else:
    for dimension in dimensions:
      if shape.len > 0:
        let axis = if dimension.getInt < 0: dimension.getInt + shape.len else: dimension.getInt
        reduced[axis] = true
  for axis, dimension in shape:
    if reduced[axis]:
      if keepdim: result.add 1
    else:
      result.add dimension


proc expectExtremaTensor(node: JsonNode, shape: seq[int], dtype, label: string) =
  let fieldNames = if dtype == "uint64":
    @["shape", "numel", "dtype", "device", "values", "word_encoding"]
  else:
    @["shape", "numel", "dtype", "device", "values"]
  expectExtremaFields(node, fieldNames, label)
  require(field(node, "shape") == jsonShape(shape), label & " shape mismatch")
  require(field(node, "numel").kind == JInt and field(node, "numel").getInt == extremaNumel(shape),
    label & " numel mismatch")
  require(text(field(node, "dtype"), label & " dtype") == dtype and
    text(field(node, "device"), label & " device") == "cpu", label & " dtype/device mismatch")
  if dtype == "uint64":
    require(text(field(node, "word_encoding"), label & " word encoding") == "signed_twos_complement",
      label & " uint64 word encoding mismatch")
  let values = field(node, "values")
  require(values.kind == JArray and values.len == extremaNumel(shape), label & " values length mismatch")
  for value in values:
    if dtype == "bool":
      require(value.kind == JBool, label & " bool values must be JSON booleans")
    elif dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
      require(value.kind == JInt, label & " integer values must be exact JSON integers")
    elif dtype in ["complex64", "complex128"]:
      require(false, label & " nonempty complex output is unsupported")
    else:
      require(extremaFiniteNumber(value) or
        (value.kind == JString and value.getStr in ["nan", "+inf", "-inf"]),
        label & " invalid floating-point output value")


proc expectExtremaExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == inputs.len,
    "extrema expected inputs do not match sources")
  for inputName, spec in inputs:
    let byRequest = field(expected, inputName)
    require(byRequest.kind == JObject and byRequest.len == spec["requests"].len,
      "extrema expected requests are incomplete for " & inputName)
    let shape = extremaLogicalShape(inputName)
    for request in spec["requests"]:
      let requestId = request["id"].getStr
      let observed = field(byRequest, requestId)
      let label = "extrema " & inputName & "." & requestId
      let expectedError = extremaExpectedError(inputName, spec, request)
      if expectedError.len > 0:
        expectExtremaFields(observed, ["error"], label)
        require(text(field(observed, "error"), label & " error") == expectedError,
          label & " error classification mismatch")
      else:
        let outputShape = extremaOutputShape(shape, request["dimensions"], request["keepdim"].getBool)
        expectExtremaTensor(observed, outputShape, spec["dtype"].getStr, label)


const
  IndexedExtremaDtypeNames = [
    "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
    "uint16", "uint32", "uint64", "complex64", "complex128"]
  IndexedExtremaCommonCaseNames = [
    "ties", "scalar", "singleton", "singleton_axis", "empty_outer", "empty_reduced",
    "empty_singleton", "empty_scalar_axis", "transpose", "offset", "reverse", "broadcast",
    "permuted"]


proc indexedExtremaCaseNames(dtype: string): seq[string] =
  for name in IndexedExtremaCommonCaseNames: result.add name
  if dtype in ["float32", "float64"]:
    result.add "nan_first"
    result.add "zero_ties"
    result.add "infinities"
  elif dtype notin ["bool", "complex64", "complex128"]:
    result.add "integer_bounds"


proc indexedExtremaInputShape(name: string): seq[int] =
  case name
  of "ties", "transpose": @[2, 3]
  of "scalar": @[]
  of "singleton": @[1]
  of "singleton_axis": @[2, 1, 3]
  of "empty_outer": @[0, 3]
  of "empty_reduced": @[2, 0]
  of "empty_singleton": @[0, 1]
  of "empty_scalar_axis": @[0]
  of "offset": @[3, 4]
  of "reverse": @[5]
  of "broadcast": @[1, 3]
  of "permuted": @[4, 8, 3]
  of "nan_first", "zero_ties", "infinities": @[2, 3]
  of "integer_bounds": @[2, 2]
  else: invalid("unknown indexed extrema input " & name)


proc indexedExtremaInputLayout(name: string): string =
  case name
  of "transpose", "offset", "reverse", "broadcast": name
  of "permuted": "permute"
  else: "identity"


proc indexedExtremaInputAxes(name: string): JsonNode =
  result = newJArray()
  result.add newJNull()
  let axes = case name
    of "ties": @[0, 1, -1, 2, -3]
    of "scalar": @[0, -1, 1, -2]
    of "singleton": @[0, -1, 1]
    of "singleton_axis": @[1, -2]
    of "empty_outer": @[0, 1, 2]
    of "empty_reduced": @[0, 1, -1]
    of "empty_singleton": @[0, 1]
    of "empty_scalar_axis": @[0, -1]
    of "transpose", "offset", "broadcast", "nan_first", "zero_ties", "infinities",
        "integer_bounds": @[0, 1]
    of "reverse": @[0, -1]
    of "permuted": @[0, 1, 2]
    else: invalid("unknown indexed extrema input axes " & name)
  for axis in axes: result.add newJInt(axis)


proc indexedExtremaCommonValues(name: string): seq[int64] =
  case name
  of "ties", "transpose", "singleton_axis": result = @[2'i64, 1, 3, 1, 3, 3]
  of "scalar", "singleton": result = @[3'i64]
  of "empty_outer", "empty_reduced", "empty_singleton", "empty_scalar_axis": result = @[]
  of "offset":
    for index in 0 ..< 12: result.add (index mod 3).int64
  of "reverse": result = @[2'i64, 1, 3, 1, 3]
  of "broadcast": result = @[1'i64, 3, 2]
  of "permuted":
    for index in 0 ..< 96: result.add (index mod 7).int64
  else: invalid("unknown indexed extrema common values " & name)


proc indexedExtremaInputWords(name, dtype: string): seq[int64] =
  if name != "integer_bounds":
    return indexedExtremaCommonValues(name)
  case dtype
  of "int8":
    result = @[high(int8).int64, high(int8).int64 - 1, low(int8).int64, high(int8).int64]
  of "int16":
    result = @[high(int16).int64, high(int16).int64 - 1, low(int16).int64, high(int16).int64]
  of "int32":
    result = @[high(int32).int64, high(int32).int64 - 1, low(int32).int64, high(int32).int64]
  of "int64":
    result = @[high(int64), high(int64) - 1, low(int64), high(int64)]
  of "uint8":
    result = @[high(uint8).int64, high(uint8).int64 - 1, 0'i64, high(uint8).int64]
  of "uint16":
    result = @[high(uint16).int64, high(uint16).int64 - 1, 0'i64, high(uint16).int64]
  of "uint32":
    result = @[high(uint32).int64, high(uint32).int64 - 1, 0'i64, high(uint32).int64]
  of "uint64":
    result = @[-1'i64, -2, 0, -1]
  else: invalid("indexed extrema bounds are invalid for " & dtype)


proc indexedExtremaRealValues(name: string): JsonNode =
  case name
  of "ties", "transpose", "singleton_axis": result = %*[2, 1, 3, 1, 3, 3]
  of "scalar", "singleton": result = %*[3]
  of "empty_outer", "empty_reduced", "empty_singleton", "empty_scalar_axis":
    result = newJArray()
  of "offset": result = %*[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
  of "reverse": result = %*[2, 1, 3, 1, 3]
  of "broadcast": result = %*[1, 3, 2]
  of "permuted":
    result = newJArray()
    for index in 0 ..< 96: result.add newJInt(index mod 7)
  of "nan_first": result = %*[2, "nan", "+inf", "nan", "-inf", "nan"]
  of "zero_ties": result = %*["-0", 0, "-0", 0, "-0", 0]
  of "infinities": result = %*["-inf", "-inf", "+inf", "+inf", "+inf", "-inf"]
  else: invalid("unknown indexed extrema real values " & name)


proc indexedExtremaComplexValues(name: string): JsonNode =
  result = newJArray()
  for value in indexedExtremaCommonValues(name):
    result.add %*[value, 0]


proc indexedExtremaLogicalShape(shape: seq[int], layout: string): seq[int] =
  result = shape
  case layout
  of "transpose": swap(result[0], result[1])
  of "offset": result = @[shape[1], shape[0] - 1]
  of "broadcast": result = @[4, 3]
  of "permute": result = @[shape[2], shape[1], shape[0]]
  else: discard


proc indexedExtremaRequestId(operation: string, axis: JsonNode, keepdim: bool): string =
  let axisName = if axis.kind == JNull: "all" else: "axis_" & $axis.getInt
  result = operation & "_" & axisName & (if keepdim: "_keep" else: "_drop")


proc indexedExtremaExpectedError(dtype: string, shape: seq[int], axis: JsonNode): string =
  if dtype in ["bool", "complex64", "complex128"]:
    return "unsupported_dtype"
  let rank = shape.len
  let axisLimit = max(1, rank)
  var normalizedAxis = -1
  if axis.kind == JInt:
    let value = axis.getInt
    if value < -axisLimit or value >= axisLimit:
      return "axis_out_of_range"
    if rank > 0:
      normalizedAxis = if value < 0: value + rank else: value
  var emptyInput = false
  for dimension in shape:
    if dimension == 0: emptyInput = true
  if emptyInput:
    if axis.kind == JNull: return "empty_reduction"
    if rank > 0 and shape[normalizedAxis] == 0: return "empty_reduced_axis"
  if dtype in ["uint16", "uint32", "uint64"]:
    if rank == 0 or axis.kind == JNull:
      return "unsupported_dtype"
    if shape[normalizedAxis] == 1 or emptyInput:
      return ""
    return "unsupported_dtype"


proc indexedExtremaOutputShape(shape: seq[int], axis: JsonNode, keepdim: bool): seq[int] =
  var reduced = newSeq[bool](shape.len)
  if axis.kind == JNull:
    for dimension in 0 ..< shape.len: reduced[dimension] = true
  elif shape.len > 0:
    let value = axis.getInt
    let normalized = if value < 0: value + shape.len else: value
    reduced[normalized] = true
  for dimension, size in shape:
    if reduced[dimension]:
      if keepdim: result.add 1
    else:
      result.add size


proc expectIndexedExtremaInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == IndexedExtremaDtypeNames.len,
    "indexed extrema inputs do not match required dtypes")
  for dtype in IndexedExtremaDtypeNames:
    let cases = field(inputs, dtype)
    let caseNames = indexedExtremaCaseNames(dtype)
    require(cases.kind == JObject and cases.len == caseNames.len,
      "indexed extrema cases do not match required coverage for " & dtype)
    for name in caseNames:
      let spec = field(cases, name)
      let integerInput = dtype in [
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
      let payloadKey = if integerInput: "words" else: "values"
      expectExtremaFields(spec, ["shape", "layout", "axes", payloadKey],
        "indexed extrema input " & dtype & "." & name)
      let shape = indexedExtremaInputShape(name)
      require(field(spec, "shape") == jsonShape(shape),
        "indexed extrema input shape mismatch for " & dtype & "." & name)
      require(text(field(spec, "layout"), "indexed extrema layout") ==
        indexedExtremaInputLayout(name), "indexed extrema layout mismatch for " & name)
      require(field(spec, "axes") == indexedExtremaInputAxes(name),
        "indexed extrema axes mismatch for " & name)
      let payload = field(spec, payloadKey)
      let expectedPayload = if integerInput:
          block:
            var values = newJArray()
            for word in indexedExtremaInputWords(name, dtype): values.add newJInt(word)
            values
        elif dtype in ["complex64", "complex128"]:
          indexedExtremaComplexValues(name)
        else:
          indexedExtremaRealValues(name)
      require(payload == expectedPayload,
        "indexed extrema input values mismatch for " & dtype & "." & name)


proc expectIndexedExtremaExpected(expected, inputs: JsonNode) =
  require(expected.kind == JObject and expected.len == IndexedExtremaDtypeNames.len,
    "indexed extrema expected dtypes are incomplete")
  for dtype in IndexedExtremaDtypeNames:
    let byCase = field(expected, dtype)
    let caseNames = indexedExtremaCaseNames(dtype)
    require(byCase.kind == JObject and byCase.len == caseNames.len,
      "indexed extrema expected cases are incomplete for " & dtype)
    for name in caseNames:
      let spec = inputs[dtype][name]
      let byRequest = field(byCase, name)
      let axes = field(spec, "axes")
      require(byRequest.kind == JObject and byRequest.len == axes.len * 4,
        "indexed extrema request observations are incomplete for " & dtype & "." & name)
      let shape = indexedExtremaLogicalShape(indexedExtremaInputShape(name),
        indexedExtremaInputLayout(name))
      for operation in ["argmin", "argmax"]:
        for axis in axes:
          for keepdim in [false, true]:
            let requestId = indexedExtremaRequestId(operation, axis, keepdim)
            let observed = field(byRequest, requestId)
            let label = "indexed extrema " & dtype & "." & name & "." & requestId
            let expectedError = indexedExtremaExpectedError(dtype, shape, axis)
            if expectedError.len > 0:
              expectExtremaFields(observed, ["error"], label)
              require(text(field(observed, "error"), label & " error") == expectedError,
                label & " error classification mismatch")
            else:
              let outputShape = indexedExtremaOutputShape(shape, axis, keepdim)
              expectExtremaFields(observed, ["shape", "numel", "dtype", "device", "values"], label)
              expectTensor(observed, false, "int64")
              require(field(observed, "shape") == jsonShape(outputShape),
                label & " shape mismatch")
              let values = field(observed, "values")
              let limit = if axis.kind == JNull:
                  extremaNumel(shape)
                elif shape.len == 0:
                  1
                else:
                  let value = axis.getInt
                  shape[if value < 0: value + shape.len else: value]
              for value in values:
                require(value.getBiggestInt >= 0 and value.getBiggestInt < limit.int64,
                  label & " index is outside its reduced dimension")


const ExtremaGradientInputNames = [
  "rows_ties", "columns_ties", "multi_axes", "empty_axes_all", "whole", "nan_row",
  "infinite_values", "infinite_seed", "negative_zero_seed", "nan_seed", "scalar_negative_zero",
  "scalar_nan", "scalar_infinity", "empty_outer", "empty_middle", "transpose", "offset",
  "reverse", "broadcast", "permuted", "strided_seed"]


proc extremaGradientInputShape(name: string): seq[int] =
  case name
  of "rows_ties", "whole", "nan_row", "infinite_values", "infinite_seed",
      "negative_zero_seed", "nan_seed", "transpose", "strided_seed":
    @[2, 3]
  of "columns_ties": @[3, 2]
  of "multi_axes": @[2, 3, 2]
  of "empty_axes_all": @[7]
  of "scalar_negative_zero", "scalar_nan", "scalar_infinity": @[]
  of "empty_outer": @[0, 3]
  of "empty_middle": @[2, 0, 3]
  of "offset": @[3, 4]
  of "reverse": @[5]
  of "broadcast": @[1, 3]
  of "permuted": @[4, 8, 3]
  else: invalid("unknown extrema gradient input " & name)


proc extremaGradientLayout(name: string): string =
  case name
  of "transpose", "offset", "reverse", "broadcast": name
  of "permuted": "permute"
  else: "identity"


proc extremaGradientDimensions(name: string): JsonNode =
  case name
  of "rows_ties", "nan_row", "infinite_values", "infinite_seed", "negative_zero_seed",
      "transpose", "empty_outer": result = %1
  of "columns_ties", "offset", "reverse", "broadcast", "strided_seed": result = %0
  of "multi_axes": result = %*[0, -1]
  of "empty_middle": result = %*[0, 2]
  of "permuted": result = %*[0, 1]
  of "empty_axes_all": result = newJArray()
  of "whole", "scalar_nan": result = newJNull()
  of "nan_seed", "scalar_infinity": result = %0
  of "scalar_negative_zero": result = %(-1)
  else: invalid("unknown extrema gradient dimensions " & name)


proc extremaGradientSpecials(name: string): JsonNode =
  case name
  of "nan_row": result = %*[{"index": 1, "value": "nan"}]
  of "infinite_values":
    result = %*[{"index": 0, "value": "-inf"}, {"index": 1, "value": "-inf"},
                {"index": 3, "value": "+inf"}, {"index": 5, "value": "+inf"}]
  of "scalar_nan": result = %*[{"index": 0, "value": "nan"}]
  of "scalar_infinity": result = %*[{"index": 0, "value": "+inf"}]
  else: result = newJArray()


proc extremaGradientSeedSpec(name: string): JsonNode =
  case name
  of "rows_ties", "nan_row":
    result = %*{"kind": "values", "values": [2, -3]}
  of "infinite_seed":
    result = %*{"kind": "values", "values": ["+inf", "-inf"]}
  of "negative_zero_seed":
    result = %*{"kind": "constant", "value": "-0"}
  of "nan_seed":
    result = %*{"kind": "constant", "value": "nan"}
  of "strided_seed":
    result = %*{"kind": "strided_sequence"}
  else:
    result = %*{"kind": "constant", "value": -3.0}


proc extremaGradientInputValues(name: string): JsonNode =
  case name
  of "rows_ties", "whole", "infinite_seed", "negative_zero_seed", "nan_seed",
      "transpose", "strided_seed":
    result = %*[1, 1, 3, 3, 3, 2]
  of "columns_ties": result = %*[-2, 4, -2, 4, 1, 3]
  of "multi_axes":
    result = newJArray()
    for index in 0 ..< 12: result.add newJInt(index mod 3)
  of "empty_axes_all": result = %*[3, 3, 3, 3, 3, 3, 3]
  of "nan_row": result = %*[1, 0, -2, 3, 3, 1]
  of "infinite_values": result = %*[0, 0, 3, 0, 1, 0]
  of "scalar_negative_zero":
    result = newJArray()
    result.add newJFloat(-0.0)
  of "scalar_nan", "scalar_infinity": result = %*[0]
  of "empty_outer", "empty_middle": result = newJArray()
  of "offset":
    result = newJArray()
    for index in 0 ..< 12: result.add newJInt(index mod 3)
  of "reverse": result = %*[2, 1, 3, 1, 3]
  of "broadcast": result = %*[1, 3, 2]
  of "permuted":
    result = newJArray()
    for index in 0 ..< 96: result.add newJInt(index mod 7)
  else: invalid("unknown extrema gradient values " & name)


proc extremaGradientOutputShape(shape: seq[int], dimensions: JsonNode,
                                keepdim: bool): seq[int] =
  var reduced = newSeq[bool](shape.len)
  let allAxes = dimensions.kind == JNull or (dimensions.kind == JArray and dimensions.len == 0)
  if allAxes:
    for axis in 0 ..< shape.len: reduced[axis] = true
  elif dimensions.kind == JInt:
    if shape.len > 0:
      let axis = if dimensions.getInt < 0: dimensions.getInt + shape.len else: dimensions.getInt
      reduced[axis] = true
  else:
    for dimension in dimensions:
      if shape.len > 0:
        let rawAxis = dimension.getInt
        let axis = if rawAxis < 0: rawAxis + shape.len else: rawAxis
        reduced[axis] = true
  for axis, dimension in shape:
    if reduced[axis]:
      if keepdim: result.add 1
    else:
      result.add dimension


proc extremaGradientSourceShape(name: string): seq[int] =
  result = extremaGradientInputShape(name)
  case extremaGradientLayout(name)
  of "transpose": swap(result[0], result[1])
  of "offset": result = @[result[1], result[0] - 1]
  of "broadcast": result = @[4, 3]
  of "permute": result = @[result[2], result[1], result[0]]
  else: discard


proc isNegativeZeroValue(value: JsonNode): bool =
  value.kind == JFloat and value.getFloat == 0.0 and
    (cast[uint64](value.getFloat) shr 63) == 1'u64


proc expectExtremaGradientTensor(node: JsonNode, shape: seq[int], dtype, label: string,
                                 gradient: bool) =
  if gradient:
    expectExtremaFields(node, ["shape", "numel", "dtype", "device", "values",
      "negative_zero_indices"], label)
  else:
    expectExtremaFields(node, ["shape", "numel", "dtype", "device", "values"], label)
  expectTensor(node, false, dtype)
  require(field(node, "shape") == jsonShape(shape), label & " shape mismatch")
  if gradient:
    let values = field(node, "values")
    let indices = field(node, "negative_zero_indices")
    require(indices.kind == JArray, label & " negative_zero_indices must be an array")
    var expectedIndices = newJArray()
    for index in 0 ..< values.len:
      if isNegativeZeroValue(values[index]): expectedIndices.add newJInt(index)
    var previous = -1
    for index in indices:
      require(index.kind == JInt and index.getInt > previous and index.getInt < values.len,
        label & " negative_zero_indices must be sorted unique in-range indices")
      previous = index.getInt
    require(indices == expectedIndices,
      label & " negative_zero_indices do not match gradient value sign bits")


proc expectExtremaGradientInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == ExtremaGradientInputNames.len,
    "extrema gradient inputs do not match required probe cases")
  var operations = newJArray()
  for operation in ["amin", "amax"]:
    for keepdim in [false, true]:
      operations.add %*{"operation": operation, "keepdim": keepdim}
  for name in ExtremaGradientInputNames:
    require(inputs.hasKey(name), "missing extrema gradient input " & name)
    let spec = field(inputs, name)
    expectExtremaFields(spec, ["x", "layout", "nonfinite", "axes", "seed", "operations"],
      "extrema gradient input " & name)
    let x = field(spec, "x")
    expectExtremaFields(x, ["shape", "values"], "extrema gradient source " & name)
    expectInputTensor(x)
    if name == "scalar_negative_zero":
      require(isNegativeZeroValue(field(x, "values")[0]),
        "extrema gradient source negative zero sign mismatch")
    else:
      require(field(x, "values") == extremaGradientInputValues(name),
        "extrema gradient source values mismatch for " & name)
    require(field(x, "shape") == jsonShape(extremaGradientInputShape(name)),
      "extrema gradient source shape mismatch for " & name)
    require(text(field(spec, "layout"), "extrema gradient layout") == extremaGradientLayout(name),
      "extrema gradient layout mismatch for " & name)
    require(field(spec, "nonfinite") == extremaGradientSpecials(name),
      "extrema gradient nonfinite source mismatch for " & name)
    require(field(spec, "axes") == extremaGradientDimensions(name),
      "extrema gradient dimensions mismatch for " & name)
    require(field(spec, "seed") == extremaGradientSeedSpec(name),
      "extrema gradient seed mismatch for " & name)
    require(field(spec, "operations") == operations,
      "extrema gradient operations must cover amin/amax and both keepdim modes for " & name)


proc expectExtremaGradientExpected(expected, inputs: JsonNode) =
  expectExtremaFields(expected, ["float32", "float64"], "extrema gradient dtypes")
  for dtype in ["float32", "float64"]:
    let byInput = field(expected, dtype)
    expectExtremaFields(byInput, ExtremaGradientInputNames, "extrema gradient cases for " & dtype)
    for name in ExtremaGradientInputNames:
      let spec = inputs[name]
      let requests = spec["operations"]
      let observations = field(byInput, name)
      require(observations.kind == JArray and observations.len == requests.len,
        "extrema gradient observations are incomplete for " & dtype & "." & name)
      let sourceShape = extremaGradientSourceShape(name)
      for index in 0 ..< requests.len:
        let request = requests[index]
        let observed = observations[index]
        let label = "extrema gradient " & dtype & "." & name & "[" & $index & "]"
        expectExtremaFields(observed, ["output", "grad"], label)
        let outputShape = extremaGradientOutputShape(
          sourceShape, spec["axes"], request["keepdim"].getBool)
        expectExtremaGradientTensor(field(observed, "output"), outputShape, dtype,
          label & " output", false)
        expectExtremaGradientTensor(field(observed, "grad"), sourceShape, dtype,
          label & " gradient", true)


proc expectPairedExtremaInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == IndexedExtremaDtypeNames.len,
    "paired extrema inputs do not match required dtypes")
  var indexedInputs = newJObject()
  for dtype in IndexedExtremaDtypeNames:
    let cases = field(inputs, dtype)
    let caseNames = indexedExtremaCaseNames(dtype)
    require(cases.kind == JObject and cases.len == caseNames.len,
      "paired extrema cases do not match required coverage for " & dtype)
    var indexedCases = newJObject()
    for name in caseNames:
      let spec = field(cases, name)
      let axes = field(spec, "axes")
      require(axes.kind == JArray, "paired extrema axes must be an array")
      var indexedAxes = newJArray()
      indexedAxes.add newJNull()
      for axis in axes:
        require(axis.kind == JInt, "paired extrema axes must omit whole-reduction null")
        indexedAxes.add axis
      var indexedSpec = newJObject()
      for key, value in spec:
        if key != "axes": indexedSpec[key] = value
      indexedSpec["axes"] = indexedAxes
      indexedCases[name] = indexedSpec
    indexedInputs[dtype] = indexedCases
  expectIndexedExtremaInputs(indexedInputs)


proc pairedExtremaExpectedError(dtype: string, shape: seq[int], axis: JsonNode): string =
  let rank = shape.len
  let value = axis.getInt
  let axisLimit = max(1, rank)
  if value < -axisLimit or value >= axisLimit:
    return "axis_out_of_range"
  if rank > 0:
    let normalizedAxis = if value < 0: value + rank else: value
    if shape[normalizedAxis] == 0:
      return "empty_reduced_axis"
  if dtype in ["complex64", "complex128"]:
    return "unsupported_dtype"
  if dtype in ["uint16", "uint32", "uint64"]:
    if rank == 0:
      return ""
    for dimension in shape:
      if dimension == 0:
        return ""
    return "unsupported_dtype"


proc expectPairedExtremaValueTensor(node: JsonNode, shape: seq[int],
                                    dtype, label: string) =
  if dtype in ["float32", "float64"]:
    expectExtremaGradientTensor(node, shape, dtype, label, true)
  else:
    expectIntegerOutputTensor(node, shape, dtype)


proc expectPairedExtremaExpected(expected, inputs: JsonNode) =
  expectExtremaFields(expected, IndexedExtremaDtypeNames, "paired extrema dtypes")
  for dtype in IndexedExtremaDtypeNames:
    let byCase = field(expected, dtype)
    let caseNames = indexedExtremaCaseNames(dtype)
    expectExtremaFields(byCase, caseNames, "paired extrema cases for " & dtype)
    for name in caseNames:
      let spec = inputs[dtype][name]
      let axes = field(spec, "axes")
      let byRequest = field(byCase, name)
      require(byRequest.kind == JObject and byRequest.len == axes.len * 4,
        "paired extrema request observations are incomplete for " & dtype & "." & name)
      let shape = indexedExtremaLogicalShape(indexedExtremaInputShape(name),
        indexedExtremaInputLayout(name))
      for operation in ["min", "max"]:
        for axis in axes:
          for keepdim in [false, true]:
            let requestId = indexedExtremaRequestId(operation, axis, keepdim)
            let observed = field(byRequest, requestId)
            let label = "paired extrema " & dtype & "." & name & "." & requestId
            let expectedError = pairedExtremaExpectedError(dtype, shape, axis)
            if expectedError.len > 0:
              expectExtremaFields(observed, ["error"], label)
              require(text(field(observed, "error"), label & " error") == expectedError,
                label & " error precedence mismatch")
            else:
              expectExtremaFields(observed, ["values", "indices"], label)
              let outputShape = indexedExtremaOutputShape(shape, axis, keepdim)
              expectPairedExtremaValueTensor(field(observed, "values"), outputShape,
                dtype, label & " values")
              expectIntegerOutputTensor(field(observed, "indices"), outputShape, "int64")
              let value = axis.getInt
              let limit = if shape.len == 0: 1
                else: shape[if value < 0: value + shape.len else: value]
              for index in field(observed, "indices")["values"]:
                require(index.getBiggestInt >= 0 and index.getBiggestInt < limit.int64,
                  label & " index is outside its reduced dimension")


const AminmaxInputNames = [
  "bool", "int8", "int16", "int32", "int64", "uint8", "transpose", "offset",
  "reverse", "broadcast", "scalar", "empty_reduced", "empty_outer", "zero_ties_f32",
  "zero_ties_f64", "payload_f32", "payload_f64", "scalar_payload_f32",
  "scalar_payload_f64", "infinities_f32", "infinities_f64", "uint16_matrix",
  "uint32_singleton", "uint64_empty", "uint16_scalar", "uint32_scalar", "uint64_scalar",
  "complex64_empty", "complex128_scalar"]


proc aminmaxInputDtype(name: string): string =
  case name
  of "bool": result = "bool"
  of "int8", "int16", "int32", "int64", "uint8", "uint16_matrix", "uint16_scalar",
      "uint32_singleton", "uint32_scalar", "uint64_empty", "uint64_scalar",
      "scalar", "offset", "broadcast", "empty_reduced", "empty_outer":
    result = case name
      of "uint16_matrix", "uint16_scalar": "uint16"
      of "uint32_singleton", "uint32_scalar": "uint32"
      of "uint64_empty", "uint64_scalar": "uint64"
      of "scalar", "int64": "int64"
      of "int8", "empty_reduced", "empty_outer": "int8"
      of "int16": "int16"
      of "int32", "offset": "int32"
      of "uint8", "broadcast": "uint8"
      else: name
  of "transpose", "zero_ties_f64", "payload_f64", "scalar_payload_f64", "infinities_f64":
    result = "float64"
  of "reverse", "zero_ties_f32", "payload_f32", "scalar_payload_f32", "infinities_f32":
    result = "float32"
  of "complex64_empty": result = "complex64"
  of "complex128_scalar": result = "complex128"
  else: invalid("unknown aminmax input " & name)


proc aminmaxInputShape(name: string): seq[int] =
  case name
  of "bool", "int8", "int64", "uint8", "zero_ties_f32", "zero_ties_f64",
      "infinities_f32", "infinities_f64", "uint16_matrix", "complex128_scalar":
    if name == "complex128_scalar": result = @[]
    elif name == "int64" or name == "bool" or name == "int8" or name == "uint8" or
        name.startsWith("zero_ties") or name.startsWith("infinities") or
        name == "uint16_matrix":
      result = @[2, 3]
  of "int16", "int32", "payload_f64": result = @[2, 2]
  of "transpose", "payload_f32": result = @[2, if name == "transpose": 3 else: 4]
  of "offset": result = @[3, 4]
  of "reverse": result = @[5]
  of "uint32_singleton": result = @[1]
  of "broadcast": result = @[1, 3]
  of "scalar", "scalar_payload_f32", "scalar_payload_f64", "uint16_scalar",
      "uint32_scalar", "uint64_scalar": result = @[]
  of "empty_reduced": result = @[2, 0]
  of "empty_outer", "uint64_empty", "complex64_empty": result = @[0, 3]
  else: invalid("unknown aminmax input shape " & name)


proc aminmaxInputLayout(name: string): string =
  result = if name in ["transpose", "offset", "reverse", "broadcast"]:
    name
  else:
    "identity"


proc aminmaxInputPayload(name: string): tuple[key: string, data: JsonNode] =
  result.key = "words"
  case name
  of "bool": result.data = %*[0, 1, 0, 1, 0, 1]
  of "int8": result.data = %*[-128, 127, -1, 1, 2, 3]
  of "int16": result.data = %*[-32768, 32767, -1, 5]
  of "int32": result.data = %*[-2147483648, 2147483647, -1, 5]
  of "int64": result.data = %*[low(int64), high(int64), -2, 1, 4, -9]
  of "uint8": result.data = %*[0, 255, 1, 2, 128, 3]
  of "transpose":
    result.key = "values"
    result.data = %*[2.0, -3.0, 5.0, 7.0, 0.0, -1.0]
  of "offset": result.data = %*[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  of "reverse":
    result.key = "values"
    result.data = %*[1.0, -5.0, 2.0, 0.0, 4.0]
  of "broadcast": result.data = %*[2, 3, 0]
  of "scalar": result.data = %*[low(int64)]
  of "empty_reduced", "empty_outer", "uint64_empty", "complex64_empty":
    if name == "complex64_empty": result.key = "values"
    result.data = newJArray()
  of "zero_ties_f32":
    result.key = "bits"
    result.data = %*["0x80000000", "0x00000000", "0x80000000",
      "0x00000000", "0x80000000", "0x00000000"]
  of "zero_ties_f64":
    result.key = "bits"
    result.data = %*["0x8000000000000000", "0x0000000000000000",
      "0x8000000000000000", "0x0000000000000000", "0x8000000000000000",
      "0x0000000000000000"]
  of "payload_f32":
    result.key = "bits"
    result.data = %*["0x3f800000", "0x7fc12345", "0x3f800000", "0x3f800000",
      "0x7fc54321", "0x3f800000", "0x3f800000", "0x3f800000"]
  of "payload_f64":
    result.key = "bits"
    result.data = %*["0x3ff0000000000000", "0x7ff8123456789abc",
      "0x7ff876543210abcd", "0x3ff0000000000000"]
  of "scalar_payload_f32":
    result.key = "bits"
    result.data = %*["0x7fc12345"]
  of "scalar_payload_f64":
    result.key = "bits"
    result.data = %*["0x7ff8123456789abc"]
  of "infinities_f32":
    result.key = "bits"
    result.data = %*["0xff800000", "0x7f800000", "0x00000000", "0x7f800000",
      "0xff800000", "0x3f800000"]
  of "infinities_f64":
    result.key = "bits"
    result.data = %*["0xfff0000000000000", "0x7ff0000000000000",
      "0x0000000000000000", "0x7ff0000000000000", "0xfff0000000000000",
      "0x3ff0000000000000"]
  of "uint16_matrix": result.data = %*[0, 65535, 1, 2, 40000, 3]
  of "uint32_singleton": result.data = %*[4294967295]
  of "uint16_scalar": result.data = %*[65535]
  of "uint32_scalar": result.data = %*[4294967295]
  of "uint64_scalar": result.data = %*[-1]
  of "complex128_scalar":
    result.key = "values"
    result.data = %*[[1.0, -2.0]]
  else: invalid("unknown aminmax input payload " & name)


proc aminmaxRequestId(dimensions: JsonNode, keepdim: bool): string =
  let dimName = if dimensions.kind == JNull: "all" else: "axis_" & $dimensions.getInt
  result = dimName & (if keepdim: "_keep" else: "_drop")


proc aminmaxAxes(name: string): seq[int] =
  if name in ["reverse", "uint32_singleton", "scalar", "scalar_payload_f32",
      "scalar_payload_f64", "uint16_scalar", "uint32_scalar", "uint64_scalar",
      "complex128_scalar"]:
    result = @[0, -1, 1, -2]
  else:
    result = @[0, 1, -1, -2, 2, -3]


proc aminmaxRequests(name: string): JsonNode =
  result = newJArray()
  var dimensions = @[newJNull()]
  for axis in aminmaxAxes(name): dimensions.add newJInt(axis)
  for dimension in dimensions:
    for keepdim in [false, true]:
      var request = newJObject()
      request["id"] = %aminmaxRequestId(dimension, keepdim)
      request["dim"] = if dimension.kind == JNull: newJNull() else: newJInt(dimension.getInt)
      request["keepdim"] = %keepdim
      result.add request


proc aminmaxShapeNode(shape: seq[int]): JsonNode =
  result = newJArray()
  for dimension in shape: result.add newJInt(dimension)


proc expectAminmaxInputs(inputs: JsonNode) =
  expectExtremaFields(inputs, AminmaxInputNames, "aminmax inputs")
  for name in AminmaxInputNames:
    let spec = field(inputs, name)
    let payload = aminmaxInputPayload(name)
    expectExtremaFields(spec, ["dtype", "shape", "layout", payload.key, "requests"],
      "aminmax input " & name)
    require(field(spec, "dtype") == %aminmaxInputDtype(name),
      "aminmax dtype mismatch for " & name)
    require(field(spec, "shape") == aminmaxShapeNode(aminmaxInputShape(name)),
      "aminmax shape mismatch for " & name)
    require(field(spec, "layout") == %aminmaxInputLayout(name),
      "aminmax layout mismatch for " & name)
    require(field(spec, payload.key) == payload.data,
      "aminmax payload mismatch for " & name)
    require(field(spec, "requests") == aminmaxRequests(name),
      "aminmax requests mismatch for " & name)


proc aminmaxLogicalShape(name: string): seq[int] =
  let shape = aminmaxInputShape(name)
  case name
  of "transpose": result = @[shape[1], shape[0]]
  of "offset": result = @[shape[1], shape[0] - 1]
  of "broadcast": result = @[4, 3]
  else: result = shape


proc aminmaxExpectedError(dtype: string, shape: seq[int], dimensions: JsonNode): string =
  if dtype in ["complex64", "complex128"]:
    return "unsupported_dtype"
  let rank = shape.len
  if dimensions.kind == JNull:
    for size in shape:
      if size == 0: return "empty_reduction"
    if dtype in ["uint16", "uint32", "uint64"]: return "unsupported_dtype"
    return ""
  require(dimensions.kind == JInt, "aminmax dim must be null or an integer")
  let axis = dimensions.getInt
  let axisLimit = max(1, rank)
  if axis < -axisLimit or axis >= axisLimit: return "axis_out_of_range"
  if rank > 0:
    let normalized = if axis < 0: axis + rank else: axis
    if shape[normalized] == 0: return "empty_reduced_axis"
    if dtype in ["uint16", "uint32", "uint64"]: return "unsupported_dtype"


proc aminmaxOutputShape(shape: seq[int], dimensions: JsonNode,
                        keepdim: bool): seq[int] =
  if dimensions.kind == JNull:
    if keepdim:
      result = newSeq[int](shape.len)
      for index in 0 ..< result.len: result[index] = 1
    return
  if shape.len == 0: return @[]
  let axis = dimensions.getInt
  let normalized = if axis < 0: axis + shape.len else: axis
  if keepdim:
    result = newSeq[int](shape.len)
    for index, size in shape: result[index] = size
    result[normalized] = 1
  else:
    for index, size in shape:
      if index != normalized: result.add size


proc aminmaxHexBits(node: JsonNode, dtype, label: string): uint64 =
  require(node.kind == JString, label & " bit pattern must be a string")
  let value = node.getStr
  let digits = if dtype == "float32": 8 else: 16
  require(value.len == digits + 2 and value[0 .. 1] == "0x",
    label & " bit pattern has an invalid width or prefix")
  for index in 2 ..< value.len:
    var nibble = 0'u64
    case value[index]
    of '0' .. '9': nibble = uint64(ord(value[index]) - ord('0'))
    of 'a' .. 'f': nibble = uint64(ord(value[index]) - ord('a') + 10)
    else: require(false, label & " bit pattern is not lowercase hexadecimal")
    result = (result shl 4) or nibble


proc expectAminmaxFloatValue(valueNode, bitsNode: JsonNode,
                             dtype, label: string) =
  let bits = aminmaxHexBits(bitsNode, dtype, label)
  let decoded = if dtype == "float32": float64(cast[float32](uint32(bits)))
    else: cast[float64](bits)
  if valueNode.kind == JString:
    case valueNode.getStr
    of "nan": require(classify(decoded) == fcNan, label & " NaN bits mismatch")
    of "+inf": require(classify(decoded) == fcInf, label & " positive infinity bits mismatch")
    of "-inf": require(classify(decoded) == fcNegInf, label & " negative infinity bits mismatch")
    else: invalid(label & " has an invalid non-finite observation")
  else:
    require(valueNode.kind in {JInt, JFloat}, label & " float observation must be numeric")
    let observed = valueNode.getFloat
    let observedBits = if dtype == "float32":
      uint64(cast[uint32](observed.float32))
    else:
      cast[uint64](observed)
    require(bits == observedBits, label & " value and bit pattern disagree")


proc expectAminmaxTensor(node: JsonNode, shape: seq[int], dtype, label: string) =
  if dtype in ["float32", "float64"]:
    expectExtremaFields(node, ["shape", "numel", "dtype", "device", "values", "bits"], label)
    expectTensor(node, false, dtype)
    let count = extremaNumel(shape)
    require(field(node, "shape") == aminmaxShapeNode(shape) and
      field(node, "numel").kind == JInt and field(node, "numel").getInt == count and
      field(node, "device") == %"cpu", label & " metadata mismatch")
    let values = field(node, "values")
    let bits = field(node, "bits")
    require(values.kind == JArray and bits.kind == JArray and
      values.len == count and bits.len == count, label & " values/bits count mismatch")
    for index in 0 ..< count:
      expectAminmaxFloatValue(values[index], bits[index], dtype, label & "[" & $index & "]")
  else:
    expectIntegerOutputTensor(node, shape, dtype)


proc expectAminmaxExpected(expected, inputs: JsonNode) =
  expectExtremaFields(expected, AminmaxInputNames, "aminmax expected inputs")
  for name in AminmaxInputNames:
    let spec = field(inputs, name)
    let requests = field(spec, "requests")
    let byRequest = field(expected, name)
    var requestNames: seq[string]
    for request in requests: requestNames.add request["id"].getStr
    expectExtremaFields(byRequest, requestNames, "aminmax requests for " & name)
    let dtype = aminmaxInputDtype(name)
    let shape = aminmaxLogicalShape(name)
    for request in requests:
      let requestId = request["id"].getStr
      let observed = field(byRequest, requestId)
      let label = "aminmax " & name & "." & requestId
      let expectedError = aminmaxExpectedError(dtype, shape, request["dim"])
      if expectedError.len > 0:
        expectExtremaFields(observed, ["error"], label)
        require(field(observed, "error") == %expectedError, label & " error precedence mismatch")
      else:
        expectExtremaFields(observed, ["min", "max", "distinct"], label)
        let outputShape = aminmaxOutputShape(shape, request["dim"], request["keepdim"].getBool)
        expectAminmaxTensor(field(observed, "min"), outputShape, dtype, label & " min")
        expectAminmaxTensor(field(observed, "max"), outputShape, dtype, label & " max")
        require(field(observed, "distinct").kind == JBool and
          field(observed, "distinct").getBool, label & " tuple outputs must be independent")


const PairedExtremaGradientInputNames = [
  "ties", "transpose", "reverse", "broadcast", "nan_first", "zero_ties",
  "scalar", "empty", "cube", "offset", "infinities"]


proc pairedExtremaGradientInputSpec(name: string): JsonNode =
  case name
  of "ties":
    result = %*{
      "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
      "layout": "identity", "nonfinite": [], "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "transpose":
    result = %*{
      "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
      "layout": "transpose", "nonfinite": [], "axes": [0, -2],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "reverse":
    result = %*{
      "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
      "layout": "reverse", "nonfinite": [], "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "broadcast":
    result = %*{
      "x": {"shape": [1, 3], "values": [1, 1, 2]},
      "layout": "broadcast", "nonfinite": [], "axes": [1, -1, 0, -2],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "nan_first":
    result = %*{
      "x": {"shape": [2, 3], "values": [0, 0, 0, 0, -1, 2]},
      "layout": "identity",
      "nonfinite": [
        {"index": 1, "value": "nan"}, {"index": 2, "value": "nan"},
        {"index": 3, "value": "nan"}],
      "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "zero_ties":
    result = %*{
      "x": {"shape": [2, 3], "values": [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0]},
      "layout": "identity", "nonfinite": [], "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "scalar":
    result = %*{
      "x": {"shape": [], "values": [0]},
      "layout": "identity", "nonfinite": [{"index": 0, "value": "nan"}],
      "axes": [-1, 0],
      "seed_kinds": ["finite", "negative_zero", "nonfinite"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "empty":
    result = %*{
      "x": {"shape": [0, 3], "values": []},
      "layout": "identity", "nonfinite": [], "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "cube":
    result = %*{
      "x": {"shape": [2, 2, 3], "values": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]},
      "layout": "identity", "nonfinite": [],
      "axes": [1, -2, 0, -3, 2, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "offset":
    result = %*{
      "x": {"shape": [3, 4], "values": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]},
      "layout": "offset", "nonfinite": [], "axes": [0, -2],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  of "infinities":
    result = %*{
      "x": {"shape": [2, 4], "values": [0, 0, 0, 0, 0, 0, 0, 0]},
      "layout": "identity",
      "nonfinite": [
        {"index": 0, "value": "-inf"}, {"index": 1, "value": "-inf"},
        {"index": 3, "value": "+inf"}, {"index": 4, "value": "+inf"},
        {"index": 6, "value": "+inf"}, {"index": 7, "value": "-inf"}],
      "axes": [1, -1],
      "seed_kinds": ["finite", "negative_zero", "nonfinite", "strided"],
      "invalid_seed": {"kind": "shape_mismatch"}}
  else:
    invalid("unknown paired extrema gradient input " & name)


proc pairedExtremaGradientSourceShape(spec: JsonNode): seq[int] =
  for dimension in spec["x"]["shape"]: result.add dimension.getInt
  case spec["layout"].getStr
  of "transpose": swap(result[0], result[1])
  of "offset": result = @[result[1], result[0] - 1]
  of "broadcast": result = @[2, 3]
  else: discard


proc pairedExtremaGradientOutputShape(shape: seq[int], axis: int,
                                      keepdim: bool): seq[int] =
  let normalizedAxis = if shape.len == 0: 0
    elif axis < 0: axis + shape.len
    else: axis
  for dimension, size in shape:
    if dimension == normalizedAxis:
      if keepdim: result.add 1
    else:
      result.add size


proc expectPairedExtremaGradientInputs(inputs: JsonNode) =
  require(inputs.kind == JObject and inputs.len == PairedExtremaGradientInputNames.len,
    "paired extrema gradient inputs do not match required probe cases")
  for name in PairedExtremaGradientInputNames:
    let spec = field(inputs, name)
    let expected = pairedExtremaGradientInputSpec(name)
    let label = "paired extrema gradient input " & name
    expectExtremaFields(spec, ["x", "layout", "nonfinite", "axes", "seed_kinds",
      "invalid_seed"], label)
    expectInputTensor(field(spec, "x"))
    for key in ["x", "layout", "nonfinite", "axes", "seed_kinds", "invalid_seed"]:
      require(field(spec, key) == expected[key], label & " " & key & " mismatch")
    if name == "zero_ties":
      let values = field(field(spec, "x"), "values")
      for index in [0, 2, 4]:
        require(isNegativeZeroValue(values[index]), label & " lost a negative-zero source")
      for index in [1, 3, 5]:
        require(values[index].getFloat == 0.0 and not isNegativeZeroValue(values[index]),
          label & " changed a positive-zero source")


proc expectPairedExtremaGradientObservation(observed: JsonNode, sourceShape,
    outputShape: seq[int], axis: int, dtype, label: string, invalidSeed = false) =
  if invalidSeed:
    expectExtremaFields(observed, ["error"], label)
    require(text(field(observed, "error"), label & " error") == "seed_shape_mismatch",
      label & " invalid seed was not rejected for its shape")
    return

  expectExtremaFields(observed, ["values", "indices", "seed", "gradient",
    "indices_require_grad"], label)
  expectExtremaGradientTensor(field(observed, "values"), outputShape, dtype,
    label & " values", true)
  expectIntegerOutputTensor(field(observed, "indices"), outputShape, "int64")
  let limit = if sourceShape.len == 0: 1
    else: sourceShape[if axis < 0: axis + sourceShape.len else: axis]
  for index in field(observed, "indices")["values"]:
    require(index.getBiggestInt >= 0 and index.getBiggestInt < limit.int64,
      label & " index is outside its reduced dimension")
  require(field(observed, "indices_require_grad").kind == JBool and
    not field(observed, "indices_require_grad").getBool,
    label & " indices unexpectedly require gradients")
  expectExtremaGradientTensor(field(observed, "seed"), outputShape, dtype,
    label & " seed", true)
  expectExtremaGradientTensor(field(observed, "gradient"), sourceShape, dtype,
    label & " gradient", true)


proc expectPairedExtremaGradientExpected(expected, inputs: JsonNode) =
  expectExtremaFields(expected, ["float32", "float64"], "paired extrema gradient dtypes")
  for dtype in ["float32", "float64"]:
    let byCase = field(expected, dtype)
    expectExtremaFields(byCase, PairedExtremaGradientInputNames,
      "paired extrema gradient cases for " & dtype)
    for name in PairedExtremaGradientInputNames:
      let spec = inputs[name]
      let byRequest = field(byCase, name)
      let axes = field(spec, "axes")
      let seedKinds = field(spec, "seed_kinds")
      let requestCount = axes.len * 2 * 2 * (seedKinds.len + 1)
      let label = "paired extrema gradient " & dtype & "." & name
      require(byRequest.kind == JObject and byRequest.len == requestCount,
        label & " request observations are incomplete")
      let sourceShape = pairedExtremaGradientSourceShape(spec)
      for operation in ["min", "max"]:
        for axisNode in axes:
          let axis = axisNode.getInt
          for keepdim in [false, true]:
            let requestId = indexedExtremaRequestId(operation, axisNode, keepdim)
            let outputShape = pairedExtremaGradientOutputShape(sourceShape, axis, keepdim)
            for seedKindNode in seedKinds:
              let seedKind = seedKindNode.getStr
              expectPairedExtremaGradientObservation(
                field(byRequest, requestId & "_seed_" & seedKind), sourceShape,
                outputShape, axis, dtype, label & "." & requestId & " seed " & seedKind)
            expectPairedExtremaGradientObservation(
              field(byRequest, requestId & "_invalid_seed"), sourceShape,
              outputShape, axis, dtype, label & "." & requestId & " invalid seed",
              invalidSeed = true)

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
    of "tensor.complex_scans":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "complex scan fixtures require 1e-6 tolerances")
      require(inputs.kind == JObject and inputs.len == ComplexScanInputNames.len,
        "complex scan inputs do not match required cases")
      for name in ComplexScanInputNames:
        require(inputs.hasKey(name), "missing complex scan input " & name)
        expectComplexScanInput(inputs[name], name)
    of "tensor.complex_scan_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "complex scan dtype fixtures require 1e-6 tolerances")
      expectComplexScanDtypeInputs(inputs)
    of "tensor.complex_mean_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "complex mean dtype fixtures require 1e-6 tolerances")
      expectComplexMeanDtypeInputs(inputs)
    of "tensor.complex_sum_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "complex sum dtype fixtures require 1e-6 tolerances")
      expectComplexSumDtypeInputs(inputs)
    of "tensor.complex_prod_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "complex product dtype fixtures require 1e-6 tolerances")
      expectComplexProdDtypeInputs(inputs)
    of "autograd.extrema":
      require(c["atol"].getFloat == 1e-7 and c["rtol"].getFloat == 1e-7,
        "extrema gradient fixtures require 1e-7 tolerances")
      expectExtremaGradientInputs(inputs)
    of "autograd.paired_extrema":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "paired extrema gradient fixtures require exact-value tolerances")
      expectPairedExtremaGradientInputs(inputs)
    of "tensor.extrema":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "extrema fixtures require exact-value tolerances")
      expectExtremaInputs(inputs)
    of "tensor.indexed_extrema":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "indexed extrema fixtures require exact-value tolerances")
      expectIndexedExtremaInputs(inputs)
    of "tensor.paired_extrema":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "paired extrema fixtures require exact-value tolerances")
      expectPairedExtremaInputs(inputs)
    of "tensor.aminmax":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "aminmax fixtures require exact-value tolerances")
      expectAminmaxInputs(inputs)

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
    of "tensor.integer_output_dtype":
      require(c["atol"].getFloat == 0 and c["rtol"].getFloat == 0,
        "integer output dtype fixtures require exact-value tolerances")
      expectIntegerOutputInputs(inputs)
    of "tensor.mean_dtype", "tensor.sum_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "real dtype fixtures require 1e-6 tolerances")
      expectRealDtypeInputs(inputs)
    of "tensor.prod_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "real dtype product fixtures require 1e-6 tolerances")
      expectRealDtypeInputs(inputs, includeProductExtras = true)
    of "tensor.cumsum_dtype", "tensor.cumprod_dtype":
      require(c["atol"].getFloat == 1e-6 and c["rtol"].getFloat == 1e-6,
        "real dtype scan fixtures require 1e-6 tolerances")
      expectRealDtypeInputs(inputs, includeScanExtras = true)
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
    of "tensor.complex_scans":
      expectComplexScanExpected(expected, inputs)
    of "tensor.complex_scan_dtype":
      expectComplexScanDtypeExpected(expected, inputs)
    of "tensor.complex_mean_dtype":
      expectComplexMeanDtypeExpected(expected, inputs)
    of "tensor.complex_sum_dtype":
      expectComplexSumDtypeExpected(expected, inputs)
    of "tensor.complex_prod_dtype":
      expectComplexProdDtypeExpected(expected, inputs)
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
    of "autograd.extrema":
      expectExtremaGradientExpected(expected, inputs)
    of "autograd.paired_extrema":
      expectPairedExtremaGradientExpected(expected, inputs)
    of "tensor.extrema":
      expectExtremaExpected(expected, inputs)
    of "tensor.indexed_extrema":
      expectIndexedExtremaExpected(expected, inputs)
    of "tensor.paired_extrema":
      expectPairedExtremaExpected(expected, inputs)
    of "tensor.aminmax":
      expectAminmaxExpected(expected, inputs)
    of "tensor.integer_output_dtype":
      expectIntegerOutputExpected(expected, inputs)
    of "tensor.float64_reductions":
      expectFloat64Reductions(expected, inputs)
    of "tensor.mean_dtype", "tensor.sum_dtype":
      expectRealDtypeExpected(expected, inputs, id == "tensor.mean_dtype")
    of "tensor.prod_dtype":
      expectRealDtypeExpected(expected, inputs, false)
    of "tensor.cumsum_dtype", "tensor.cumprod_dtype":
      expectRealDtypeScanExpected(expected, inputs)
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
