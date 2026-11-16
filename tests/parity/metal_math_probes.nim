# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Compares public GPU arithmetic to separately generated genuine MPS observations.
import std/[json, math]
import ../../src/arraymancer
import ./parity_support

proc metalInput*(node: JsonNode): MetalTensor[float32] =
  var shape: Metadata
  shape.len = node["shape"].len
  for i in 0 ..< shape.len: shape[i] = node["shape"][i].getInt
  var cpu = newTensorUninit[float32](shape)
  let values = cast[ptr UncheckedArray[float32]](cpu.get_offset_ptr)
  for i in 0 ..< cpu.size: values[i] = float32(node["values"][i].getFloat)
  cpu.metal

proc observeMetal*(tensor: MetalTensor[float32]): JsonNode =
  var shape = newJArray()
  for dim in tensor.shape: shape.add(%dim)
  result = %*{"shape": shape, "numel": tensor.size, "dtype": "float32", "device": "metal", "values": []}
  let cpu = tensor.cpu
  let values = cast[ptr UncheckedArray[float32]](cpu.get_offset_ptr)
  for i in 0 ..< tensor.size:
    case values[i].classify
    of fcNan: result["values"].add(%"nan")
    of fcInf: result["values"].add(%"+inf")
    of fcNegInf: result["values"].add(%"-inf")
    else: result["values"].add(%values[i])

proc metalEvidence*(tensor: MetalTensor[float32]): JsonNode =
  var native: AMEvidence
  checkMetal metalApi().evidence(tensor.storage.owner.handle, addr native)
  let kernel = case native.kernelBits
    of 101..109: "math_elementwise"
    of 120: "math_matmul"
    of 121: "math_reduce"
    of 122: "math_axis_sum"
    of 123: "math_squared_difference"
    of 124: "math_mse_backward"
    else: ""
  result = %*{"invocation_id": native.invocationId,
    "kind": (if native.kind == 2: "gpu_command" else: "device_storage_control"),
    "completed": native.completed == 1, "device_registry_id": $native.registryId,
    "kernel_labels": [kernel], "transfer_bytes": native.transferBytes,
    "materialization_bytes": native.materializationBytes, "cpu_fallback_count": native.cpuFallbackCount}

proc metalExpected*(expected: JsonNode): JsonNode =
  ## Normalize only the reference backend label; actual observations are untouched.
  result = parseJson($expected)
  proc relabel(node: JsonNode) =
    if node.kind == JObject:
      if node.hasKey("device") and node["device"].getStr in ["mps", "mps:0"]: node["device"] = %"metal"
      for key, child in node: relabel(child)
    elif node.kind == JArray:
      for child in node: relabel(child)
  relabel(result)

proc metalMathObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.tensor_math", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let fixtures = loadMetalFixtures()
    var invocations = newJArray()
    for item in fixtures["cases"]:
      let id = item["id"].getStr
      if id notin ["tensor.broadcast_same_rank", "tensor.broadcast_leading_dims", "tensor.matmul_2d"]: continue
      let a = metalInput(item["inputs"]["a"])
      let b = metalInput(item["inputs"]["b"])
      let output = if id == "tensor.matmul_2d": a * b else: a +. b
      let actual = %*{"result": observeMetal(output)}
      result.actual[id] = actual
      let differences = compareObservations(metalExpected(item["expected"]), actual,
        item["atol"].getFloat, item["rtol"].getFloat)
      for difference in differences:
        result.differences.add Difference(path: "/" & id & difference.path,
          actual: difference.actual, expected: difference.expected)
      result.evidence = metalEvidence(output)
      invocations.add(result.evidence)
    let scalar = newTensorWith[float32]([], 2'f32).metal + 1'f32
    let empty = newMetalTensorUninit[float32](2, 0, 3) + 1'f32
    let actualMetadata = %*{"scalar": observeMetal(scalar), "empty": observeMetal(empty)}
    result.actual["metadata"] = actualMetadata
    let metadataDifferences = compareObservations(%*{
      "scalar": {"shape": [], "numel": 1, "dtype": "float32", "device": "metal", "values": [3.0]},
      "empty": {"shape": [2, 0, 3], "numel": 0, "dtype": "float32", "device": "metal", "values": []}}, actualMetadata, 0, 0)
    result.differences.add(metadataDifferences)
    result.evidence = parseJson($result.evidence)
    result.evidence["invocations"] = invocations
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
