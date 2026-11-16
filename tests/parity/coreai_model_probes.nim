# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes SDK model metadata, actual inference outputs and retained descriptor diagnostics.
import std/[algorithm, json, options, os, tempfiles]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./parity_support

proc shapeNode(shape: Metadata): JsonNode =
  result = newJArray()
  for dim in shape: result.add %dim

proc arrayDescriptorNode(desc: CoreAIArrayDescriptor): JsonNode =
  %*{"name": desc.name, "dtype": desc.dtype, "shape": shapeNode(desc.shape),
    "dynamic_shape": desc.dynamicShape}

proc functionDescriptorNode(desc: CoreAIFunctionDescriptor): JsonNode =
  result = %*{"name": desc.name, "inputs": [], "outputs": [], "states": []}
  for item in desc.inputs: result["inputs"].add arrayDescriptorNode(item)
  for item in desc.outputs: result["outputs"].add arrayDescriptorNode(item)
  for item in desc.states: result["states"].add arrayDescriptorNode(item)

proc observeCoreAI*(array: CoreAIArray): JsonNode =
  let desc = array.descriptor
  let host = array.cpu(float32)
  result = %*{"shape": shapeNode(desc.shape), "dtype": desc.dtype,
    "backend": "coreai", "values": []}
  for _, value in enumerate(host): result["values"].add %value

proc coreAIEvidence*(job: CoreAIJob): JsonNode =
  let native = job.coreAIJobEvidence
  result = %*{"invocation_id": native.invocationId,
    "kind": (if native.kind == 3: "coreai_inference" else: "device_storage_control"),
    "completed": native.completed, "requested_policy": $native.requestedPolicy,
    "available_units_mask": native.availableMask, "allowed_units_mask": native.allowedMask,
    "preferred_unit": native.preferredUnit, "actual_placement": nil,
    "asset_digest": native.assetDigest, "function": native.functionName,
    "architecture": native.architecture, "bridge_host_copy_bytes": native.hostCopyBytes,
    "bridge_materialization_bytes": native.materializationBytes,
    "bridge_cpu_fallback_count": native.cpuFallbackCount,
    "borrowed_input_count": native.borrowedInputCount, "producer_count": native.producerCount}
  if native.actualPlacement >= 0: result["actual_placement"] = %native.actualPlacement

proc identityDescriptor(): JsonNode =
  %*{"name": "identity",
    "inputs": [{"name": "x", "dtype": "float32", "shape": [2], "dynamic_shape": false}],
    "outputs": [{"name": "y", "dtype": "float32", "shape": [2], "dynamic_shape": false}],
    "states": []}

proc coreAIModelObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.model_functions", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let manifest = coreAIManifestObservation()
    result.actual["source_asset"] = manifest
    result.differences.add compareObservations(%true, manifest["matches_manifest"], 0, 0)
    var invocations = newJArray()
    for index, path in coreAIModelPaths():
      let model = openCoreAIModel(path)
      var names = model.functionNames
      names.sort()
      let function = model.loadFunction("identity").get
      let job = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1.25'f32, -2].toTensor))])
      let outputs = job.wait()
      var observedOutputs = newJArray()
      for binding in outputs:
        observedOutputs.add %*{"name": binding.name, "array": observeCoreAI(binding.array)}
      let actual = %*{"function_names": names, "descriptor": functionDescriptorNode(function.descriptor),
        "outputs": observedOutputs, "absent_function": model.loadFunction("absent_function").isNone}
      let key = if index == 0: "source" else: "compiled"
      result.actual[key] = actual
      result.differences.add compareObservations(%*{"function_names": ["identity", "identity_matrix"],
        "descriptor": identityDescriptor(), "outputs": [{"name": "y", "array": {
          "shape": [2], "dtype": "float32", "backend": "coreai", "values": [1.25, -2.0]}}],
        "absent_function": true}, actual, 0, 0)
      result.evidence = coreAIEvidence(job)
      let actualHash = hashCoreAIAsset(path).digest
      result.differences.add compareObservations(%actualHash, result.evidence["asset_digest"], 0, 0)
      invocations.add result.evidence
    let (file, path) = createTempFile("arraymancer-corrupt-model-", ".aimodel")
    file.write("not a Core AI model")
    file.close()
    defer: removeFile(path)
    var invalidModel = false
    var diagnostics = newJArray()
    try: discard openCoreAIModel(path)
    except CoreAIError as error:
      invalidModel = true
      diagnostics.add %*{"category": "model_load", "message": error.msg}
    result.actual["invalid_model_errors"] = %invalidModel
    result.differences.add compareObservations(%true, %invalidModel, 0, 0)
    result.evidence = result.evidence.copy()
    result.evidence["invocations"] = invocations
    result.evidence["native_diagnostics"] = diagnostics
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}

proc coreAIDescriptorObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.ndarray_descriptors", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let model = openCoreAIModel(coreAISourceAsset)
    let function = model.loadFunction("identity").get
    let job = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1.25'f32, -2].toTensor))])
    let outputs = job.wait()
    var actualOutputs = newJArray()
    for binding in outputs: actualOutputs.add %*{"name": binding.name, "array": observeCoreAI(binding.array)}
    result.actual["outputs"] = actualOutputs
    result.differences.add compareObservations(%*[{"name": "y", "array": {
      "shape": [2], "dtype": "float32", "backend": "coreai", "values": [1.25, -2.0]}}], actualOutputs, 0, 0)
    var diagnostics = newJArray()
    let badArrays = [("wrong_dtype", toCoreAI([1'i32, 2].toTensor)),
      ("wrong_shape", toCoreAI([1'f32, 2, 3].toTensor)),
      ("wrong_rank", toCoreAI([[1'f32, 2]].toTensor))]
    var errors = newJObject()
    for item in badArrays:
      var rejected = false
      try: discard function.infer([CoreAIBinding(name: "x", array: item[1])])
      except CoreAIDescriptorError as error:
        rejected = true
        diagnostics.add %*{"category": "descriptor", "scenario": item[0], "message": error.msg}
      errors[item[0]] = %rejected
    errors["input_dtype_after"] = %badArrays[0][1].descriptor.dtype
    result.actual["strict_validation"] = errors
    result.differences.add compareObservations(%*{"wrong_dtype": true, "wrong_shape": true,
      "wrong_rank": true, "input_dtype_after": "int32"}, errors, 0, 0)
    let dynamic = model.loadFunction("identity_matrix").get.descriptor
    result.actual["dynamic_descriptor"] = functionDescriptorNode(dynamic)
    result.actual["unresolved_minimum_bytes"] = %dynamic.inputs[0].minimumBytes
    result.differences.add compareObservations(%*{"name": "identity_matrix",
      "inputs": [{"name": "x", "dtype": "float32", "shape": [-1, 3], "dynamic_shape": true}],
      "outputs": [{"name": "y", "dtype": "float32", "shape": [-1, 3], "dynamic_shape": true}],
      "states": []}, result.actual["dynamic_descriptor"], 0, 0)
    result.differences.add compareObservations(%0, result.actual["unresolved_minimum_bytes"], 0, 0)
    result.evidence = coreAIEvidence(job)
    result.evidence["native_diagnostics"] = diagnostics
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
