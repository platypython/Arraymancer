# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes actual SDK layouts, shared-buffer identity and producer-scoped borrow exclusion.
import std/[json, options]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./[parity_support, coreai_model_probes]

proc dimensions(values: Metadata): JsonNode =
  result = newJArray()
  for value in values: result.add %value

proc inferenceError(error: ref CoreAIError): JsonNode =
  let category = case error.nativeCode
    of 102: "descriptor"
    of 104: "timeout"
    of 105: "buffer_busy"
    else: "native_error"
  %*{"category": category, "native_code": error.nativeCode, "message": error.msg}

proc coreAILayoutObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.strided_layout", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let dynamic = function.descriptor.inputs[0]
    let resolved = resolveDescriptor(dynamic, [2, 3].toMetadata)
    let extent = resolved.strides[0] + 2 * resolved.strides[1] + 1
    result.actual["resolved"] = %*{"shape": dimensions(resolved.shape),
      "preferred_strides": dimensions(resolved.strides), "minimum_bytes": resolved.minimumBytes,
      "resolved": resolved.resolved, "dynamic_shape": resolved.dynamicShape}
    let resolutionValid = resolved.shape == [2, 3].toMetadata and resolved.resolved and
      not resolved.dynamicShape and resolved.strides[0] > 0 and resolved.strides[1] > 0 and
      resolved.minimumBytes >= uint64(extent * sizeof(float32))
    result.differences.add compareObservations(%true, %resolutionValid, 0, 0)
    let storage = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let strided = storage.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1)
    let borrowed = coreAIView(strided)
    let inputs = [toCoreAI([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor), borrowed,
      toCoreAI([[1'f32, 2, 3], [4'f32, 5, 6], [7'f32, 8, 9]].toTensor)]
    var cases = newJArray()
    var invocations = newJArray()
    var borrowedJob: CoreAIJob
    for index, input in inputs:
      let job = function.submitInference([CoreAIBinding(name: "x", array: input)],
        states = [], outputViews = [])
      let outputs = job.wait()
      var observed = newJArray()
      for binding in outputs:
        let array = observeCoreAI(binding.array)
        let layout = binding.array.descriptor
        observed.add %*{"name": binding.name, "array": array,
          "strides": dimensions(layout.strides), "output_bridge_copy_bytes": layout.hostCopyBytes}
      let rows = if index == 2: 3 else: 2
      let values = if index == 2: %*[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        else: %*[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      let expected = %*[{"name": "y", "array": {"shape": [rows, 3], "dtype": "float32",
        "backend": "coreai", "values": values}, "strides": [3, 1],
        "output_bridge_copy_bytes": rows * 3 * sizeof(float32)}]
      result.differences.add compareObservations(expected, observed, 0, 0)
      let info = input.descriptor
      let expectedCopies = if index == 1: 0 else: rows * 3 * sizeof(float32)
      result.differences.add compareObservations(%expectedCopies, %info.hostCopyBytes, 0, 0)
      cases.add %*{"case": (if index == 0: "contiguous" elif index == 1: "strided" else: "rows3"),
        "outputs": observed, "input_bridge_copy_bytes": info.hostCopyBytes, "borrowed": info.borrowed}
      invocations.add coreAIEvidence(job)
      if index == 1: borrowedJob = job
    result.actual["cases"] = cases
    let host = storage.cpu
    let guards = %*[host[0], host[7]]
    result.actual["guards"] = guards
    result.differences.add compareObservations(%*[99.0, 99.0], guards, 0, 0)
    result.evidence = coreAIEvidence(borrowedJob)
    result.evidence["invocations"] = invocations
    result.evidence["sdk_internal_copy_bytes"] = newJNull()
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CoreAIError as error:
    result.error = inferenceError(error)
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}

proc coreAIInteropObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.metal_buffer_interop", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let source = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let ones = newTensorWith[float32]([8], 1'f32).metal
    let producer = newMetalStream()
    let unrelated = newMetalStream()
    let gate = newMetalEvent()
    let otherGate = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      signalMetalEvent(otherGate, 1)
      metalSynchronize(producer)
      metalSynchronize(unrelated)
    wait(producer, gate, 1)
    wait(unrelated, otherGate, 1)
    var pending: MetalTensor[float32]
    withMetalStream(producer): pending = source + 0'f32
    let view = coreAIView(pending.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1))
    let info = view.descriptor
    let originalIdentity = originalMetalBufferIdentity(pending)
    result.actual["input"] = %*{"shape": dimensions(info.shape), "strides": dimensions(info.strides),
      "byte_offset": info.byteOffset, "borrowed": info.borrowed,
      "buffer_identity": $info.bufferIdentity, "original_buffer_identity": $originalIdentity,
      "input_bridge_copy_bytes": info.hostCopyBytes}
    let job = function.submitInference([CoreAIBinding(name: "x", array: view)],
      states = [], outputViews = [])
    var scalarRejected, mathRejected, pendingWait: bool
    var diagnostics = newJArray()
    try: pending[1] = 100'f32
    except MetalBufferBusyError as error:
      scalarRejected = true
      diagnostics.add %*{"category": "buffer_busy", "operation": "scalar_write", "message": error.msg}
    try: pending += ones
    except MetalBufferBusyError as error:
      mathRejected = true
      diagnostics.add %*{"category": "buffer_busy", "operation": "inplace_math", "message": error.msg}
    try: discard job.wait(timeoutMs = 0)
    except CoreAITimeoutError as error:
      pendingWait = true
      diagnostics.add inferenceError(error)
    signalMetalEvent(gate, 1)
    released = true
    let outputs = job.wait()
    var observed = newJArray()
    for binding in outputs: observed.add %*{"name": binding.name, "array": observeCoreAI(binding.array),
      "strides": dimensions(binding.array.descriptor.strides)}
    result.actual["outputs"] = observed
    result.differences.add compareObservations(%*[{"name": "y", "array": {"shape": [2, 3],
      "dtype": "float32", "backend": "coreai", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
      "strides": [3, 1]}], observed, 0, 0)
    let host = pending.cpu
    result.evidence = coreAIEvidence(job)
    let facts = %*{"same_buffer": info.bufferIdentity == originalIdentity,
      "byte_offset": info.byteOffset, "borrowed": info.borrowed,
      "scalar_write_rejected": scalarRejected, "math_write_rejected": mathRejected,
      "wait_was_pending": pendingWait, "unrelated_queue_pending": not metalStreamIsComplete(unrelated),
      "guards": [host[0], host[7]], "input_bridge_copy_bytes": view.descriptor.hostCopyBytes,
      "borrowed_input_count": result.evidence["borrowed_input_count"],
      "producer_count": result.evidence["producer_count"]}
    result.actual["invariants"] = facts
    result.differences.add compareObservations(%*{"same_buffer": true, "byte_offset": 4,
      "borrowed": true, "scalar_write_rejected": true, "math_write_rejected": true,
      "wait_was_pending": true, "unrelated_queue_pending": true, "guards": [99.0, 99.0],
      "input_bridge_copy_bytes": 0, "borrowed_input_count": 1, "producer_count": 1}, facts, 0, 0)
    pending[1] = 10'f32
    result.actual["post_completion_mutation"] = %pending.cpu[1]
    result.differences.add compareObservations(%10.0, result.actual["post_completion_mutation"], 0, 0)
    result.evidence["native_diagnostics"] = diagnostics
    result.evidence["sdk_internal_copy_bytes"] = newJNull()
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CoreAIError as error:
    result.error = inferenceError(error)
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
